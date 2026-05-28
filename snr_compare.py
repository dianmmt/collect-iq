"""
IQ Noise Floor Adjuster - Phương án A
======================================
- Ngưỡng dùng P95/P5 thay cho max/min tuyệt đối
- Hysteresis để tránh artifact tại biên
- Scale theo block thay vì từng điểm đơn lẻ
- Tùy chọn thêm AWGN Gaussian thuần túy hoặc nhân hệ số
"""

import numpy as np
from pathlib import Path
import argparse
import json


# ─────────────────────────────────────────────
# 1. I/O
# ─────────────────────────────────────────────

def load_iq(path: str) -> np.ndarray:
    """Đọc file bin float32 interleaved IQ → complex array."""
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 2 != 0:
        raw = raw[:-1]  # bỏ byte lẻ nếu có
    iq = raw[0::2] + 1j * raw[1::2]
    return iq


def save_iq(path: str, iq: np.ndarray):
    """Ghi complex array → float32 interleaved IQ."""
    out = np.empty(iq.size * 2, dtype=np.float32)
    out[0::2] = iq.real
    out[1::2] = iq.imag
    out.tofile(path)


# ─────────────────────────────────────────────
# 2. Ngưỡng P95/P5
# ─────────────────────────────────────────────

def compute_threshold(amplitude: np.ndarray,
                      high_pct: float = 95.0,
                      low_pct: float = 5.0) -> tuple[float, float, float]:
    """
    Tính ngưỡng từ P95 và P5 của biên độ.

    Returns:
        threshold : ngưỡng phân loại U / nhiễu
        p_high    : giá trị percentile cao (đại diện đỉnh tín hiệu U)
        p_low     : giá trị percentile thấp (đại diện sàn nhiễu)
    """
    p_high = np.percentile(amplitude, high_pct)
    p_low  = np.percentile(amplitude, low_pct)
    threshold = (p_high + p_low) / 2.0
    return threshold, p_high, p_low


# ─────────────────────────────────────────────
# 3. Phân loại với Hysteresis
# ─────────────────────────────────────────────

def classify_hysteresis(amplitude: np.ndarray,
                        threshold: float,
                        hysteresis_ratio: float = 0.05) -> np.ndarray:
    """
    Phân loại từng mẫu với hysteresis để tránh chatter tại biên.

    Vùng hysteresis = threshold ± (threshold * hysteresis_ratio)
      - Trên upper_band : chắc chắn là tín hiệu U  (is_signal = True)
      - Dưới lower_band : chắc chắn là nhiễu        (is_signal = False)
      - Trong dải giữa  : giữ nguyên trạng thái trước (sticky)

    Returns:
        is_signal : bool array, True = tín hiệu U
    """
    upper_band = threshold * (1 + hysteresis_ratio)
    lower_band = threshold * (1 - hysteresis_ratio)

    is_signal = np.zeros(len(amplitude), dtype=bool)
    current_state = False  # khởi đầu: coi là nhiễu

    for i, amp in enumerate(amplitude):
        if amp >= upper_band:
            current_state = True
        elif amp <= lower_band:
            current_state = False
        # else: giữ nguyên current_state (hysteresis)
        is_signal[i] = current_state

    return is_signal


# ─────────────────────────────────────────────
# 4. Block-based smoothing (tùy chọn)
# ─────────────────────────────────────────────

def smooth_classification(is_signal: np.ndarray,
                          block_size: int = 64,
                          signal_ratio: float = 0.5) -> np.ndarray:
    """
    Xử lý theo block để tránh artifact điểm đơn lẻ bị misclassify.
    Nếu trong 1 block có >= signal_ratio mẫu là U → cả block là U.

    Returns:
        is_signal_smooth : bool array sau khi làm mịn
    """
    n = len(is_signal)
    is_signal_smooth = is_signal.copy()

    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        block = is_signal[start:end]
        if block.mean() >= signal_ratio:
            is_signal_smooth[start:end] = True
        else:
            is_signal_smooth[start:end] = False

    return is_signal_smooth


# ─────────────────────────────────────────────
# 5. Điều chỉnh nhiễu
# ─────────────────────────────────────────────

def adjust_noise(iq: np.ndarray,
                 is_signal: np.ndarray,
                 db_adjust: float,
                 mode: str = "scale") -> np.ndarray:
    """
    Điều chỉnh vùng nhiễu theo dB.

    Parameters:
        iq         : complex IQ array gốc
        is_signal  : mask True = giữ nguyên (tín hiệu U)
        db_adjust  : số dB muốn tăng (+) hoặc giảm (-)
        mode       : "scale"  → nhân hệ số biên độ (điều chỉnh nhiễu hiện có)
                     "awgn"   → inject thêm AWGN Gaussian độc lập
                                (chỉ có ý nghĩa khi db_adjust > 0)

    Returns:
        iq_out : IQ array đã điều chỉnh
    """
    scale = 10 ** (db_adjust / 20.0)
    iq_out = iq.copy()
    noise_mask = ~is_signal

    if mode == "scale":
        iq_out[noise_mask] *= scale

    elif mode == "awgn":
        # Ước lượng công suất nhiễu hiện tại
        noise_power = np.mean(np.abs(iq[noise_mask]) ** 2)
        # Công suất AWGN cần thêm để đạt mức tăng db_adjust dB
        # P_new = P_noise * scale²  →  P_add = P_new - P_noise
        p_add = noise_power * (scale ** 2 - 1)
        if p_add > 0:
            sigma = np.sqrt(p_add / 2)
            n_noise = noise_mask.sum()
            awgn = sigma * (np.random.randn(n_noise) + 1j * np.random.randn(n_noise))
            iq_out[noise_mask] += awgn
        else:
            # db_adjust âm: scale xuống nhiễu hiện có
            iq_out[noise_mask] *= scale

    return iq_out


# ─────────────────────────────────────────────
# 6. Stats & report
# ─────────────────────────────────────────────

def compute_stats(amplitude: np.ndarray,
                  is_signal: np.ndarray,
                  threshold: float,
                  p_high: float,
                  p_low: float) -> dict:
    n_total  = len(amplitude)
    n_signal = is_signal.sum()
    n_noise  = n_total - n_signal

    signal_amp = amplitude[is_signal]
    noise_amp  = amplitude[~is_signal]

    def safe_mean(arr): return float(np.mean(arr)) if arr.size else 0.0
    def safe_db(arr):
        m = safe_mean(arr)
        return float(20 * np.log10(m)) if m > 0 else -np.inf

    return {
        "n_total"        : int(n_total),
        "n_signal"       : int(n_signal),
        "n_noise"        : int(n_noise),
        "signal_ratio_pct": round(100 * n_signal / n_total, 2),
        "threshold"      : round(float(threshold), 6),
        "p95"            : round(float(p_high), 6),
        "p5"             : round(float(p_low), 6),
        "signal_mean_amp": round(safe_mean(signal_amp), 6),
        "noise_mean_amp" : round(safe_mean(noise_amp), 6),
        "signal_mean_dB" : round(safe_db(signal_amp), 2),
        "noise_mean_dB"  : round(safe_db(noise_amp), 2),
        "snr_dB"         : round(safe_db(signal_amp) - safe_db(noise_amp), 2),
    }


# ─────────────────────────────────────────────
# 7. Pipeline chính
# ─────────────────────────────────────────────

def process(input_path: str,
            output_path: str,
            db_adjust: float,
            mode: str = "scale",
            high_pct: float = 95.0,
            low_pct: float = 5.0,
            hysteresis_ratio: float = 0.05,
            block_size: int = 64,
            verbose: bool = True) -> dict:

    # --- Load ---
    iq = load_iq(input_path)
    amplitude = np.abs(iq)

    # --- Threshold ---
    threshold, p_high, p_low = compute_threshold(amplitude, high_pct, low_pct)

    # --- Phân loại với hysteresis ---
    is_signal = classify_hysteresis(amplitude, threshold, hysteresis_ratio)

    # --- Làm mịn theo block ---
    is_signal = smooth_classification(is_signal, block_size)

    # --- Stats trước khi chỉnh ---
    stats_before = compute_stats(amplitude, is_signal, threshold, p_high, p_low)

    # --- Điều chỉnh nhiễu ---
    iq_out = adjust_noise(iq, is_signal, db_adjust, mode)

    # --- Stats sau khi chỉnh ---
    amp_out = np.abs(iq_out)
    stats_after = compute_stats(amp_out, is_signal, threshold, p_high, p_low)

    # --- Lưu output ---
    save_iq(output_path, iq_out)

    report = {
        "input"       : input_path,
        "output"      : output_path,
        "db_adjust"   : db_adjust,
        "mode"        : mode,
        "params"      : {
            "high_pct"        : high_pct,
            "low_pct"         : low_pct,
            "hysteresis_ratio": hysteresis_ratio,
            "block_size"      : block_size,
        },
        "before"      : stats_before,
        "after"       : stats_after,
    }

    if verbose:
        print(json.dumps(report, indent=2, ensure_ascii=False))

    return report


# ─────────────────────────────────────────────
# 8. CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Điều chỉnh nền nhiễu trong file IQ (float32 bin)"
    )
    parser.add_argument("input",  help="File IQ đầu vào (.bin float32)")
    parser.add_argument("output", help="File IQ đầu ra (.bin float32)")
    parser.add_argument("db",     type=float,
                        help="Điều chỉnh dB (+tăng / -giảm nền nhiễu)")
    parser.add_argument("--mode", choices=["scale", "awgn"], default="awgn",
                        help="scale: nhân hệ số | awgn: inject nhiễu Gaussian (mặc định: scale)")
    parser.add_argument("--high-pct",  type=float, default=95.0,
                        help="Percentile cao cho ngưỡng (mặc định: 95)")
    parser.add_argument("--low-pct",   type=float, default=5.0,
                        help="Percentile thấp cho ngưỡng (mặc định: 5)")
    parser.add_argument("--hysteresis", type=float, default=0.05,
                        help="Tỷ lệ hysteresis quanh ngưỡng (mặc định: 0.05 = 5%%)")
    parser.add_argument("--block-size", type=int, default=64,
                        help="Kích thước block làm mịn phân loại (mặc định: 64)")

    args = parser.parse_args()

    process(
        input_path      = args.input,
        output_path     = args.output,
        db_adjust       = args.db,
        mode            = args.mode,
        high_pct        = args.high_pct,
        low_pct         = args.low_pct,
        hysteresis_ratio= args.hysteresis,
        block_size      = args.block_size,
    )
