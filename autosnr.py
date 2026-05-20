"""
amplitude_iq_v3.py — Tính biên độ tín hiệu và nhiễu từ file IQ float32
=======================================================================
Hỗ trợ 2 chế độ tìm ngưỡng:
  - "auto"   : Tự động tìm threshold từ dữ liệu (khuyên dùng lần đầu)
  - "manual" : Người dùng tự đặt MANUAL_FACTOR
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# ══════════════════════════════════════════════════════════════════
# THAM SỐ ĐẦU VÀO  ←  chỉnh tại đây
# ══════════════════════════════════════════════════════════════════
input_file = "signal/ktc-50db.bin"
fs         = 50_000_000.0   # Sample rate: 50 MHz (USRP B205 mini)
nperseg    = 1024           # Segment Welch

# ── Chọn chế độ threshold ──────────────────────────────────────
THRESHOLD_MODE = "auto"     # "auto" hoặc "manual"

# Dùng khi THRESHOLD_MODE = "auto"
# Tỷ lệ tối đa bin tín hiệu so với tổng băng thông (%)
# Tăng nếu tín hiệu wideband, giảm nếu muốn chặt hơn
MAX_SIGNAL_PCT = 30.0

# Dùng khi THRESHOLD_MODE = "manual"
# threshold = noise_floor × MANUAL_FACTOR
# Gợi ý: chạy "auto" trước → xem "Auto factor" in ra → dùng giá trị đó
MANUAL_FACTOR = 10.0


# ══════════════════════════════════════════════════════════════════
# ĐỌC FILE IQ
# ══════════════════════════════════════════════════════════════════
def load_iq(filepath: str) -> np.ndarray:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Không tìm thấy file: {filepath}")
    raw = np.fromfile(filepath, dtype=np.float32)
    if raw.size == 0:
        raise ValueError("File rỗng.")
    if raw.size % 2 != 0:
        raw = raw[:-1]
    return (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)


# ══════════════════════════════════════════════════════════════════
# TÌM NGƯỠNG
# ══════════════════════════════════════════════════════════════════
def find_threshold_auto(psd: np.ndarray, noise_floor: float,
                        max_signal_pct: float) -> tuple:
    """
    Tự động tìm factor tối ưu.

    Duyệt 300 giá trị factor theo thang log từ 1.5× đến 90% đỉnh PSD.
    Chọn factor nhỏ nhất sao cho tỷ lệ bin tín hiệu ≤ max_signal_pct%.

    Trả về (factor, threshold)
    """
    n_total = len(psd)
    ratio   = float(np.max(psd)) / (noise_floor + 1e-300)

    if ratio <= 1.5:
        return None, None   # không có tín hiệu nổi trên nhiễu

    candidates = np.logspace(np.log10(1.5), np.log10(ratio * 0.9), 300)

    chosen = None
    for factor in candidates:
        n_sig = int(np.sum(psd > noise_floor * factor))
        if n_sig / n_total * 100.0 <= max_signal_pct and n_sig >= 1:
            chosen = float(factor)
            break

    if chosen is None:
        chosen = ratio * 0.5    # fallback

    return chosen, float(noise_floor * chosen)


def find_threshold_manual(psd: np.ndarray, noise_floor: float,
                           factor: float) -> tuple:
    """
    Dùng factor do người dùng chỉ định.
    threshold = noise_floor × factor

    Trả về (factor, threshold)
    """
    ratio = float(np.max(psd)) / (noise_floor + 1e-300)

    if factor >= ratio:
        raise ValueError(
            f"MANUAL_FACTOR ({factor:.1f}×) lớn hơn hoặc bằng tỷ lệ đỉnh/nền "
            f"({ratio:.1f}×).\n"
            f"  → Giảm MANUAL_FACTOR xuống dưới {ratio * 0.9:.1f}"
        )
    if factor < 1.0:
        raise ValueError("MANUAL_FACTOR phải ≥ 1.0")

    return float(factor), float(noise_floor * factor)


# ══════════════════════════════════════════════════════════════════
# TÍNH BIÊN ĐỘ
# ══════════════════════════════════════════════════════════════════
def compute_amplitude(iq: np.ndarray, fs: float, nperseg: int,
                      threshold_mode: str,
                      max_signal_pct: float,
                      manual_factor: float) -> dict:
    N = len(iq)

    # Bước 1: Welch PSD
    f, psd = welch(iq, fs=fs, window='hann', nperseg=nperseg,
                   return_onesided=False, scaling='density')
    delta_f = fs / len(f)

    # Bước 2: Noise floor = median PSD
    noise_floor_density = float(np.median(psd))
    psd_peak_ratio      = float(np.max(psd)) / (noise_floor_density + 1e-300)

    print(f"  Noise floor    : {10*np.log10(noise_floor_density+1e-30):.2f} dBW/Hz")
    print(f"  PSD đỉnh/nền   : {psd_peak_ratio:.1f}× "
          f"({10*np.log10(psd_peak_ratio+1e-30):.1f} dB)")

    # Bước 3: Tìm ngưỡng theo chế độ
    if threshold_mode == "auto":
        factor, threshold = find_threshold_auto(
            psd, noise_floor_density, max_signal_pct)
        mode_label = f"auto (MAX_SIGNAL_PCT={max_signal_pct}%)"
        if factor is None:
            raise ValueError(
                "Không phát hiện được tín hiệu nổi trên nhiễu.\n"
                "  → Kiểm tra lại fs, nperseg, hoặc tăng MAX_SIGNAL_PCT."
            )
    elif threshold_mode == "manual":
        factor, threshold = find_threshold_manual(
            psd, noise_floor_density, manual_factor)
        mode_label = f"manual (MANUAL_FACTOR={manual_factor}×)"
    else:
        raise ValueError(
            f"THRESHOLD_MODE không hợp lệ: '{threshold_mode}'. "
            "Dùng 'auto' hoặc 'manual'."
        )

    signal_mask   = psd > threshold
    noise_mask    = ~signal_mask
    n_signal_bins = int(signal_mask.sum())
    n_noise_bins  = int(noise_mask.sum())
    n_total_bins  = len(psd)

    if n_signal_bins == 0:
        raise ValueError(
            f"Threshold {factor:.2f}× quá cao → không có bin tín hiệu.\n"
            "  → Giảm MANUAL_FACTOR hoặc tăng MAX_SIGNAL_PCT (auto mode)."
        )
    if n_noise_bins == 0:
        raise ValueError(
            f"Threshold {factor:.2f}× quá thấp → không có bin nhiễu.\n"
            "  → Tăng MANUAL_FACTOR hoặc giảm MAX_SIGNAL_PCT (auto mode)."
        )

    print(f"  Threshold mode : {mode_label}")
    print(f"  Factor         : {factor:.2f}×  "
          f"→  threshold = {10*np.log10(threshold+1e-30):.2f} dBW/Hz")
    print(f"  Bin tín hiệu   : {n_signal_bins:,}  "
          f"({n_signal_bins/n_total_bins*100:.2f}%)")
    print(f"  Bin nhiễu      : {n_noise_bins:,}  "
          f"({n_noise_bins/n_total_bins*100:.2f}%)")

    # Bước 4: Công suất
    p_total     = float(np.mean(np.abs(iq) ** 2))
    fill_factor = n_total_bins / n_noise_bins
    p_noise     = float(np.sum(psd[noise_mask]) * delta_f * fill_factor)
    p_signal    = p_total - p_noise

    if p_signal <= 0:
        raise ValueError(
            f"P_signal = {p_signal:.3e} ≤ 0.\n"
            f"  fill_factor = {fill_factor:.2f}× ước lượng nhiễu quá lớn.\n"
            "  → Tăng nperseg hoặc điều chỉnh threshold."
        )

    # Bước 5: Biên độ
    amp_signal_rms  = float(np.sqrt(p_signal))
    amp_noise_rms   = float(np.sqrt(p_noise))
    amp_total_rms   = float(np.sqrt(p_total))
    amp_signal_peak = amp_signal_rms * np.sqrt(2)
    amp_noise_peak  = amp_noise_rms  * np.sqrt(2)
    amp_signal_db   = 20 * np.log10(amp_signal_rms + 1e-30)
    amp_noise_db    = 20 * np.log10(amp_noise_rms  + 1e-30)
    amp_total_db    = 20 * np.log10(amp_total_rms  + 1e-30)
    snr_db          = 10 * np.log10(p_signal / p_noise)

    center_freq      = float(f[int(np.argmax(psd))])
    signal_bandwidth = n_signal_bins * delta_f

    return dict(
        amp_signal_rms=amp_signal_rms,   amp_noise_rms=amp_noise_rms,
        amp_total_rms=amp_total_rms,     amp_signal_peak=amp_signal_peak,
        amp_noise_peak=amp_noise_peak,   amp_signal_db=amp_signal_db,
        amp_noise_db=amp_noise_db,       amp_total_db=amp_total_db,
        p_signal=p_signal,               p_noise=p_noise,
        p_total=p_total,                 snr_db=snr_db,
        f=f,                             psd=psd,
        signal_mask=signal_mask,         noise_mask=noise_mask,
        noise_floor_density=noise_floor_density,
        threshold=threshold,             factor=factor,
        mode_label=mode_label,
        center_freq_hz=center_freq,      signal_bandwidth_hz=signal_bandwidth,
        n_signal_bins=n_signal_bins,     n_noise_bins=n_noise_bins,
        n_total_bins=n_total_bins,       delta_f=delta_f,
        N_samples=N,                     psd_peak_ratio=psd_peak_ratio,
    )


# ══════════════════════════════════════════════════════════════════
# IN KẾT QUẢ
# ══════════════════════════════════════════════════════════════════
def print_results(r: dict, fs: float) -> None:
    W = 62
    print(f"\n{'═'*W}")
    print(f"  KẾT QUẢ PHÂN TÍCH BIÊN ĐỘ TÍN HIỆU IQ")
    print(f"{'═'*W}")
    print(f"  Số mẫu           : {r['N_samples']:>14,}")
    print(f"  Sample rate      : {fs/1e6:>14.1f} MHz")
    print(f"  Độ phân giải Δf  : {r['delta_f']:>14.1f} Hz")
    print(f"  Tần số trung tâm : {r['center_freq_hz']/1e6:>14.4f} MHz")
    print(f"  BW tín hiệu ước  : {r['signal_bandwidth_hz']/1e6:>14.4f} MHz")
    print(f"  Threshold mode   : {r['mode_label']}")
    print(f"  Factor           : {r['factor']:>14.2f} ×")
    print(f"  PSD đỉnh/nền     : {r['psd_peak_ratio']:>14.1f} ×")
    print(f"  Bin tín hiệu     : {r['n_signal_bins']:>8,}  "
          f"({r['n_signal_bins']/r['n_total_bins']*100:.2f}%)")
    print(f"  Bin nhiễu        : {r['n_noise_bins']:>8,}  "
          f"({r['n_noise_bins']/r['n_total_bins']*100:.2f}%)")
    print(f"{'─'*W}")
    print(f"  {'':32s}  {'RMS':>9}  {'Peak':>9}  {'dB':>7}")
    print(f"  {'─'*58}")
    print(f"  {'Biên độ TÍN HIỆU':<32s}  "
          f"{r['amp_signal_rms']:>9.6f}  "
          f"{r['amp_signal_peak']:>9.6f}  "
          f"{r['amp_signal_db']:>+7.2f}")
    print(f"  {'Biên độ NHIỄU':<32s}  "
          f"{r['amp_noise_rms']:>9.6f}  "
          f"{r['amp_noise_peak']:>9.6f}  "
          f"{r['amp_noise_db']:>+7.2f}")
    print(f"  {'Biên độ TỔNG (S+N)':<32s}  "
          f"{r['amp_total_rms']:>9.6f}  "
          f"{'—':>9}  "
          f"{r['amp_total_db']:>+7.2f}")
    print(f"{'─'*W}")
    print(f"  {'Công suất tín hiệu (S)':<32s}  {r['p_signal']:>16.8f} W")
    print(f"  {'Công suất nhiễu (N)':<32s}  {r['p_noise']:>16.8f} W")
    print(f"  {'Công suất tổng (S+N)':<32s}  {r['p_total']:>16.8f} W")
    print(f"{'─'*W}")
    print(f"  SNR = 10·log10(P_s / P_n)  →  {r['snr_db']:>+.2f} dB")
    print(f"{'═'*W}\n")


# ══════════════════════════════════════════════════════════════════
# VẼ ĐỒ THỊ
# ══════════════════════════════════════════════════════════════════
def plot_results(r: dict, fs: float, input_file: str) -> None:
    f, psd   = r["f"], r["psd"]
    sort_idx = np.argsort(f)
    f_mhz    = f[sort_idx] / 1e6
    psd_db   = 10 * np.log10(psd[sort_idx] + 1e-30)
    smask    = r["signal_mask"][sort_idx]
    nmask    = r["noise_mask"][sort_idx]
    nf_db    = 10 * np.log10(r["noise_floor_density"] + 1e-30)
    thr_db   = 10 * np.log10(r["threshold"] + 1e-30)

    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    fig.suptitle(
        f"Phân tích biên độ IQ — {os.path.basename(input_file)}  "
        f"[{r['mode_label']}]\n"
        f"A_signal = {r['amp_signal_rms']:.4f} RMS ({r['amp_signal_db']:+.2f} dB)  |  "
        f"A_noise = {r['amp_noise_rms']:.4f} RMS ({r['amp_noise_db']:+.2f} dB)  |  "
        f"SNR = {r['snr_db']:.2f} dB",
        fontsize=10, fontweight='bold'
    )

    # ── Subplot 1: PSD ────────────────────────────────────────────
    ax1 = axes[0]
    y_floor = nf_db - 10
    ax1.fill_between(f_mhz, psd_db, y_floor, where=smask,
                     color='steelblue', alpha=0.4, label='Bin tín hiệu')
    ax1.fill_between(f_mhz, psd_db, y_floor, where=nmask,
                     color='salmon',    alpha=0.25, label='Bin nhiễu')
    ax1.plot(f_mhz, psd_db, color='navy', lw=0.7, label='PSD (Welch)')
    ax1.axhline(nf_db,  color='red',    ls='--', lw=1.2,
                label=f'Noise floor  {nf_db:.2f} dBW/Hz')
    ax1.axhline(thr_db, color='orange', ls=':',  lw=1.5,
                label=f'Threshold ({r["factor"]:.1f}×)  {thr_db:.2f} dBW/Hz')
    ax1.axvline(r['center_freq_hz']/1e6, color='limegreen', ls='-.', lw=1.2,
                label=f"fc = {r['center_freq_hz']/1e6:.3f} MHz")
    ax1.set_ylabel('PSD (dBW/Hz)')
    ax1.set_title(f'Phổ công suất — threshold mode: {r["mode_label"]}',
                  fontsize=10)
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.35)

    # ── Subplot 2: Biểu đồ cột biên độ ───────────────────────────
    ax2 = axes[1]
    labels    = ['Tín hiệu\n(Signal)', 'Nhiễu\n(Noise)', 'Tổng\n(S+N)']
    rms_vals  = [r['amp_signal_rms'], r['amp_noise_rms'], r['amp_total_rms']]
    peak_vals = [r['amp_signal_peak'], r['amp_noise_peak'], 0]
    c_rms     = ['steelblue', 'salmon', 'mediumpurple']
    c_peak    = ['dodgerblue', 'tomato', 'white']

    x, w = np.arange(3), 0.35
    b1 = ax2.bar(x - w/2, rms_vals,  w, color=c_rms,  alpha=0.85,
                 label='RMS', edgecolor='black', lw=0.8)
    b2 = ax2.bar(x + w/2, peak_vals, w, color=c_peak, alpha=0.75,
                 label='Peak (RMS×√2)', edgecolor='black', lw=0.8)

    for bar in b1:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2,
                 h + max(rms_vals) * 0.01,
                 f'{h:.5f}', ha='center', va='bottom', fontsize=8)
    for bar in b2:
        h = bar.get_height()
        if h > 0:
            ax2.text(bar.get_x() + bar.get_width()/2,
                     h + max(rms_vals) * 0.01,
                     f'{h:.5f}', ha='center', va='bottom', fontsize=8)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel('Biên độ (đơn vị tương đối)')
    ax2.set_title('So sánh biên độ RMS và Peak', fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, axis='y', alpha=0.35)
    ax2.text(0.98, 0.97,
             f"SNR = {r['snr_db']:.2f} dB\n"
             f"A_s / A_n = {r['amp_signal_rms']/r['amp_noise_rms']:.2f}×",
             transform=ax2.transAxes, ha='right', va='top',
             fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                       edgecolor='gray', alpha=0.9))

    plt.tight_layout()
    out = "amplitude_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"✓  Đã lưu đồ thị: {out}")
    plt.show()


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    fpath = sys.argv[1] if len(sys.argv) > 1 else input_file

    print(f"{'─'*50}")
    print(f"  File    : {fpath}")
    print(f"  Mode    : {THRESHOLD_MODE}")
    print(f"{'─'*50}")

    print(f"Đang đọc file ...")
    iq = load_iq(fpath)
    print(f"Đã đọc {len(iq):,} mẫu IQ.\n")

    print("Đang tính toán Welch PSD ...")
    results = compute_amplitude(
        iq, fs, nperseg,
        threshold_mode=THRESHOLD_MODE,
        max_signal_pct=MAX_SIGNAL_PCT,
        manual_factor=MANUAL_FACTOR,
    )

    print_results(results, fs)
    plot_results(results, fs, fpath)