
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
#tính toán biên độ trung bình của tín hiệu và nhiễu, tính toán snr , điều chỉnh snr như mong muốn
# ══════════════════════════════════════════════════════════════════
# THAM SỐ ĐẦU VÀO
# ══════════════════════════════════════════════════════════════════
input_file = "signal/ktc-50db.bin"
fs         = 50_000_000.0   # Sample rate: 50 MHz (USRP B205 mini)
nperseg    = 1024           # Độ dài segment Welch (tăng → phân giải tốt hơn)

# Ngưỡng phân biệt bin tín hiệu vs nhiễu
# Bin có PSD > noise_floor 
# × factor → coi là bin tín hiệu
SIGNAL_THRESHOLD_FACTOR = 2  # tương đương +6 dB trên noise floor


# ══════════════════════════════════════════════════════════════════
# HÀM ĐỌC FILE IQ
# ══════════════════════════════════════════════════════════════════
def load_iq(filepath: str) -> np.ndarray:
    """Đọc file IQ interleaved float32 → complex64."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Không tìm thấy file: {filepath}")
    raw = np.fromfile(filepath, dtype=np.float32)
    if raw.size == 0:
        raise ValueError("File rỗng.")
    if raw.size % 2 != 0:
        raw = raw[:-1]
    return (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)


# ══════════════════════════════════════════════════════════════════
# HÀM TÍNH BIÊN ĐỘ TÍN HIỆU VÀ NHIỄU
# ══════════════════════════════════════════════════════════════════
def compute_amplitude(iq: np.ndarray, fs: float, nperseg: int,
                      threshold_factor: float) -> dict:
    N = len(iq)

    # ── Bước 1: Welch PSD ─────────────────────────────────────────
    f, psd = welch(iq, fs=fs, window='hann', nperseg=nperseg,
                   return_onesided=False, scaling='density')
    delta_f = fs / len(f)   # Hz/bin = độ phân giải tần số

    # ── Bước 2: Noise floor bằng Median ───────────────────────────
    # Median ổn định hơn mean vì bỏ qua các đỉnh tín hiệu
    noise_floor_density = np.median(psd)   # W/Hz

    # ── Bước 3: Phân loại bin tín hiệu / nhiễu ────────────────────
    threshold   = noise_floor_density * threshold_factor
    signal_mask = psd > threshold    # bin có năng lượng cao → tín hiệu
    noise_mask  = ~signal_mask       # bin còn lại → nhiễu

    n_signal_bins = int(signal_mask.sum())
    n_noise_bins  = int(noise_mask.sum())
    n_total_bins  = len(psd)

    if n_noise_bins == 0:
        raise ValueError("Không tìm được bin nhiễu. Giảm SIGNAL_THRESHOLD_FACTOR.")
    if n_signal_bins == 0:
        raise ValueError("Không tìm được bin tín hiệu. Tăng SIGNAL_THRESHOLD_FACTOR.")

    # ── Bước 4: Tính công suất ────────────────────────────────────
    # Công suất tổng từ miền thời gian (Parseval — chính xác tuyệt đối)
    p_total = float(np.mean(np.abs(iq) ** 2))

    # Công suất nhiễu = tích phân PSD trên bin nhiễu
    # + hệ số bù (fill_factor) để ước lượng phần nhiễu bị che dưới bin tín hiệu
    fill_factor = n_total_bins / n_noise_bins
    p_noise     = float(np.sum(psd[noise_mask]) * delta_f * fill_factor)

    # Công suất tín hiệu = tổng - nhiễu
    p_signal = p_total - p_noise
    if p_signal <= 0:
        raise ValueError(
            f"P_signal ≤ 0 ({p_signal:.2e}). Tín hiệu quá yếu hoặc "
            "threshold_factor quá cao — hãy giảm SIGNAL_THRESHOLD_FACTOR."
        )

    # ── Bước 5: Biên độ RMS ───────────────────────────────────────
    # A_rms = √P  (đơn vị tương đương với đơn vị của mẫu IQ)
    amp_signal_rms = float(np.sqrt(p_signal))
    amp_noise_rms  = float(np.sqrt(p_noise))
    amp_total_rms  = float(np.sqrt(p_total))

    # Biên độ đỉnh ước lượng (peak = rms × √2, đúng cho sóng sine đơn)
    amp_signal_peak = amp_signal_rms * np.sqrt(2)
    amp_noise_peak  = amp_noise_rms  * np.sqrt(2)

    # Chuyển sang dB (tham chiếu = 1.0)
    amp_signal_db = 20 * np.log10(amp_signal_rms + 1e-30)
    amp_noise_db  = 20 * np.log10(amp_noise_rms  + 1e-30)
    amp_total_db  = 20 * np.log10(amp_total_rms  + 1e-30)

    # SNR
    snr_db = 10 * np.log10(p_signal / p_noise)

    # Tần số trung tâm tín hiệu (bin có PSD cao nhất)
    center_bin = int(np.argmax(psd))
    center_freq = float(f[center_bin])

    # Băng thông tín hiệu ước tính (số bin tín hiệu × Δf)
    signal_bandwidth = n_signal_bins * delta_f

    return {
        # Biên độ
        "amp_signal_rms"  : amp_signal_rms,
        "amp_noise_rms"   : amp_noise_rms,
        "amp_total_rms"   : amp_total_rms,
        "amp_signal_peak" : amp_signal_peak,
        "amp_noise_peak"  : amp_noise_peak,
        "amp_signal_db"   : amp_signal_db,
        "amp_noise_db"    : amp_noise_db,
        "amp_total_db"    : amp_total_db,
        # Công suất
        "p_signal"        : p_signal,
        "p_noise"         : p_noise,
        "p_total"         : p_total,
        # SNR
        "snr_db"          : snr_db,
        # Phổ
        "f"               : f,
        "psd"             : psd,
        "signal_mask"     : signal_mask,
        "noise_mask"      : noise_mask,
        "noise_floor_density" : noise_floor_density,
        "threshold"       : threshold,
        # Thông tin tín hiệu
        "center_freq_hz"  : center_freq,
        "signal_bandwidth_hz" : signal_bandwidth,
        "n_signal_bins"   : n_signal_bins,
        "n_noise_bins"    : n_noise_bins,
        "n_total_bins"    : n_total_bins,
        "delta_f"         : delta_f,
        "N_samples"       : N,
    }


# ══════════════════════════════════════════════════════════════════
# IN KẾT QUẢ
# ══════════════════════════════════════════════════════════════════
def print_results(r: dict, fs: float) -> None:
    W = 58
    print(f"\n{'═'*W}")
    print(f"  KẾT QUẢ PHÂN TÍCH BIÊN ĐỘ TÍN HIỆU IQ")
    print(f"{'═'*W}")
    print(f"  Số mẫu          : {r['N_samples']:>12,}")
    print(f"  Sample rate     : {fs/1e6:>12.1f} MHz")
    print(f"  Độ phân giải Δf : {r['delta_f']:>12.1f} Hz")
    print(f"  Tần số trung tâm: {r['center_freq_hz']/1e6:>12.4f} MHz")
    print(f"  BW tín hiệu ước : {r['signal_bandwidth_hz']/1e6:>12.4f} MHz")
    print(f"  Bin tín hiệu    : {r['n_signal_bins']:>12,}  / {r['n_total_bins']:,}")
    print(f"  Bin nhiễu       : {r['n_noise_bins']:>12,}  / {r['n_total_bins']:,}")
    print(f"{'─'*W}")
    print(f"  {'':30s}  {'RMS':>8}  {'Peak':>8}  {'dB':>7}")
    print(f"  {'─'*54}")
    print(f"  {'Biên độ TÍN HIỆU':<30s}  "
          f"{r['amp_signal_rms']:>8.6f}  "
          f"{r['amp_signal_peak']:>8.6f}  "
          f"{r['amp_signal_db']:>+7.2f}")
    print(f"  {'Biên độ NHIỄU':<30s}  "
          f"{r['amp_noise_rms']:>8.6f}  "
          f"{r['amp_noise_peak']:>8.6f}  "
          f"{r['amp_noise_db']:>+7.2f}")
    print(f"  {'Biên độ TỔNG (S+N)':<30s}  "
          f"{r['amp_total_rms']:>8.6f}  "
          f"{'—':>8}  "
          f"{r['amp_total_db']:>+7.2f}")
    print(f"{'─'*W}")
    print(f"  {'Công suất tín hiệu (S)':<30s}  {r['p_signal']:>12.6f} W")
    print(f"  {'Công suất nhiễu (N)':<30s}  {r['p_noise']:>12.6f} W")
    print(f"  {'Công suất tổng (S+N)':<30s}  {r['p_total']:>12.6f} W")
    print(f"{'─'*W}")
    print(f"  SNR = 10·log10(P_s/P_n)  →  {r['snr_db']:>+.2f} dB")
    print(f"{'═'*W}\n")


# ══════════════════════════════════════════════════════════════════
# VẼ ĐỒ THỊ
# ══════════════════════════════════════════════════════════════════
def plot_results(r: dict, fs: float, input_file: str) -> None:
    f   = r["f"]
    psd = r["psd"]
    sort_idx = np.argsort(f)
    f_mhz    = f[sort_idx] / 1e6
    psd_db   = 10 * np.log10(psd[sort_idx] + 1e-30)
    smask    = r["signal_mask"][sort_idx]
    nmask    = r["noise_mask"][sort_idx]

    nf_db    = 10 * np.log10(r["noise_floor_density"] + 1e-30)
    thr_db   = 10 * np.log10(r["threshold"] + 1e-30)

    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    fig.suptitle(
        f"Phân tích biên độ IQ — {os.path.basename(input_file)}\n"
        f"A_signal = {r['amp_signal_rms']:.4f} RMS ({r['amp_signal_db']:+.2f} dB)  |  "
        f"A_noise = {r['amp_noise_rms']:.4f} RMS ({r['amp_noise_db']:+.2f} dB)  |  "
        f"SNR = {r['snr_db']:.2f} dB",
        fontsize=11, fontweight='bold'
    )

    # ── Subplot 1: PSD với phân vùng tín hiệu / nhiễu ──────────────
    ax1 = axes[0]
    ax1.fill_between(f_mhz, psd_db, nf_db - 10,
                     where=smask, color='steelblue', alpha=0.35, label='Bin tín hiệu')
    ax1.fill_between(f_mhz, psd_db, nf_db - 10,
                     where=nmask,  color='salmon',    alpha=0.25, label='Bin nhiễu')
    ax1.plot(f_mhz, psd_db, color='navy', lw=0.7, label='PSD (Welch)')
    ax1.axhline(nf_db,  color='red',    ls='--', lw=1.2,
                label=f'Noise floor  {nf_db:.1f} dBW/Hz')
    ax1.axhline(thr_db, color='orange', ls=':',  lw=1.2,
                label=f'Threshold (+6 dB)  {thr_db:.1f} dBW/Hz')
    ax1.axvline(r['center_freq_hz']/1e6, color='limegreen', ls='-.', lw=1,
                label=f"fc = {r['center_freq_hz']/1e6:.3f} MHz")
    ax1.set_ylabel('PSD (dBW/Hz)')
    ax1.set_title('Phổ công suất — phân vùng tín hiệu / nhiễu', fontsize=10)
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.35)

    # ── Subplot 2: Thanh so sánh biên độ ───────────────────────────
    ax2 = axes[1]
    labels = ['Tín hiệu\n(Signal)', 'Nhiễu\n(Noise)', 'Tổng\n(S+N)']
    rms_vals = [r['amp_signal_rms'], r['amp_noise_rms'], r['amp_total_rms']]
    peak_vals = [r['amp_signal_peak'], r['amp_noise_peak'], 0]
    colors_rms  = ['steelblue', 'salmon', 'mediumpurple']
    colors_peak = ['dodgerblue', 'tomato', 'white']

    x = np.arange(len(labels))
    width = 0.35
    bars_rms  = ax2.bar(x - width/2, rms_vals,  width, color=colors_rms,
                        alpha=0.85, label='RMS', edgecolor='black', lw=0.8)
    bars_peak = ax2.bar(x + width/2, peak_vals, width, color=colors_peak,
                        alpha=0.75, label='Peak (ước lượng)', edgecolor='black',
                        lw=0.8, linestyle='--')

    # Gán nhãn số lên từng cột
    for bar in bars_rms:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + max(rms_vals)*0.01,
                 f'{h:.5f}', ha='center', va='bottom', fontsize=8)
    for bar in bars_peak:
        h = bar.get_height()
        if h > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, h + max(rms_vals)*0.01,
                     f'{h:.5f}', ha='center', va='bottom', fontsize=8)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel('Biên độ (đơn vị tương đối)')
    ax2.set_title('So sánh biên độ RMS và Peak', fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, axis='y', alpha=0.35)

    # Thêm thông tin SNR vào subplot 2
    ax2.text(0.98, 0.97,
             f"SNR = {r['snr_db']:.2f} dB\n"
             f"A_s / A_n = {r['amp_signal_rms']/r['amp_noise_rms']:.2f}×",
             transform=ax2.transAxes, ha='right', va='top',
             fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                       edgecolor='gray', alpha=0.9))

    plt.tight_layout()
    out_plot = "amplitude_analysis.png"
    plt.savefig(out_plot, dpi=150, bbox_inches='tight')
    print(f"✓  Đã lưu đồ thị: {out_plot}")
    plt.show()


# Thêm hàm này vào dưới hàm compute_amplitude của bạn

def adjust_snr(iq_original: np.ndarray, current_results: dict, target_snr_db: float) -> np.ndarray:
    """
    Điều chỉnh SNR của tín hiệu IQ về mức target_snr_db bằng cách cộng thêm nhiễu trắng (AWGN).
    """
    p_signal = current_results["p_signal"]
    p_noise_current = current_results["p_noise"]
    
    # 1. Tính công suất nhiễu tổng cộng cần có để đạt target SNR
    snr_linear = 10 ** (target_snr_db / 10.0)
    p_noise_required = p_signal / snr_linear
    
    # 2. Tính lượng công suất nhiễu cần bổ sung
    p_noise_to_add = p_noise_required - p_noise_current
    
    if p_noise_to_add <= 0:
        print(f"⚠️  Cảnh báo: SNR thực tế của file ({current_results['snr_db']:.2f} dB) "
              f"đã thấp hơn mức mong muốn ({target_snr_db:.2f} dB).")
        print("   -> Không thể tăng SNR bằng cách thêm nhiễu. Giữ nguyên tín hiệu gốc.")
        return iq_original.copy()
    
    # 3. Tạo nhiễu trắng Gaussian phức (Complex AWGN)
    # Vì là nhiễu phức, công suất tổng P = sigma^2_I + sigma^2_Q = 2 * sigma_vế^2
    # Do đó mỗi thành phần I và Q sẽ có độ lệch chuẩn là sqrt(P_noise / 2)
    N = len(iq_original)
    sigma = np.sqrt(p_noise_to_add / 2.0)
    
    noise_i = np.random.normal(0, sigma, N)
    noise_q = np.random.normal(0, sigma, N)
    awgn_noise = (noise_i + 1j * noise_q).astype(np.complex64)
    
    # 4. Cộng nhiễu vào tín hiệu cũ
    iq_adjusted = iq_original + awgn_noise
    print(f"✓ Đã bù thêm {p_noise_to_add:.6f} W nhiễu để ép SNR về {target_snr_db:.1f} dB.")
    
    return iq_adjusted
# ══════════════════════════════════════════════════════════════════
# LƯU IQ -> FILE BIN FLOAT32
# ══════════════════════════════════════════════════════════════════
def save_iq(iq: np.ndarray, filepath: str) -> None:
    """
    Lưu complex IQ -> interleaved float32:
    I0 Q0 I1 Q1 ...
    """
    
    out = np.empty(iq.size * 2, dtype=np.float32)

    out[0::2] = np.real(iq).astype(np.float32)
    out[1::2] = np.imag(iq).astype(np.float32)

    out.tofile(filepath)

    size_mb = os.path.getsize(filepath) / (1024 * 1024)

    print(f"✓ Đã lưu IQ: {filepath}")
    print(f"✓ Kích thước: {size_mb:.2f} MB")   
if __name__ == "__main__":
    fpath = sys.argv[1] if len(sys.argv) > 1 else input_file

    print(f"Đang đọc file: {fpath} ...")
    iq_raw = load_iq(fpath)
    print(f"Đã đọc {len(iq_raw):,} mẫu IQ.")

    # --- BƯỚC A: Đo đạc thông số gốc ---
    print("\n[Bước 1] Phân tích trạng thái file IQ gốc...")
    results_raw = compute_amplitude(iq_raw, fs, nperseg, SIGNAL_THRESHOLD_FACTOR)
    print(f"-> SNR Gốc đo được: {results_raw['snr_db']:.2f} dB")

    # --- BƯỚC B: CẤU HÌNH THAY ĐỔI SNR THEO Ý MUỐN ---
    TARGET_SNR = -10.0  # <--- THAY ĐỔI GIÁ TRỊ SNR BẠN MONG MUỐN Ở ĐÂY (dB)
    
    print(f"\n[Bước 2] Tiến hành ép SNR về mức mong muốn: {TARGET_SNR} dB...")
    iq_modified = adjust_snr(iq_raw, results_raw, TARGET_SNR)

# ── LƯU FILE IQ MỚI ─────────────────────────────────────────────
    output_iq_file = (
        os.path.splitext(fpath)[0]
        + f"_snr_{TARGET_SNR:.1f}dB.bin"
    )

    save_iq(iq_modified, output_iq_file)

    # --- BƯỚC C: Đo lại mảng IQ mới để kiểm chứng công thức ---
    print("\n[Bước 3] Tính toán lại biên độ sau khi tinh chỉnh SNR...")
    results_modified = compute_amplitude(iq_modified, fs, nperseg, SIGNAL_THRESHOLD_FACTOR)

    # In kết quả và vẽ đồ thị của tín hiệu mới đã đổi SNR
    print_results(results_modified, fs)
    plot_results(results_modified, fs, fpath + f"_snr_{TARGET_SNR}dB")