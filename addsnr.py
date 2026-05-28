<<<<<<< HEAD
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
# tính toán biên độ tín hiệu và nhiễu
# ==========================================
# 1. THAM SỐ ĐẦU VÀO
# ==========================================
input_file = "uav.bin"      
output_file = "output_iq.bin"    
desired_snr = 15.0               # SNR mong muốn (dB)
fs = 50000000.0                  # 50 MHz từ USRP B205 mini

# Chế độ điều chỉnh:
# "add_noise"    -> Giữ nguyên tín hiệu gốc, cộng thêm nhiễu để đạt SNR mong muốn (Phổ biến/An toàn nhất)
# "scale_signal" -> Giữ nguyên nền nhiễu cũ, tăng/giảm biên độ tín hiệu 
mode = "add_noise"

# ==========================================
# 2. ĐỌC FILE IQ BINARY
# ==========================================
if not os.path.exists(input_file):
    raise FileNotFoundError(f"Không tìm thấy file: {input_file}")

raw_data = np.fromfile(input_file, dtype=np.float32)
iq_signal = raw_data[0::2] + 1j * raw_data[1::2]
N = len(iq_signal)

# ==========================================
# 3. ƯỚC LƯỢNG SNR THEO CÔNG SUẤT CHUẨN (PSD WELCH)
# ==========================================
# Tính tổng công suất toàn phần miền thời gian (Parseval)
p_total = np.mean(np.abs(iq_signal)**2)

# Tính mật độ phổ công suất để tìm Noise Floor
f, psd = welch(iq_signal, fs=fs, window='hann', nperseg=2048, return_onesided=False)

# Dùng Median để tìm đúng mức nhiễu nền, bỏ qua các đỉnh tín hiệu lớn
noise_floor_density = np.median(psd)
p_noise_current = noise_floor_density * fs  # Tổng công suất nhiễu hiện tại

# Công suất tín hiệu sạch hiện tại
p_signal_current = p_total - p_noise_current

if p_signal_current <= 0:
    raise ValueError("Tín hiệu trong file quá yếu hoặc cấu hình sai, không tách được nhiễu!")

measured_snr = 10 * np.log10(p_signal_current / p_noise_current)

print("--- KẾT QUẢ PHÂN TÍCH CHUẨN CÔNG SUẤT ---")
print(f"Tổng công suất nhận được (S+N): {p_total:.6f}")
print(f"Công suất nhiễu ước lượng (N):  {p_noise_current:.6f}")
print(f"Công suất tín hiệu sạch (S):    {p_signal_current:.6f}")
print(f"SNR đo được hiện tại:            {measured_snr:.2f} dB")
print(f"SNR mong muốn:                   {desired_snr:.2f} dB")

# ==========================================
# 4. ĐIỀU CHỈNH SNR TRÊN MIỀN THỜI GIAN
# ==========================================
iq_modified = iq_signal.copy()

if mode == "add_noise":
    # Tính công suất nhiễu tổng cộng cần phải có để đạt mong muốn
    # SNR = 10*log10(P_sig / P_noise_target)
    p_noise_target = p_signal_current / (10 ** (desired_snr / 10.0))
    
    # Lượng công suất nhiễu cần cộng thêm vào
    p_noise_to_add = p_noise_target - p_noise_current
    
    if p_noise_to_add > 0:
        # Tạo nhiễu trắng Gauss phức (AWGN) với công suất p_noise_to_add
        # Chú ý: Nhiễu phức cần chia đôi công suất cho kênh I và kênh Q
        standard_deviation = np.sqrt(p_noise_to_add / 2.0)
        noise_add = (np.random.normal(0, standard_deviation, N) + 
                     1j * np.random.normal(0, standard_deviation, N))
        iq_modified = iq_signal + noise_add
        print(f"Trạng thái: Đã CỘNG THÊM nhiễu vào tín hiệu miền thời gian.")
    else:
        print("Cảnh báo: SNR hiện tại đã thấp hơn mong muốn, không thể hạ thấp thêm bằng cách cộng nhiễu!")

elif mode == "scale_signal":
    # Tính công suất tín hiệu cần đạt dựa trên nền nhiễu cố định
    p_signal_target = p_noise_current * (10 ** (desired_snr / 10.0))
    # Hệ số scale biên độ (Căn bậc hai của hệ số scale công suất)
    k_scale = np.sqrt(p_signal_target / p_signal_current)
    
    # Tách tín hiệu rời khỏi nhiễu tạm thời (về mặt toán học), scale rồi đưa nhiễu trở lại
    # Hoặc đơn giản là scale toàn bộ chuỗi nếu giả định nhiễu ban đầu rất nhỏ
    iq_modified = iq_signal * k_scale
    print(f"Trạng thái: Đã SCALE biên độ chuỗi tín hiệu thời gian.")

# ==========================================
# 5. XUẤT FILE .BIN MỚI
# ==========================================
output_raw = np.empty(2 * N, dtype=np.float32)
output_raw[0::2] = np.real(iq_modified)
output_raw[1::2] = np.imag(iq_modified)
output_raw.tofile(output_file)
print(f"Đã xuất file thành công: {output_file}")

# ==========================================
# 6. VẼ ĐỒ THỊ KIỂM TRA PHỔ PSD
# ==========================================
f_goc, psd_goc = welch(iq_signal, fs=fs, window='hann', nperseg=2048, return_onesided=False)
f_mod, psd_mod = welch(iq_modified, fs=fs, window='hann', nperseg=2048, return_onesided=False)

sort_idx = np.argsort(f_goc)

plt.figure(figsize=(12, 5))
plt.plot(f_goc[sort_idx] / 1e6, 10 * np.log10(psd_goc[sort_idx]), "b", label="Phổ biên độ Gốc")
plt.plot(f_mod[sort_idx] / 1e6, 10 * np.log10(psd_mod[sort_idx]), "g--", label="Phổ biên độ Sau Điều Chỉnh")
plt.axhline(10 * np.log10(noise_floor_density), color="r", linestyle=":", label="Nền nhiễu ước lượng ban đầu")
plt.grid(True)
plt.title(f"So sánh phổ năng lượng (Target SNR: {desired_snr} dB)")
plt.xlabel("Băng thông lệch tâm (MHz)")
plt.ylabel("Năng lượng phổ (dB/Hz)")
plt.legend()
plt.tight_layout()
plt.show()
=======
"""
addsnr.py — Điều chỉnh SNR file IQ (giảm bằng AWGN / tăng bằng spectral denoising)
             + Vẽ so sánh phổ biên độ tần số và phân phối tín hiệu/nhiễu trước/sau
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, stft, istft

from config import SAMPLE_RATE, NPERSEG, SIGNAL_THRESHOLD_FACTOR

# ══════════════════════════════════════════════════════════════════
# THAM SỐ MẶC ĐỊNH
# ══════════════════════════════════════════════════════════════════
DEFAULT_INPUT  = "signal/iq_20260521_144402.bin"
fs             = SAMPLE_RATE
nperseg        = NPERSEG


# ══════════════════════════════════════════════════════════════════
# ĐỌC FILE IQ
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
# TÍNH BIÊN ĐỘ / SNR (Welch PSD)
# ══════════════════════════════════════════════════════════════════
def compute_amplitude(iq: np.ndarray, fs_val: float, nperseg_val: int,
                      threshold_factor: float) -> dict:
    N = len(iq)
    f, psd = welch(iq, fs=fs_val, window='hann', nperseg=nperseg_val,
                   return_onesided=False, scaling='density')
    delta_f = fs_val / len(f)

    noise_floor_density = float(np.median(psd))
    psd_peak_ratio      = float(np.max(psd)) / (noise_floor_density + 1e-300)
    threshold           = noise_floor_density * threshold_factor
    signal_mask         = psd > threshold
    noise_mask          = ~signal_mask

    n_signal_bins = int(signal_mask.sum())
    n_noise_bins  = int(noise_mask.sum())
    n_total_bins  = len(psd)

    if n_noise_bins == 0:
        raise ValueError("Không tìm được bin nhiễu. Giảm SIGNAL_THRESHOLD_FACTOR.")
    if n_signal_bins == 0:
        raise ValueError("Không tìm được bin tín hiệu. Tăng SIGNAL_THRESHOLD_FACTOR.")

    p_total     = float(np.mean(np.abs(iq) ** 2))
    fill_factor = n_total_bins / n_noise_bins
    p_noise     = float(np.sum(psd[noise_mask]) * delta_f * fill_factor)
    p_signal    = p_total - p_noise

    if p_signal <= 0:
        raise ValueError(
            f"P_signal ≤ 0 ({p_signal:.2e}). Tín hiệu quá yếu hoặc "
            "threshold_factor quá cao."
        )

    amp_signal_rms  = float(np.sqrt(p_signal))
    amp_noise_rms   = float(np.sqrt(p_noise))
    amp_total_rms   = float(np.sqrt(p_total))
    amp_signal_peak = amp_signal_rms * np.sqrt(2)
    amp_noise_peak  = amp_noise_rms  * np.sqrt(2)
    amp_signal_db   = 20 * np.log10(amp_signal_rms + 1e-30)
    amp_noise_db    = 20 * np.log10(amp_noise_rms  + 1e-30)
    amp_total_db    = 20 * np.log10(amp_total_rms  + 1e-30)
    snr_db          = 10 * np.log10(p_signal / p_noise)
    center_freq     = float(f[int(np.argmax(psd))])
    signal_bw       = n_signal_bins * delta_f

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
        threshold=threshold,             psd_peak_ratio=psd_peak_ratio,
        center_freq_hz=center_freq,      signal_bandwidth_hz=signal_bw,
        n_signal_bins=n_signal_bins,     n_noise_bins=n_noise_bins,
        n_total_bins=n_total_bins,       delta_f=delta_f,
        N_samples=N,
    )


# ══════════════════════════════════════════════════════════════════
# IN KẾT QUẢ
# ══════════════════════════════════════════════════════════════════
def print_results(r: dict, fs_val: float, label: str = "") -> None:
    W = 60
    tag = f" [{label}]" if label else ""
    print(f"\n{'═'*W}")
    print(f"  KẾT QUẢ PHÂN TÍCH IQ{tag}")
    print(f"{'═'*W}")
    print(f"  Số mẫu          : {r['N_samples']:>12,}")
    print(f"  Sample rate     : {fs_val/1e6:>12.1f} MHz")
    print(f"  Tần số trung tâm: {r['center_freq_hz']/1e6:>12.4f} MHz")
    print(f"  BW tín hiệu ước : {r['signal_bandwidth_hz']/1e6:>12.4f} MHz")
    print(f"  Bin tín hiệu    : {r['n_signal_bins']:>12,}  / {r['n_total_bins']:,}")
    print(f"{'─'*W}")
    print(f"  {'':30s}  {'RMS':>8}  {'Peak':>8}  {'dB':>7}")
    print(f"  {'─'*W}")
    print(f"  {'Biên độ TÍN HIỆU':<30s}  "
          f"{r['amp_signal_rms']:>8.6f}  {r['amp_signal_peak']:>8.6f}  "
          f"{r['amp_signal_db']:>+7.2f}")
    print(f"  {'Biên độ NHIỄU':<30s}  "
          f"{r['amp_noise_rms']:>8.6f}  {r['amp_noise_peak']:>8.6f}  "
          f"{r['amp_noise_db']:>+7.2f}")
    print(f"  {'Biên độ TỔNG (S+N)':<30s}  "
          f"{r['amp_total_rms']:>8.6f}  {'—':>8}  {r['amp_total_db']:>+7.2f}")
    print(f"{'─'*W}")
    print(f"  SNR = 10·log10(P_s/P_n)  →  {r['snr_db']:>+.2f} dB")
    print(f"{'═'*W}\n")


# ══════════════════════════════════════════════════════════════════
# SPECTRAL DENOISING — lọc nhiễu miền tần số
# ══════════════════════════════════════════════════════════════════
def spectral_denoise(iq: np.ndarray, results: dict,
                     fs_val: float, nperseg_val: int) -> np.ndarray:
    """
    Zero out noise frequency bins in STFT domain to recover clean signal estimate.
    noise_mask from Welch PSD (two-sided, nperseg bins) maps directly to STFT bins.
    """
    noise_mask = results["noise_mask"]   # shape: (nperseg_val,)

    _, _, Zxx = stft(iq, fs=fs_val, nperseg=nperseg_val,
                     noverlap=nperseg_val // 2,
                     return_onesided=False)
    # Zxx shape: (nperseg_val, n_frames)

    Zxx_filtered = Zxx.copy()
    Zxx_filtered[noise_mask, :] = 0

    _, iq_clean = istft(Zxx_filtered, fs=fs_val, nperseg=nperseg_val,
                        noverlap=nperseg_val // 2,
                        input_onesided=False)

    return iq_clean[:len(iq)].astype(np.complex64)


# ══════════════════════════════════════════════════════════════════
# ĐIỀU CHỈNH SNR — cả hai hướng tăng/giảm
# ══════════════════════════════════════════════════════════════════
def adjust_snr(iq_original: np.ndarray, results: dict,
               target_snr_db: float,
               fs_val: float, nperseg_val: int) -> np.ndarray:
    """
    Điều chỉnh SNR tới target_snr_db:
      - target < SNR gốc : thêm AWGN để giảm SNR
      - target ≥ SNR gốc : spectral denoising + thêm AWGN ở mức nhiễu mới
    """
    p_signal    = results["p_signal"]
    p_noise     = results["p_noise"]
    snr_current = results["snr_db"]
    N           = len(iq_original)
    snr_linear  = 10 ** (target_snr_db / 10.0)

    if target_snr_db < snr_current:
        # ── Giảm SNR: thêm AWGN ──────────────────────────────────
        p_noise_required = p_signal / snr_linear
        p_noise_to_add   = p_noise_required - p_noise
        sigma = np.sqrt(p_noise_to_add / 2.0)
        noise = (np.random.normal(0, sigma, N) +
                 1j * np.random.normal(0, sigma, N)).astype(np.complex64)
        print(f"    [AWGN] thêm {p_noise_to_add:.4e} W nhiễu")
        return iq_original + noise
    else:
        # ── Tăng SNR: spectral denoising → thêm AWGN mức thấp ───
        print(f"    [Denoise] lọc nhiễu tần số + AWGN mức {target_snr_db:+.1f} dB")
        iq_clean = spectral_denoise(iq_original, results, fs_val, nperseg_val)
        p_clean  = float(np.mean(np.abs(iq_clean) ** 2))
        if p_clean <= 1e-30:
            raise ValueError("Spectral denoising trả về tín hiệu zero-power.")
        p_noise_target = p_clean / snr_linear
        sigma = np.sqrt(p_noise_target / 2.0)
        noise = (np.random.normal(0, sigma, N) +
                 1j * np.random.normal(0, sigma, N)).astype(np.complex64)
        return (iq_clean + noise).astype(np.complex64)


# ══════════════════════════════════════════════════════════════════
# LƯU FILE IQ
# ══════════════════════════════════════════════════════════════════
def save_iq(iq: np.ndarray, filepath: str) -> None:
    """Lưu complex IQ → interleaved float32 (I0 Q0 I1 Q1 ...)."""
    out = np.empty(iq.size * 2, dtype=np.float32)
    out[0::2] = np.real(iq).astype(np.float32)
    out[1::2] = np.imag(iq).astype(np.float32)
    out.tofile(filepath)
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"    Lưu: {os.path.basename(filepath)}  ({size_mb:.2f} MB)")


# ══════════════════════════════════════════════════════════════════
# HÀM VẼ PHỔ (helper)
# ══════════════════════════════════════════════════════════════════
def _draw_psd_ax(ax, r: dict, title: str) -> None:
    f, psd   = r["f"], r["psd"]
    sort_idx = np.argsort(f)
    f_mhz    = f[sort_idx] / 1e6
    psd_db   = 10 * np.log10(psd[sort_idx] + 1e-30)
    smask    = r["signal_mask"][sort_idx]
    nmask    = r["noise_mask"][sort_idx]
    nf_db    = 10 * np.log10(r["noise_floor_density"] + 1e-30)
    thr_db   = 10 * np.log10(r["threshold"] + 1e-30)
    y_floor  = nf_db - 10

    ax.fill_between(f_mhz, psd_db, y_floor, where=smask,
                    color='steelblue', alpha=0.4, label='Signal bins')
    ax.fill_between(f_mhz, psd_db, y_floor, where=nmask,
                    color='salmon',    alpha=0.25, label='Noise bins')
    ax.plot(f_mhz, psd_db, color='navy', lw=0.7, label='PSD (Welch)')
    ax.axhline(nf_db,  color='red',    ls='--', lw=1.2,
               label=f'Noise floor {nf_db:.1f} dBW/Hz')
    ax.axhline(thr_db, color='orange', ls=':',  lw=1.5,
               label=f'Threshold {thr_db:.1f} dBW/Hz')
    ax.axvline(r['center_freq_hz'] / 1e6, color='limegreen', ls='-.', lw=1.1,
               label=f"fc={r['center_freq_hz']/1e6:.3f} MHz")
    ax.set_xlabel('Tần số (MHz)')
    ax.set_ylabel('PSD (dBW/Hz)')
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)


def _draw_amp_ax(ax, r: dict, title: str) -> None:
    labels    = ['Signal', 'Noise', 'Total\n(S+N)']
    rms_vals  = [r['amp_signal_rms'], r['amp_noise_rms'], r['amp_total_rms']]
    peak_vals = [r['amp_signal_peak'], r['amp_noise_peak'], 0]
    c_rms     = ['steelblue', 'salmon', 'mediumpurple']
    c_peak    = ['dodgerblue', 'tomato', 'white']

    x, w = np.arange(3), 0.35
    b1 = ax.bar(x - w/2, rms_vals,  w, color=c_rms,  alpha=0.85,
                label='RMS', edgecolor='black', lw=0.8)
    b2 = ax.bar(x + w/2, peak_vals, w, color=c_peak, alpha=0.75,
                label='Peak', edgecolor='black', lw=0.8)

    y_max = max(rms_vals) if max(rms_vals) > 0 else 1.0
    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2,
                h + y_max * 0.01, f'{h:.5f}',
                ha='center', va='bottom', fontsize=7)
    for bar in b2:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    h + y_max * 0.01, f'{h:.5f}',
                    ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Biên độ (tương đối)')
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)
    ax.text(0.98, 0.97,
            f"SNR = {r['snr_db']:.2f} dB\n"
            f"A_s/A_n = {r['amp_signal_rms']/(r['amp_noise_rms']+1e-30):.2f}×",
            transform=ax.transAxes, ha='right', va='top',
            fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                      edgecolor='gray', alpha=0.9))


# ══════════════════════════════════════════════════════════════════
# VẼ SO SÁNH 2×2
# ══════════════════════════════════════════════════════════════════
def plot_comparison(r_orig: dict, r_new: dict,
                    target_snr_db: float,
                    fpath_orig: str, out_prefix: str) -> None:
    """
    2×2 figure:
      [0,0] PSD gốc        [0,1] PSD sau điều chỉnh
      [1,0] Bar chart gốc  [1,1] Bar chart sau điều chỉnh
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    orig_name = os.path.basename(fpath_orig)
    fig.suptitle(
        f"So sánh SNR: {r_orig['snr_db']:.2f} dB  →  {r_new['snr_db']:.2f} dB  "
        f"(target: {target_snr_db:+.0f} dB)   [{orig_name}]",
        fontsize=11, fontweight='bold'
    )

    _draw_psd_ax(
        axes[0, 0], r_orig,
        f"Phổ biên độ tần số — GỐC  (SNR = {r_orig['snr_db']:.2f} dB)"
    )
    _draw_psd_ax(
        axes[0, 1], r_new,
        f"Phổ biên độ tần số — SAU ĐC  (SNR = {r_new['snr_db']:.2f} dB)"
    )
    _draw_amp_ax(
        axes[1, 0], r_orig,
        f"Phân phối biên độ — GỐC  (SNR = {r_orig['snr_db']:.2f} dB)"
    )
    _draw_amp_ax(
        axes[1, 1], r_new,
        f"Phân phối biên độ — SAU ĐC  (SNR = {r_new['snr_db']:.2f} dB)"
    )

    plt.tight_layout()
    out_png = f"{out_prefix}_compare.png"
    plt.savefig(out_png, dpi=120, bbox_inches='tight')
    print(f"    Plot: {os.path.basename(out_png)}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════
# SUMMARY PLOT
# ══════════════════════════════════════════════════════════════════
def plot_snr_sweep_summary(snr_targets: list, measured_snrs: list,
                           stem: str) -> None:
    """Scatter: target SNR (x) vs measured SNR (y) với đường lý tưởng y=x."""
    t = np.array(snr_targets, dtype=float)
    m = np.array(measured_snrs, dtype=float)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot([t.min(), t.max()], [t.min(), t.max()],
            'k--', lw=1.3, label='Lý tưởng (y = x)')
    ax.scatter(t, m, color='steelblue', s=65, zorder=5, label='Đo thực tế')

    for ti, mi in zip(t, m):
        ax.annotate(f'{mi:.1f}', (ti, mi),
                    textcoords='offset points', xytext=(6, 5), fontsize=7)

    ax.set_xlabel('SNR target (dB)', fontsize=11)
    ax.set_ylabel('SNR đo được (dB)', fontsize=11)
    ax.set_title('Tổng hợp sweep SNR: target vs measured', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.35)

    plt.tight_layout()
    out_png = f"{stem}_snr_sweep_summary.png"
    plt.savefig(out_png, dpi=120, bbox_inches='tight')
    print(f"\n  Summary plot: {out_png}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Điều chỉnh SNR file IQ và vẽ so sánh phổ trước/sau"
    )
    parser.add_argument("file", nargs="?", default=DEFAULT_INPUT,
                        help="Đường dẫn file IQ (.bin)")
    parser.add_argument("--snr-targets", nargs="+", type=float,
                        default=list(np.arange(-20, 22, 2)),
                        metavar="DB",
                        help="Danh sách SNR target (dB). Mặc định: -20 đến +20, bước 2")
    parser.add_argument("--no-plot", action="store_true",
                        help="Bỏ qua bước vẽ đồ thị")
    args = parser.parse_args()

    fpath       = args.file
    snr_targets = sorted(args.snr_targets)

    print(f"\n{'═'*60}")
    print(f"  File    : {fpath}")
    print(f"  Targets : {[f'{s:+.0f}' for s in snr_targets]}")
    print(f"{'═'*60}")

    # ── Bước 1: Phân tích file gốc ───────────────────────────────
    print("\n[Bước 1] Đọc và phân tích file IQ gốc ...")
    iq_raw = load_iq(fpath)
    print(f"  Đã đọc {len(iq_raw):,} mẫu IQ.")
    results_raw = compute_amplitude(iq_raw, fs, nperseg, SIGNAL_THRESHOLD_FACTOR)
    print_results(results_raw, fs, label="GỐC")

    stem          = os.path.splitext(fpath)[0]
    measured_snrs = []
    valid_targets = []
    saved_count   = 0

    # ── Bước 2: Điều chỉnh từng mức SNR ─────────────────────────
    for i, target_snr in enumerate(snr_targets, 1):
        print(f"\n[{i}/{len(snr_targets)}] SNR target: {target_snr:+.1f} dB  "
              f"(gốc: {results_raw['snr_db']:.2f} dB)")

        # Điều chỉnh và lưu (luôn thực hiện)
        try:
            iq_adj  = adjust_snr(iq_raw, results_raw, target_snr, fs, nperseg)
            out_bin = f"{stem}_snr_{target_snr:+.1f}dB.bin"
            save_iq(iq_adj, out_bin)
            saved_count += 1
        except Exception as exc:
            print(f"    ⚠ Lỗi điều chỉnh/lưu: {exc}")
            continue

        # Đo SNR sau điều chỉnh (có thể thất bại với SNR rất thấp)
        try:
            results_adj = compute_amplitude(iq_adj, fs, nperseg, SIGNAL_THRESHOLD_FACTOR)
            measured    = results_adj["snr_db"]
            measured_snrs.append(measured)
            valid_targets.append(target_snr)
            print(f"    Measured SNR: {measured:.2f} dB  "
                  f"(lỗi: {measured - target_snr:+.2f} dB)")

            if not args.no_plot:
                out_prefix = f"{stem}_snr_{target_snr:+.1f}dB"
                plot_comparison(results_raw, results_adj,
                                target_snr, fpath, out_prefix)
        except Exception as exc:
            print(f"    ⚠ Không đo được SNR (file đã lưu): {exc}")

    # ── Bước 3: Summary plot ─────────────────────────────────────
    if not args.no_plot and valid_targets:
        plot_snr_sweep_summary(valid_targets, measured_snrs, stem)

    print(f"\n{'═'*60}")
    print(f"  Lưu file  : {saved_count}/{len(snr_targets)} targets")
    print(f"  Đo SNR    : {len(valid_targets)}/{len(snr_targets)} targets")
    print(f"{'═'*60}\n")
>>>>>>> d1850a4 (update source adding noise)
