"""
Tính SNR cơ bản từ file IQ binary (float32)
Dữ liệu: interleaved I/Q float32 [I0, Q0, I1, Q1, ...]
"""

import numpy as np
import sys
import os

def load_iq(filepath):
    """Đọc file IQ float32, trả về mảng complex64."""
    raw = np.fromfile(filepath, dtype=np.float32)
    if len(raw) % 2 != 0:
        raw = raw[:-1]  # bỏ byte lẻ cuối
    iq = raw[0::2] + 1j * raw[1::2]
    return iq.astype(np.complex64)

def snr_power_method(iq):
    """
    Phương pháp 1: SNR dựa trên công suất tín hiệu vs nhiễu
    - Tín hiệu: công suất trung bình của toàn bộ mẫu
    - Nhiễu: ước lượng từ phần noise floor trong phổ tần số
    """
    N = len(iq)
    spectrum = np.fft.fft(iq, n=N)
    power_spectrum = (np.abs(spectrum) ** 2) / N

    # Tìm đỉnh tín hiệu (top 1%)
    sorted_power = np.sort(power_spectrum)[::-1]
    top_idx = max(1, int(N * 0.01))
    signal_power = np.mean(sorted_power[:top_idx])

    # Noise floor = trung bình 50% dưới cùng
    noise_idx = int(N * 0.50)
    noise_power = np.mean(sorted_power[noise_idx:])

    snr_linear = signal_power / (noise_power + 1e-30)
    snr_db = 10 * np.log10(snr_linear)
    return snr_db, signal_power, noise_power

def snr_variance_method(iq):
    """
    Phương pháp 2: SNR từ phương sai (đơn giản, thường dùng cho UAV telemetry)
    Giả định: tín hiệu = mean, nhiễu = phương sai (std)
    """
    amplitude = np.abs(iq)
    signal_mean = np.mean(amplitude)
    noise_std  = np.std(amplitude)
    snr_linear = (signal_mean / (noise_std + 1e-30)) ** 2
    snr_db = 10 * np.log10(snr_linear)
    return snr_db

def snr_fft_peak(iq):
    """
    Phương pháp 3: SNR từ đỉnh FFT (phổ biến trong SDR / UAV link)
    SNR = P_peak / P_noise_average
    """
    N = len(iq)
    spectrum = np.fft.fftshift(np.fft.fft(iq, n=N))
    power = (np.abs(spectrum) ** 2) / N

    peak_power = np.max(power)
    # Loại bỏ 5 bin quanh đỉnh để tính noise floor
    peak_bin = np.argmax(power)
    mask = np.ones(N, dtype=bool)
    mask[max(0, peak_bin-5):min(N, peak_bin+6)] = False
    noise_floor = np.mean(power[mask])

    snr_linear = peak_power / (noise_floor + 1e-30)
    snr_db = 10 * np.log10(snr_linear)
    return snr_db, peak_bin, N

def main(filepath):
    if not os.path.exists(filepath):
        print(f"[LỖI] Không tìm thấy file: {filepath}")
        sys.exit(1)

    filesize = os.path.getsize(filepath)
    print(f"File       : {filepath}")
    print(f"Kích thước : {filesize:,} bytes")

    iq = load_iq(filepath)
    n_samples = len(iq)
    print(f"Số mẫu IQ  : {n_samples:,}")
    print(f"Công suất TB: {np.mean(np.abs(iq)**2):.6f}")
    print()

    # --- Phương pháp 1 ---
    snr1, sp, np_ = snr_power_method(iq)
    print(f"[PP1] SNR (Spectrum - top 1% vs bottom 50%) : {snr1:.2f} dB")
    print(f"      Signal power : {10*np.log10(sp+1e-30):.2f} dBW")
    print(f"      Noise  power : {10*np.log10(np_+1e-30):.2f} dBW")

    # --- Phương pháp 2 ---
    snr2 = snr_variance_method(iq)
    print(f"\n[PP2] SNR (Variance method)                  : {snr2:.2f} dB")

    # --- Phương pháp 3 ---
    snr3, peak_bin, N = snr_fft_peak(iq)
    freq_offset = (peak_bin - N//2) / N  # normalized
    print(f"\n[PP3] SNR (FFT peak vs noise floor)          : {snr3:.2f} dB")
    print(f"      Tần số đỉnh (normalized) : {freq_offset:.4f}")

    print()
    print("=" * 50)
    print(f"  SNR ước tính (trung bình 3 phương pháp): {np.mean([snr1, snr2, snr3]):.2f} dB")
    print("=" * 50)

if __name__ == "__main__":
    filepath = sys.argv[1] if len(sys.argv) > 1 else "uav.bin"
    main(filepath)