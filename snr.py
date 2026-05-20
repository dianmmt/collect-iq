import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sp_signal

# --- Đọc file IQ (raw binary float32 hoặc complex64) ---
# Nếu dùng UHD/GNU Radio lưu ra file
iq = np.fromfile("output.bin", dtype=np.complex64)

# Hoặc từ uhd_rx_cfile / osmocom
# iq = np.fromfile("capture.32fc", dtype=np.complex64)

sample_rate = 50e6   # Hz — điều chỉnh theo cấu hình B205
center_freq = 2450e6 # Hz

# --- Tính PSD bằng Welch ---
freqs, psd = sp_signal.welch(iq, fs=sample_rate,
                              nperseg=1024, return_onesided=False)
freqs = np.fft.fftshift(freqs)
psd   = np.fft.fftshift(psd)
psd_dB = 10 * np.log10(psd + 1e-30)

# --- Xác định vùng tín hiệu và vùng nhiễu ---
# Ví dụ: tín hiệu nằm trong ±50 kHz quanh center
bw_signal = 100e3   # Hz — băng thông tín hiệu
bw_noise   = 200e3  # Hz — vùng nhiễu (ngoài tín hiệu)

f_abs = np.abs(freqs)
signal_mask = f_abs < (bw_signal / 2)
noise_mask  = (f_abs > (bw_signal / 2)) & (f_abs < (bw_signal / 2 + bw_noise))

P_signal = np.mean(psd[signal_mask])
P_noise  = np.mean(psd[noise_mask])

SNR_linear = P_signal / P_noise
SNR_dB     = 10 * np.log10(SNR_linear)
print(f"SNR = {SNR_dB:.2f} dB")
