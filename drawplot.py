import numpy as np
import matplotlib.pyplot as plt

# ===== CONFIG =====
filename = "signal/uav.bin"
fs = 50_000_000  # sample rate (Hz)

# ===== LOAD IQ =====
raw = np.fromfile(filename, dtype=np.float32)

# I,Q interleaved -> complex
iq = raw[0::2] + 1j * raw[1::2]

# ===== FFT =====
N = len(iq)

window = np.hanning(N)
iq_win = iq * window

fft_data = np.fft.fftshift(np.fft.fft(iq_win))
freq = np.fft.fftshift(np.fft.fftfreq(N, d=1/fs))

# amplitude in dB
spectrum = 20 * np.log10(np.abs(fft_data) + 1e-12)

# ===== PLOT =====
plt.figure(figsize=(12, 5))
plt.plot(freq / 1e6, spectrum)

plt.xlabel("Frequency (MHz)")
plt.ylabel("Amplitude (dB)")
plt.title("IQ Spectrum")
plt.grid(True)

plt.tight_layout()
plt.show()