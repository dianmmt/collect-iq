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