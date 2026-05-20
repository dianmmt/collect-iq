import os
import numpy as np
import matplotlib.pyplot as plt

def addnoise(signal, noise, snr_db):
    """
    Hàm thêm nhiễu vào tín hiệu dựa trên mức SNR yêu cầu (tương đương hàm MATLAB).
    """
    # Đảm bảo nhiễu có chiều dài bằng hoặc lớn hơn tín hiệu
    s_len = len(signal)
    n_len = len(noise)
    if s_len > n_len:
        raise ValueError("Error: Chiều dài tín hiệu (signal) lớn hơn chiều dài nhiễu (noise)!")
    
    # Nếu nhiễu dài hơn tín hiệu, cắt một đoạn ngẫu nhiên giống như hàm MATLAB
    if n_len > s_len:
        start_idx = np.random.randint(0, n_len - s_len + 1)
        noise = noise[start_idx:start_idx + s_len]
        
    # Tính chuẩn (norm) của tín hiệu và nhiễu
    norm_signal = np.linalg.norm(signal)
    norm_noise = np.linalg.norm(noise)
    
    # Tính toán hệ số tỷ lệ dựa trên SNR (dB)
    # Công thức: 10^(0.05 * snr) tương đương với 10^(snr / 20)
    scaled_noise = (noise / norm_noise) * norm_signal / (10.0 ** (0.05 * snr_db))
    
    # Tạo tín hiệu hỗn hợp
    noisy_signal = signal + scaled_noise
    
    return noisy_signal, scaled_noise


# =========================================================================
# CHƯƠNG TRÌNH CHÍNH (MAIN)
# =========================================================================
if __name__ == "__main__":
    
    # 1. CẤU HÌNH THÔNG SỐ
    file_signal = 'uav.bin'  # Tên file IQ gốc của bạn
    file_output = 'uav-noise.bin'   # Tên file kết quả xuất ra
    desired_snr = 20.0                 # SNR mong muốn (dB) (Tăng giảm tùy ý)
    data_type   = np.float32           # Kiểu dữ liệu (thường là np.float32 hoặc np.int16)

    # 2. ĐỌC FILE TÍN HIỆU IQ GỐC
    if not os.path.exists(file_signal):
        raise FileNotFoundError(f"Không tìm thấy file '{file_signal}' trong thư mục hiện tại!")
        
    # Đọc toàn bộ file binary dưới dạng mảng 1 chiều
    raw_data = np.fromfile(file_signal, dtype=data_type)
    
    # Tách dữ liệu đan xen [I1, Q1, I2, Q2, ...] thành số phức I + j*Q
    I = raw_data[0::2]
    Q = raw_data[1::2]
    signal = I + 1j * Q

    # 3. TẠO TÍN HIỆU NHIỄU PHỨC (COMPLEX GAUSSIAN NOISE)
    # Nhiễu phức trắng chuẩn hóa công suất (chia cho sqrt(2))
    noise_real = np.random.randn(len(signal))
    noise_imag = np.random.randn(len(signal))
    noise = (noise_real + 1j * noise_imag) / np.sqrt(2)

    # 4. THAY ĐỔI SNR BẰNG HÀM ADDNOISE
    noisy_signal, _ = addnoise(signal, noise, desired_snr)

    # 5. CHUYỂN ĐỔI NGƯỢC VÀ GHI RA FILE BIN MỚI
    I_noisy = np.real(noisy_signal)
    Q_noisy = np.imag(noisy_signal)

    # Trộn đan xen lại thành một mảng phẳng [I1, Q1, I2, Q2, ...]
    raw_output = np.empty(len(I_noisy) * 2, dtype=data_type)
    raw_output[0::2] = I_noisy
    raw_output[1::2] = Q_noisy

    # Ghi dữ liệu xuống file binary
    raw_output.tofile(file_output)

    print("=== CHẠY THÀNH CÔNG ===")
    print(f"Đã tạo xong file: {file_output} với SNR = {desired_snr} dB")

    # 6. VẼ ĐỒ THỊ KIỂM TRA (Hiển thị 500 mẫu đầu tiên)
    num_samples_to_plot = min(500, len(signal))
    
    plt.figure(figsize=(10, 6))
    
    # Đồ thị gốc
    plt.subplot(2, 1, 1)
    plt.plot(I[:num_samples_to_plot], label='I (Real)', color='blue')
    plt.plot(Q[:num_samples_to_plot], label='Q (Imag)', color='red')
    plt.title('Tín hiệu IQ gốc (500 mẫu đầu)')
    plt.legend()
    plt.grid(True)
    
    # Đồ thị sau khi đổi SNR
    plt.subplot(2, 1, 2)
    plt.plot(I_noisy[:num_samples_to_plot], label='I (Noisy)', color='blue')
    plt.plot(Q_noisy[:num_samples_to_plot], label='Q (Noisy)', color='red')
    plt.title(f'Tín hiệu sau khi chỉnh SNR = {desired_snr} dB')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
