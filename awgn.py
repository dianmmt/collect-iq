"""
iq_awgn_inject.py  (v5 — signal attenuation + selective AWGN)
--------------------------------------------------------------
Thêm AWGN có chọn lọc VÀ giảm biên độ tín hiệu vào file IQ binary (float32 interleaved).

Quy trình:
  1. FFT averaging → noise_floor_db = median của bottom-40% bins
  2. midVal = (max_db + noise_floor_db) / 2
     Mask: bin k là SIGNAL nếu dB[k] >= midVal - margin
           bin k là NOISE  nếu dB[k] <  midVal - margin
  3. [MỚI] Attenuation tín hiệu:
       Mỗi khối N mẫu:
         FFT(block) → nhân signal bins với atten_linear → IFFT → output
       atten_linear = 10^(-atten_db / 20)   [biên độ, không phải power]
  4. AWGN inject (noise bins):
       sigma tính từ noise_floor_db + lift_db
       AWGN phức → FFT → zero signal bins → IFFT → cộng vào block
  5. Ghi file output float32 (cùng kích thước input)

Tham số mới:
  --atten  : Mức giảm biên độ tín hiệu (dB, dương = giảm).
             Ví dụ: --atten 6  → giảm ~50% biên độ (~-6 dB power)
             Mặc định: 0.0 (giữ nguyên)

Cách dùng:
  python iq_awgn_inject.py input.bin output.bin \
      --lift 10 --atten 6 \
      --fft-size 1024 --avg-frames 64 \
      --window hann --margin 5 --seed 42

Yêu cầu: numpy  (pip install numpy)
"""

import argparse
import sys
import numpy as np


# ---------------------------------------------------------------------------
# Window
# ---------------------------------------------------------------------------

def make_window(N: int, name: str) -> np.ndarray:
    n = np.arange(N)
    if name == "hann":
        return 0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1))
    elif name == "hamming":
        return 0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))
    elif name == "blackman":
        return (0.42
                - 0.5  * np.cos(2 * np.pi * n / (N - 1))
                + 0.08 * np.cos(4 * np.pi * n / (N - 1)))
    elif name == "flattop":
        return (1
                - 1.93 * np.cos(2 * np.pi * n / (N - 1))
                + 1.29 * np.cos(4 * np.pi * n / (N - 1))
                - 0.388 * np.cos(6 * np.pi * n / (N - 1))
                + 0.032 * np.cos(8 * np.pi * n / (N - 1)))
    else:
        return np.ones(N)   # rect


# ---------------------------------------------------------------------------
# Bước 1: Phân tích phổ
# ---------------------------------------------------------------------------

def analyze_spectrum(iq, fft_size, avg_frames, window_name):
    n_samples = len(iq)
    n_frames  = min(avg_frames, n_samples // fft_size)
    if n_frames < 1:
        raise ValueError(f"File quá ngắn ({n_samples} mẫu) cho FFT size {fft_size}.")

    win       = make_window(fft_size, window_name)
    win_power = float(np.mean(win ** 2))
    accum     = np.zeros(fft_size, dtype=np.float64)

    for f in range(n_frames):
        block    = iq[f * fft_size : (f + 1) * fft_size]
        spectrum = np.fft.fft(block * win)
        accum   += np.abs(spectrum) ** 2 / (fft_size ** 2 * win_power)
    accum /= n_frames

    db_shifted     = 10 * np.log10(np.fft.fftshift(accum) + 1e-300)
    max_db         = float(db_shifted.max())
    p40            = float(np.percentile(db_shifted, 40))
    noise_floor_db = float(np.median(db_shifted[db_shifted < p40]))

    return db_shifted, max_db, noise_floor_db, n_frames


# ---------------------------------------------------------------------------
# Bước 2: Tạo mask
# ---------------------------------------------------------------------------

def build_mask(db_shifted, mid_val, margin_db, fft_size):
    threshold        = mid_val - margin_db
    is_noise_shifted = db_shifted < threshold
    is_noise         = np.fft.ifftshift(is_noise_shifted)   # về thứ tự bin FFT chuẩn
    return is_noise, threshold


# ---------------------------------------------------------------------------
# Bước 3: Tính sigma AWGN
# ---------------------------------------------------------------------------

def compute_sigma(noise_floor_db, lift_db, fft_size, n_noise_bins):
    """
    Normalization: power_per_bin = |FFT(x*win)|^2 / (N^2 * win_power)
    Per-sample power = power_per_bin * N = 10^(noise_floor/10) * N
    Sau zero + IFFT:  E[|awgn_filtered|^2] = sigma^2 * 2 * n_noise / N
    Giải:
        sigma = sqrt( 10^((noise_floor+lift)/10) * N^2 / (2 * n_noise) )
    """
    target_per_sample = (10 ** (noise_floor_db / 10.0)) * fft_size * (10 ** (lift_db / 10.0))
    return float(np.sqrt(target_per_sample * fft_size / (2.0 * n_noise_bins)))


# ---------------------------------------------------------------------------
# [MỚI] Bước 4a: Giảm biên độ tín hiệu trong miền tần số
# ---------------------------------------------------------------------------

def attenuate_signal(block, sig_mask, atten_linear):
    """
    Giảm biên độ các bin tín hiệu theo hệ số atten_linear.

    Cách làm:
      1. FFT(block)
      2. Nhân các signal bins với atten_linear  (0 < atten_linear <= 1)
      3. IFFT → trả về block đã suy giảm
      4. Lấy phần thực của IFFT để tránh lỗi số (phần ảo ≈ 0 với IQ phức)

    Tham số:
      block        : mảng complex64, 1 khối FFT
      sig_mask     : bool array, True = bin tín hiệu (thứ tự FFT chuẩn)
      atten_linear : float, 10^(-atten_db/20). = 1.0 → không đổi
    """
    if atten_linear == 1.0:
        return block   # bỏ qua nếu không cần attenuation

    spec = np.fft.fft(block)
    spec[sig_mask] *= atten_linear       # chỉ nhân signal bins
    attenuated = np.fft.ifft(spec)
    return attenuated.astype(block.dtype)


# ---------------------------------------------------------------------------
# Bước 4b: Inject AWGN vào noise bins
# ---------------------------------------------------------------------------

def inject_awgn(block, noise_mask, sigma, rng):
    """
    Tạo AWGN phức → FFT → zero signal bins → IFFT → cộng vào block.

    Tham số:
      block      : mảng complex, 1 khối FFT (đã qua attenuation)
      noise_mask : bool array, True = bin noise (thứ tự FFT chuẩn)
      sigma      : độ lệch chuẩn Gaussian cho mỗi thành phần I, Q
      rng        : numpy Generator
    """
    N = len(block)
    awgn = (rng.normal(0.0, sigma, N)
            + 1j * rng.normal(0.0, sigma, N))

    awgn_freq              = np.fft.fft(awgn)
    awgn_freq[~noise_mask] = 0.0          # zero signal bins
    awgn_filtered          = np.fft.ifft(awgn_freq)

    return block + awgn_filtered.astype(block.dtype)


# ---------------------------------------------------------------------------
# Bước 5: Xử lý từng khối (gộp attenuation + AWGN)
# ---------------------------------------------------------------------------

def process_blocks(iq, is_noise, sigma, atten_linear, fft_size, rng):
    """
    Với mỗi khối fft_size mẫu:
      1. Attenuate signal bins  (nếu atten_linear < 1)
      2. Inject AWGN vào noise bins  (nếu sigma > 0)

    is_noise    : True = bin noise
    atten_linear: hệ số biên độ cho signal bins, = 10^(-atten_db/20)
    """
    n_samples  = len(iq)
    n_complete = (n_samples // fft_size) * fft_size
    output     = iq.copy()
    sig_mask   = ~is_noise   # True = bin tín hiệu

    n_blocks     = n_complete // fft_size
    report_every = max(1, n_blocks // 10)

    for b, start in enumerate(range(0, n_complete, fft_size)):
        block = iq[start : start + fft_size]

        # --- 4a. Giảm biên độ tín hiệu ---
        block = attenuate_signal(block, sig_mask, atten_linear)

        # --- 4b. Inject AWGN vào noise bins ---
        if sigma > 0.0:
            block = inject_awgn(block, is_noise, sigma, rng)

        output[start : start + fft_size] = block

        if (b + 1) % report_every == 0 or b == n_blocks - 1:
            print(f"  [{(b+1)/n_blocks*100:5.1f}%] khối {b+1}/{n_blocks}", end="\r")

    print()
    leftover = n_samples - n_complete
    if leftover > 0:
        print(f"  Lưu ý: {leftover} mẫu cuối không đủ 1 khối → giữ nguyên.")
    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Thêm AWGN có chọn lọc + giảm biên độ tín hiệu vào file IQ float32 (v5).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input",  help="File IQ gốc (.bin, float32 I,Q interleaved)")
    parser.add_argument("output", help="File IQ output (.bin)")
    parser.add_argument("--lift",        type=float, default=10.0,
                        help="Mức nâng noise floor (dB). 0 = không inject AWGN")
    parser.add_argument("--atten",       type=float, default=20.0,
                        help="Mức giảm biên độ tín hiệu (dB, dương = giảm). "
                             "Ví dụ: 6 → giảm ~50%% biên độ. 0 = giữ nguyên")
    parser.add_argument("--fft-size",    type=int,   default=1024)
    parser.add_argument("--avg-frames",  type=int,   default=32)
    parser.add_argument("--window",      type=str,   default="hann",
                        choices=["hann","hamming","blackman","flattop","rect"])
    parser.add_argument("--margin",      type=float, default=0.0,
                        help="Buffer bảo vệ sườn tín hiệu (dB). Khuyến nghị 3~10 dB")
    parser.add_argument("--sample-rate", type=float, default=50e6,
                        help="Sample rate Hz (chỉ để hiển thị)")
    parser.add_argument("--seed",        type=int,   default=None)
    args = parser.parse_args()

    if args.fft_size < 64 or (args.fft_size & (args.fft_size - 1)) != 0:
        sys.exit(f"Lỗi: --fft-size phải là lũy thừa 2 >= 64. Nhận: {args.fft_size}")
    if args.atten < 0:
        sys.exit("Lỗi: --atten phải >= 0 (dương = giảm tín hiệu).")

    # ── Đọc file ──────────────────────────────────────────────────────────
    print(f"\n[1/5] Đọc file: {args.input}")
    try:
        raw = np.fromfile(args.input, dtype=np.float32)
    except FileNotFoundError:
        sys.exit(f"Lỗi: không tìm thấy file '{args.input}'")
    if raw.size == 0:
        sys.exit("Lỗi: file rỗng.")
    if raw.size % 2 != 0:
        print("  Cảnh báo: số float32 lẻ, bỏ phần tử cuối.")
        raw = raw[:-1]

    iq        = (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)
    n_samples = len(iq)
    print(f"  {n_samples:,} mẫu IQ | {raw.nbytes/1024:.1f} KB | "
          f"{n_samples/args.sample_rate*1000:.2f} ms @ {args.sample_rate/1e6:.3f} MHz")

    # ── Phân tích phổ ─────────────────────────────────────────────────────
    print(f"\n[2/5] Phân tích phổ "
          f"(FFT={args.fft_size}, avg={args.avg_frames}, "
          f"window={args.window}, margin={args.margin} dB)")

    db_shifted, max_db, noise_floor_db, n_frames = analyze_spectrum(
        iq, args.fft_size, args.avg_frames, args.window
    )
    mid_val = (max_db + noise_floor_db) / 2.0

    is_noise, threshold = build_mask(db_shifted, mid_val, args.margin, args.fft_size)
    n_noise = int(is_noise.sum())
    n_sig   = args.fft_size - n_noise

    if n_noise == 0:
        sys.exit("Lỗi: không có bin nhiễu nào. Giảm --margin hoặc kiểm tra file.")

    print(f"  max           = {max_db:.2f} dBFS")
    print(f"  noise_floor   = {noise_floor_db:.2f} dBFS  (median bottom-40% bins)")
    print(f"  midVal        = {mid_val:.2f} dBFS")
    print(f"  threshold     = {threshold:.2f} dBFS  (midVal - {args.margin} dB)")
    print(f"  Signal bins   : {n_sig:5d} / {args.fft_size}  ({100*n_sig/args.fft_size:.1f}%)")
    print(f"  Noise  bins   : {n_noise:5d} / {args.fft_size}  ({100*n_noise/args.fft_size:.1f}%)")

    # ── Tính các tham số xử lý ────────────────────────────────────────────
    print(f"\n[3/5] Tính tham số xử lý")

    # Attenuation
    atten_linear = 10 ** (-args.atten / 20.0)   # biên độ (voltage), không phải power
    atten_power  = 20 * np.log10(atten_linear)  # = -atten_db (để kiểm tra)
    expected_sig_db = max_db - args.atten        # ước lượng đỉnh sau attenuation

    print(f"  [Tín hiệu]")
    print(f"  atten         = -{args.atten:.1f} dB  →  hệ số biên độ = {atten_linear:.6f}")
    print(f"  max gốc       = {max_db:.2f} dBFS")
    print(f"  max kỳ vọng   = {expected_sig_db:.2f} dBFS  (sau attenuation)")

    # AWGN sigma
    if args.lift > 0:
        sigma        = compute_sigma(noise_floor_db, args.lift, args.fft_size, n_noise)
        expected_awgn = noise_floor_db + args.lift
        print(f"\n  [Noise]")
        print(f"  noise_floor   = {noise_floor_db:.2f} dBFS")
        print(f"  lift          = +{args.lift:.1f} dB")
        print(f"  kỳ vọng output= {expected_awgn:.2f} dBFS")
        print(f"  sigma         = {sigma:.6e}")
    else:
        sigma = 0.0
        print(f"\n  [Noise] lift=0 → bỏ qua inject AWGN")

    # ── Xử lý khối ───────────────────────────────────────────────────────
    n_blocks = n_samples // args.fft_size
    print(f"\n[4/5] Xử lý {n_blocks:,} khối × {args.fft_size} mẫu ...")
    print(f"  Attenuation tín hiệu : {'TẮT' if args.atten == 0 else f'-{args.atten:.1f} dB (×{atten_linear:.4f} biên độ)'}")
    print(f"  Inject AWGN          : {'TẮT' if sigma == 0 else f'+{args.lift:.1f} dB  (sigma={sigma:.3e})'}")

    rng    = np.random.default_rng(args.seed)
    iq_out = process_blocks(iq, is_noise, sigma, atten_linear, args.fft_size, rng)

    # ── Ghi file ──────────────────────────────────────────────────────────
    print(f"\n[5/5] Ghi file: {args.output}")
    out_raw       = np.empty(n_samples * 2, dtype=np.float32)
    out_raw[0::2] = iq_out.real
    out_raw[1::2] = iq_out.imag
    out_raw.tofile(args.output)
    print(f"  Đã ghi {n_samples:,} mẫu IQ ({out_raw.nbytes/1024:.1f} KB)")

    print(f"""
═══════════════════════════════════════════════════════
Hoàn tất  (v5)
───────────────────────────────────────────────────────
  Input          : {args.input}
  Output         : {args.output}

  [Phổ gốc]
  max            : {max_db:.2f} dBFS
  noise_floor    : {noise_floor_db:.2f} dBFS
  midVal         : {mid_val:.2f} dBFS
  threshold      : {threshold:.2f} dBFS
  Signal bins    : {n_sig} / {args.fft_size}
  Noise  bins    : {n_noise} / {args.fft_size}

  [Attenuation tín hiệu]
  atten          : -{args.atten:.1f} dB  (hệ số = {atten_linear:.6f})
  max kỳ vọng    : {expected_sig_db:.2f} dBFS

  [AWGN inject]
  lift           : +{args.lift:.1f} dB
  noise kỳ vọng  : {noise_floor_db + args.lift:.2f} dBFS
  sigma          : {sigma:.6e}

  [Config]
  FFT size       : {args.fft_size}   window  : {args.window}
  avg frames     : {args.avg_frames}   margin  : {args.margin} dB
  seed           : {args.seed if args.seed is not None else 'ngẫu nhiên'}
═══════════════════════════════════════════════════════
""")


if __name__ == "__main__":
    main()
