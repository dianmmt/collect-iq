"""
=============================================================
  IQ .bin file → STFT → Ảnh phổ
  Input : file .bin (interleaved float32 I/Q)
  Output: ảnh WebP / JPEG / PNG cho mỗi segment
=============================================================
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
from matplotlib import cm as mpl_cm

try:
    from scipy.signal import stft as scipy_stft
    from scipy.signal.windows import hamming as scipy_hamming
except ImportError:
    from scipy.signal import stft as scipy_stft
    from scipy.signal import hamming as scipy_hamming

from config import (
    SAMPLE_RATE, CENTER_FREQ,
    DURATION_SEC as DURATION_TIME,
    STFT_POINT, OVERLAP,
    SPEC_OUTPUT_DIR as OUTPUT_DIR, CMAP,
    SAVE_FORMAT, WEBP_QUALITY, JPEG_QUALITY,
    IMG_WIDTH, IMG_HEIGHT, N_SAVE_WORKERS,
)

# ═════════════════════════════════════════════════════════════
#  CẤU HÌNH — chỉnh sửa tại đây
# ═════════════════════════════════════════════════════════════

<<<<<<< HEAD
<<<<<<< HEAD
INPUT_FILE    = "uav_2.bin"   # đường dẫn file .bin cần xử lý
=======
INPUT_FILE    = "signal/uav_snr_-10.0dB.bin"   # đường dẫn file .bin cần xử lý
>>>>>>> f7a65cc (lastest version of ktc)

SAMPLE_RATE   = 50e6            # Hz — phải khớp với lúc thu
CENTER_FREQ   = 2450e6          # Hz — chỉ dùng để tham khảo

DURATION_TIME = 0.1             # giây / segment → 1 ảnh
STFT_POINT    = 1024            # số điểm FFT
OVERLAP       = 0.7             # tỉ lệ overlap (0.0 – <1.0)

OUTPUT_DIR    = "spectrogram_output"   # thư mục lưu ảnh
CMAP          = "jet"           # colormap: jet, hot, viridis, inferno, grey, ...

# Định dạng lưu ảnh:
#   "webp_lossy"    → WebP lossy  (nhỏ + nhanh, khuyến nghị)
#   "webp_lossless" → WebP lossless (không mất dữ liệu màu)
#   "jpeg"          → JPEG
#   "png"           → PNG lossless
SAVE_FORMAT   = "webp_lossy"

WEBP_QUALITY  = 80              # 0-100, chỉ dùng khi SAVE_FORMAT="webp_lossy"
JPEG_QUALITY  = 85              # 0-95,  chỉ dùng khi SAVE_FORMAT="jpeg"
IMG_WIDTH     = 1000            # px chiều ngang ảnh output
IMG_HEIGHT    = 700             # px chiều dọc  ảnh output

N_SAVE_WORKERS = 4              # số thread lưu ảnh song song
=======
INPUT_FILE = "uav_2205_10lift_20downn.bin"  # đường dẫn file .bin cần xử lý
>>>>>>> d1850a4 (update source adding noise)

# ═════════════════════════════════════════════════════════════

_EXT_MAP = {
    "webp_lossy":    ".webp",
    "webp_lossless": ".webp",
    "jpeg":          ".jpg",
    "png":           ".png",
}

# ── CuPy (tự động detect GPU) ─────────────────────────────────
USE_CUDA = False
cp = None
try:
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() > 0:
        USE_CUDA = True
        props    = cp.cuda.runtime.getDeviceProperties(0)
        dev_name = props["name"]
        if isinstance(dev_name, (bytes, bytearray)):
            dev_name = dev_name.decode()
        print(f"[CUDA] GPU : {dev_name}")
except Exception as e:
    print(f"[CUDA] Không dùng được CuPy ({e}). Chạy CPU.")


# ══════════════════════════════════════════════════════════════
#  LUT COLORMAP
# ══════════════════════════════════════════════════════════════
def build_colormap_lut(name: str) -> np.ndarray:
    cmap   = mpl_cm.get_cmap(name, 256)
    colors = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)
    return colors  # (256, 3) uint8

COLORMAP_LUT = build_colormap_lut(CMAP)


# ══════════════════════════════════════════════════════════════
#  RENDER: power array → PIL Image
# ══════════════════════════════════════════════════════════════
def power_to_pil(power: np.ndarray) -> Image.Image:
    p_min = power.min()
    p_max = power.max()
    denom = p_max - p_min
    if denom < 1e-9:
        denom = 1e-9

    idx = np.clip(
        ((power - p_min) / denom * 255.0).astype(np.uint8),
        0, 255
    )
    rgb = COLORMAP_LUT[idx]
    rgb = rgb[::-1, :, :]  # flip dọc: origin=lower

    img = Image.fromarray(rgb, mode="RGB")
    if img.size != (IMG_WIDTH, IMG_HEIGHT):
        img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.LANCZOS)
    return img


# ══════════════════════════════════════════════════════════════
#  SAVE: PIL Image → file
# ══════════════════════════════════════════════════════════════
def save_pil_image(img: Image.Image, path: str) -> None:
    if SAVE_FORMAT == "webp_lossy":
        img.save(path, format="WEBP", quality=WEBP_QUALITY, method=4)
    elif SAVE_FORMAT == "webp_lossless":
        img.save(path, format="WEBP", lossless=True)
    elif SAVE_FORMAT == "jpeg":
        img.save(path, format="JPEG", quality=JPEG_QUALITY,
                 optimize=True, subsampling=2)
    else:  # png
        img.save(path, format="PNG", optimize=True)


# ══════════════════════════════════════════════════════════════
#  CUDA STFT CONTEXT
# ══════════════════════════════════════════════════════════════
class CudaSTFTContext:
    def __init__(self, nperseg: int, noverlap: int, fs: float):
        self.nperseg  = nperseg
        self.noverlap = noverlap
        self.hop      = nperseg - noverlap
        self.fs       = fs
        if self.hop <= 0:
            raise ValueError("OVERLAP quá lớn (hop <= 0)")
        self.stream = cp.cuda.Stream(non_blocking=True)
        with self.stream:
            self.window_gpu = cp.hamming(nperseg).astype(cp.float32)

    def compute(self, iq_segment: np.ndarray) -> np.ndarray:
        hop        = self.hop
        nperseg    = self.nperseg
        n          = iq_segment.size
        num_frames = 1 + (n - nperseg) // hop
        if num_frames <= 0:
            raise ValueError("Segment quá ngắn so với STFT_POINT")
        with self.stream:
            x_gpu = cp.asarray(iq_segment, dtype=cp.complex64)
            try:
                strides = (x_gpu.strides[0] * hop, x_gpu.strides[0])
                frames  = cp.lib.stride_tricks.as_strided(
                    x_gpu, shape=(num_frames, nperseg), strides=strides
                ).copy()
            except Exception:
                frames = cp.stack(
                    [x_gpu[i:i + nperseg] for i in range(num_frames)]
                )
            frames  = frames * self.window_gpu
            Zxx     = cp.fft.fft(frames, axis=1)
            Zxx     = cp.roll(Zxx, nperseg // 2, axis=1)
            power   = 10.0 * cp.log10(cp.abs(Zxx) + 1e-12)
            power_T = power.T
        self.stream.synchronize()
        return cp.asnumpy(power_T)


# ══════════════════════════════════════════════════════════════
#  CPU STFT
# ══════════════════════════════════════════════════════════════
def compute_stft_cpu(segment: np.ndarray, nperseg: int, noverlap: int) -> np.ndarray:
    window = scipy_hamming(nperseg)
    f, t, Zxx = scipy_stft(
        x=segment, fs=SAMPLE_RATE, window=window,
        nperseg=nperseg, noverlap=noverlap,
        return_onesided=False, boundary=None, padded=False
    )
    f    = np.fft.fftshift(f)
    Zxx  = np.fft.fftshift(Zxx, axes=0)
    power = 10.0 * np.log10(np.abs(Zxx) + 1e-12)
    return power


# ══════════════════════════════════════════════════════════════
#  ĐỌC FILE BIN → complex64
# ══════════════════════════════════════════════════════════════
def load_bin_iq(filepath: str) -> np.ndarray:
    """
    Đọc file .bin interleaved float32 [I0, Q0, I1, Q1, ...]
    Trả về np.ndarray complex64 shape (N,)
    """
    raw = np.fromfile(filepath, dtype=np.float32)
    if raw.size % 2 != 0:
        raw = raw[:-1]
    iq = raw[0::2] + 1j * raw[1::2]
    return iq.astype(np.complex64)


# ══════════════════════════════════════════════════════════════
#  SAVE TASK (chạy trong ThreadPool)
# ══════════════════════════════════════════════════════════════
def save_task(power: np.ndarray, save_path: str, seg_idx: int, dt_stft: float):
    t0 = time.perf_counter()
    img = power_to_pil(power)
    save_pil_image(img, save_path)
    dt_save = (time.perf_counter() - t0) * 1000
    fname = os.path.basename(save_path)
    print(f"  ✓ [{seg_idx:04d}] {fname} | STFT {dt_stft:.1f} ms | Save {dt_save:.1f} ms")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  IQ .bin → STFT → Ảnh phổ")
    print("=" * 60)

    if not os.path.isfile(INPUT_FILE):
        print(f"[ERROR] Không tìm thấy file: {INPUT_FILE}")
        return

    # ── Đọc file ───────────────────────────────────────────────
    print(f"[LOAD] {INPUT_FILE}")
    t0      = time.perf_counter()
    iq_all  = load_bin_iq(INPUT_FILE)
    dt_load = (time.perf_counter() - t0) * 1000

    total_samps = iq_all.size
    total_sec   = total_samps / SAMPLE_RATE
    print(f"[LOAD] {total_samps:,} samples | {total_sec*1000:.2f} ms | Đọc xong trong {dt_load:.1f} ms")

    # ── Tính segment ───────────────────────────────────────────
    samples_per_seg = int(SAMPLE_RATE * DURATION_TIME)
    noverlap        = int(STFT_POINT * OVERLAP)
    num_segments    = total_samps // samples_per_seg

    if num_segments == 0:
        print(f"[ERROR] File quá ngắn: cần ít nhất {samples_per_seg:,} samples, "
              f"chỉ có {total_samps:,}.")
        return

    ext      = _EXT_MAP[SAVE_FORMAT]
    bin_stem = os.path.splitext(os.path.basename(INPUT_FILE))[0]

    print(f"[CFG]  samples/seg = {samples_per_seg:,} | segments = {num_segments}")
    print(f"[CFG]  STFT = {STFT_POINT} pts | Overlap = {OVERLAP} | noverlap = {noverlap}")
    print(f"[CFG]  Backend = {'CUDA' if USE_CUDA else 'CPU'}")
    print(f"[CFG]  Save format = {SAVE_FORMAT} | {IMG_WIDTH}x{IMG_HEIGHT} px")
    print(f"[CFG]  Colormap = {CMAP} | Output = {OUTPUT_DIR}")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── CUDA context ────────────────────────────────────────────
    cuda_ctx = None
    if USE_CUDA:
        try:
            cuda_ctx = CudaSTFTContext(STFT_POINT, noverlap, SAMPLE_RATE)
            print("[CUDA] Context sẵn sàng.")
        except Exception as e:
            print(f"[CUDA] Khởi tạo thất bại: {e}. Fallback CPU.")

    # ── Xử lý từng segment ─────────────────────────────────────
    t_total = time.perf_counter()
    futures = []

    with ThreadPoolExecutor(max_workers=N_SAVE_WORKERS,
                            thread_name_prefix="SaveWorker") as executor:
        for seg_idx in range(num_segments):
            start   = seg_idx * samples_per_seg
            segment = iq_all[start:start + samples_per_seg]

            t_stft = time.perf_counter()
            try:
                if USE_CUDA and cuda_ctx is not None:
                    power = cuda_ctx.compute(segment)
                else:
                    power = compute_stft_cpu(segment, STFT_POINT, noverlap)
            except Exception as e:
                print(f"[STFT ERROR] Segment {seg_idx}: {e}")
                continue
            dt_stft = (time.perf_counter() - t_stft) * 1000

            fname     = f"{bin_stem}_seg{seg_idx:04d}{ext}"
            save_path = os.path.join(OUTPUT_DIR, fname)

            fut = executor.submit(save_task, power.copy(), save_path, seg_idx, dt_stft)
            futures.append(fut)

        for fut in futures:
            fut.result()

    dt_total = time.perf_counter() - t_total
    print("=" * 60)
    print(f"[DONE] {num_segments} anh | Tong: {dt_total:.2f} s "
          f"| TB: {dt_total / num_segments * 1000:.1f} ms/anh")
    print(f"[DONE] Anh luu tai: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
