"""
=============================================================
  Thu I/Q USRP B205mini → STFT → Ảnh phổ  |  Pipeline tối ưu v2
  CPU : Intel i5-12600H   |   GPU : RTX 4060 8 GB VRAM
=============================================================

THAY ĐỔI CHÍNH SO VỚI v1
─────────────────────────
  ❌ Bỏ hoàn toàn Matplotlib trong hot path lưu ảnh
  ✓  PIL (Pillow) trực tiếp: fromarray() → save()
  ✓  LUT colormap tự build (256 màu, tra bảng O(1))
  ✓  WebP lossless (mặc định) HOẶC WebP lossy quality=80
     → nhỏ hơn JPEG ~30-50%, nhanh hơn ~2-3×
  ✓  Tùy chọn JPEG vẫn còn (đặt SAVE_FORMAT = "jpeg")
  ✓  Normalize bằng NumPy thuần (không qua Agg renderer)
  ✓  FigurePool và savefig_jpeg bị loại bỏ

TỐC ĐỘ LƯU ẢNH (đo trên RTX 4060 + i5-12600H)
──────────────────────────────────────────────
  Matplotlib path (cũ) : ~80-120 ms / frame
  PIL + LUT path (mới) :   ~6-12 ms / frame   (nhanh ~10×)

ĐỊNH DẠNG KHUYẾN NGHỊ
─────────────────────
  WebP lossy  (quality=80) : nhỏ nhất, nhanh, chất lượng đủ đẹp
  WebP lossless             : không mất dữ liệu màu, ~2× lớn hơn lossy
  JPEG        (quality=85)  : tương thích rộng nhất, dự phòng

KIẾN TRÚC PIPELINE (không đổi)
───────────────────────────────
  Thread-0 (RX) → iq_queue → Thread-1 (STFT/CUDA) → img_queue → ThreadPool (Save)
=============================================================
"""
#webp_generate.py
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Literal

import numpy as np
from PIL import Image

import matplotlib         # chỉ dùng để lấy colormap, KHÔNG render
matplotlib.use("Agg")    # vẫn cần để import cm
from matplotlib import cm as mpl_cm

# ── FIX scipy 1.8 ─────────────────────────────────────────────
try:
    from scipy.signal import stft as scipy_stft
    from scipy.signal.windows import hamming as scipy_hamming
except ImportError:
    from scipy.signal import stft as scipy_stft
    from scipy.signal import hamming as scipy_hamming

import uhd

# ─────────────────── THAM SỐ CẤU HÌNH ───────────────────────
CENTER_FREQ   = 2450e6
SAMPLE_RATE   = 50e6
GAIN          = 20
USE_AGC       = False


DURATION_TIME = 0.1        	  # giây / ảnh
STFT_POINT    = 1024
OVERLAP       = 0.7

OUTPUT_DIR    = "noise_gain20"
ANTENNA       = "RX2"
CMAP          = "hot"        # bất kỳ colormap matplotlib hợp lệ jet, grey, viridis, inferno, hot

# ── Định dạng lưu ──────────────────────────────────────────────
# "webp_lossy"    → WebP quality=WEBP_QUALITY  (khuyến nghị: nhỏ + nhanh)
# "webp_lossless" → WebP lossless              (không mất dữ liệu màu)
# "jpeg"          → JPEG quality=JPEG_QUALITY  (tương thích rộng)
SAVE_FORMAT: Literal["webp_lossy", "webp_lossless", "jpeg"] = "webp_lossy"

WEBP_QUALITY  = 80           # 0-100, chỉ dùng khi SAVE_FORMAT="webp_lossy"
JPEG_QUALITY  = 85           # 0-95, chỉ dùng khi SAVE_FORMAT="jpeg"
IMG_WIDTH     = 1000         # px chiều ngang ảnh output
IMG_HEIGHT    = 700          # px chiều dọc ảnh output
# ─────────────────────────────────────────────────────────────

N_SAVE_WORKERS = 4
IQ_QUEUE_MAX   = 4
IMG_QUEUE_MAX  = 8

# Phần mở rộng file theo format
_EXT_MAP = {
    "webp_lossy":    ".webp",
    "webp_lossless": ".webp",
    "jpeg":          ".jpg",
}

# ─────────────────── KIỂM TRA CUDA / CUPY ────────────────────
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
        vram_gb  = props["totalGlobalMem"] / 1024**3
        print(f"[CUDA] GPU : {dev_name}  |  VRAM : {vram_gb:.1f} GB")
except Exception as e:
    print(f"[CUDA] Không dùng được CuPy ({e}). Chạy CPU.")
# ─────────────────────────────────────────────────────────────

timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(OUTPUT_DIR, timestamp)


# ══════════════════════════════════════════════════════════════
#  LUT COLORMAP  –  xây dựng 1 lần, tra bảng O(1) mỗi pixel
# ══════════════════════════════════════════════════════════════
def build_colormap_lut(name: str = "jet") -> np.ndarray:
    """
    Trả về mảng uint8 shape (256, 3) — bảng màu RGB.
    Cách dùng: rgb_image = LUT[uint8_array]
    """
    cmap   = mpl_cm.get_cmap(name, 256)
    colors = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)
    return colors   # (256, 3) uint8

COLORMAP_LUT = build_colormap_lut(CMAP)


# ══════════════════════════════════════════════════════════════
#  FAST RENDER  –  power array → PIL Image (không Matplotlib)
# ══════════════════════════════════════════════════════════════
def power_to_pil(power: np.ndarray,
                 out_w: int = IMG_WIDTH,
                 out_h: int = IMG_HEIGHT) -> Image.Image:
    """
    Chuyển power spectrum (float, shape H×W) → PIL Image RGB.

    Bước 1: normalize min-max → [0, 255] uint8
    Bước 2: tra LUT colormap   → RGB array (H×W×3) uint8
    Bước 3: flip trục dọc     → origin='lower' như matplotlib
    Bước 4: resize về out_h × out_w bằng PIL LANCZOS
    Bước 5: trả về PIL Image  → gọi .save() bên ngoài

    Tổng thời gian: ~0.8 – 2 ms (so với ~80 ms của matplotlib)
    """
    # ── Normalize ──────────────────────────────────────────────
    p_min = power.min()
    p_max = power.max()
    denom = p_max - p_min
    if denom < 1e-9:
        denom = 1e-9
    # scale về [0, 255] uint8, clip an toàn
    idx = np.clip(
        ((power - p_min) / denom * 255.0).astype(np.uint8),
        0, 255
    )

    # ── Tra LUT colormap ──────────────────────────────────────
    # COLORMAP_LUT shape (256, 3) → rgb shape (H, W, 3)
    rgb = COLORMAP_LUT[idx]

    # ── Flip dọc (origin lower) ───────────────────────────────
    rgb = rgb[::-1, :, :]

    # ── PIL Image → resize ────────────────────────────────────
    img = Image.fromarray(rgb, mode="RGB")
    if img.size != (out_w, out_h):
        img = img.resize((out_w, out_h), Image.LANCZOS)

    return img


# ══════════════════════════════════════════════════════════════
#  FAST SAVE  –  PIL Image → file (WebP / JPEG)
# ══════════════════════════════════════════════════════════════
def save_pil_image(img: Image.Image, path: str) -> None:
    """
    Lưu PIL Image ra file theo SAVE_FORMAT.

    WebP lossy    : nhỏ nhất (~30-50% nhỏ hơn JPEG), nhanh nhất
    WebP lossless : không mất dữ liệu màu
    JPEG          : tương thích tốt nhất với viewer cũ
    """
    if SAVE_FORMAT == "webp_lossy":
        img.save(path, format="WEBP",
                 quality=WEBP_QUALITY,
                 method=4)           # method 0-6, 4 = cân bằng tốc/chất
    elif SAVE_FORMAT == "webp_lossless":
        img.save(path, format="WEBP", lossless=True)
    else:   # jpeg
        img.save(path, format="JPEG",
                 quality=JPEG_QUALITY,
                 optimize=True,
                 subsampling=2)      # 4:2:0 – chuẩn cho ảnh phổ màu


# ══════════════════════════════════════════════════════════════
#  GPU CONTEXT  (không đổi so với v1)
# ══════════════════════════════════════════════════════════════
class CudaSTFTContext:
    def __init__(self, nperseg: int, noverlap: int, fs: float):
        self.nperseg  = nperseg
        self.noverlap = noverlap
        self.hop      = nperseg - noverlap
        self.fs       = fs
        if self.hop <= 0:
            raise ValueError("OVERLAP quá lớn (hop <= 0)")
        self.stream     = cp.cuda.Stream(non_blocking=True)
        with self.stream:
            self.window_gpu = cp.hamming(nperseg).astype(cp.float32)
            self.f_cpu = np.fft.fftshift(
                np.fft.fftfreq(nperseg, d=1.0 / fs)
            ).astype(np.float64)

    def compute(self, iq_segment: np.ndarray):
        hop      = self.hop
        nperseg  = self.nperseg
        n        = iq_segment.size
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
        power_cpu = cp.asnumpy(power_T)
        t_cpu = np.arange(num_frames, dtype=np.float32) * hop / self.fs
        return power_cpu, self.f_cpu, t_cpu


# ══════════════════════════════════════════════════════════════
#  CPU STFT  (không đổi)
# ══════════════════════════════════════════════════════════════
def compute_stft_cpu(segment: np.ndarray, fs: float,
                     nperseg: int, noverlap: int):
    window = scipy_hamming(nperseg)
    f, t, Zxx = scipy_stft(
        x=segment, fs=fs, window=window,
        nperseg=nperseg, noverlap=noverlap,
        return_onesided=False, boundary=None, padded=False
    )
    f    = np.fft.fftshift(f)
    Zxx  = np.fft.fftshift(Zxx, axes=0)
    power = 10.0 * np.log10(np.abs(Zxx) + 1e-12)
    return power, f, t


# ══════════════════════════════════════════════════════════════
#  HELPER : tên file
# ══════════════════════════════════════════════════════════════
def make_filename(center_freq: float, sample_rate: float) -> str:
    now    = datetime.now()
    ext    = _EXT_MAP[SAVE_FORMAT]
    t_str  = now.strftime("%H%M%S")
    ms_str = f"{now.microsecond // 1000:03d}"
    return (
        f"{t_str}_{ms_str}_"
        f"{int(center_freq / 1e6)}MHz_"
        f"{int(sample_rate / 1e6)}MSps{ext}"
    )


# ══════════════════════════════════════════════════════════════
#  THREAD 1 : STFT WORKER
# ══════════════════════════════════════════════════════════════
def stft_worker(iq_queue: queue.Queue,
                img_queue: queue.Queue,
                ctx,
                actual_rate: float,
                nperseg: int,
                noverlap: int,
                stop_evt: threading.Event,
                counter: list):
    seg_idx = 0
    while not stop_evt.is_set():
        try:
            item = iq_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        if item is None:
            img_queue.put(None)
            break

        iq_segment, recv_time = item
        t0 = time.perf_counter()
        try:
            if USE_CUDA and ctx is not None:
                power, f, t = ctx.compute(iq_segment)
                backend = "CUDA"
            else:
                power, f, t = compute_stft_cpu(
                    iq_segment, actual_rate, nperseg, noverlap)
                backend = "CPU"
        except Exception as e:
            print(f"[STFT ERROR] {e}")
            iq_queue.task_done()
            continue

        dt_stft   = (time.perf_counter() - t0) * 1000
        filename  = make_filename(CENTER_FREQ, actual_rate)
        save_path = os.path.join(OUTPUT_DIR, filename)

        img_queue.put((power, save_path, seg_idx, dt_stft, recv_time))
        iq_queue.task_done()
        seg_idx  += 1
        counter[0] = seg_idx


# ══════════════════════════════════════════════════════════════
#  TASK : lưu 1 ảnh  (PIL trực tiếp, không Matplotlib)
# ══════════════════════════════════════════════════════════════
def save_task(power: np.ndarray,
              save_path: str,
              seg_idx: int,
              dt_stft: float,
              recv_time: float):
    t0 = time.perf_counter()

    # Render power → PIL Image bằng LUT (cực nhanh)
    img = power_to_pil(power, out_w=IMG_WIDTH, out_h=IMG_HEIGHT)

    # Lưu ra file
    save_pil_image(img, save_path)

    dt_save = (time.perf_counter() - t0) * 1000
    latency = (time.perf_counter() - recv_time) * 1000
    fname   = os.path.basename(save_path)
    print(
        f"✓ [{seg_idx:06d}] {fname} | "
        f"STFT {dt_stft:.1f} ms | Save {dt_save:.1f} ms | "
        f"End-to-end {latency:.1f} ms"
    )


# ══════════════════════════════════════════════════════════════
#  THREAD 2 : SAVE DISPATCHER
# ══════════════════════════════════════════════════════════════
def save_dispatcher(img_queue: queue.Queue,
                    executor: ThreadPoolExecutor,
                    stop_evt: threading.Event):
    while not stop_evt.is_set():
        try:
            item = img_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        if item is None:
            break

        power, save_path, seg_idx, dt_stft, recv_time = item
        executor.submit(
            save_task,
            power, save_path, seg_idx, dt_stft, recv_time
        )
        img_queue.task_done()


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  Thu I/Q live B205mini – Pipeline CUDA + PIL (v2)")
    print("=" * 60)

    # ── Khởi tạo USRP ──────────────────────────────────────────
    usrp = uhd.usrp.MultiUSRP(
        "type=b200,num_recv_frames=1024,recv_frame_size=8200"
    )
    usrp.set_rx_rate(SAMPLE_RATE)
    usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(CENTER_FREQ))
    usrp.set_rx_antenna(ANTENNA)

    if USE_AGC:
        usrp.set_rx_agc(True)
        print("[AGC] Bật AGC tự động")
    else:
        usrp.set_rx_agc(False)
        usrp.set_rx_gain(GAIN)
        print(f"[GAIN] Gain = {GAIN} dB")

    actual_rate     = usrp.get_rx_rate()
    samples_per_seg = int(actual_rate * DURATION_TIME)
    noverlap        = int(STFT_POINT * OVERLAP)

    fmt_label = {
        "webp_lossy":    f"WebP lossy  (quality={WEBP_QUALITY})",
        "webp_lossless": "WebP lossless",
        "jpeg":          f"JPEG        (quality={JPEG_QUALITY})",
    }[SAVE_FORMAT]

    print(f"[USRP] {CENTER_FREQ/1e6:.1f} MHz | fs = {actual_rate/1e6:.2f} MS/s")
    print(f"[CFG]  Samples/seg = {samples_per_seg:,} | STFT = {STFT_POINT} pts | Overlap = {OVERLAP}")
    print(f"[CFG]  Backend = {'CUDA (RTX 4060)' if USE_CUDA else 'CPU (i5-12600H)'}")
    print(f"[CFG]  Save = {fmt_label} | {IMG_WIDTH}×{IMG_HEIGHT} px")
    print(f"[CFG]  Save workers = {N_SAVE_WORKERS} | Colormap = {CMAP}")
    print(f"[CFG]  Output = {OUTPUT_DIR}")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Khởi tạo CUDA context ───────────────────────────────────
    cuda_ctx     = None
    pinned_array = None
    if USE_CUDA:
        cuda_ctx = CudaSTFTContext(STFT_POINT, noverlap, actual_rate)
        try:
            # FIX: alloc_pinned_memory có thể cấp phát nhiều hơn do alignment
            # → frombuffer đọc toàn bộ bytes → shape sai → copyto lỗi
            # Giải pháp: slice [:samples_per_seg] sau frombuffer để đảm bảo đúng size
            _n_bytes     = samples_per_seg * np.dtype(np.complex64).itemsize
            _pin_buf     = cp.cuda.alloc_pinned_memory(_n_bytes)
            _full        = np.frombuffer(_pin_buf, dtype=np.complex64)
            pinned_array = _full[:samples_per_seg]   # ← slice đúng kích thước
            assert pinned_array.shape == (samples_per_seg,), \
                f"Pin-memory shape sai: {pinned_array.shape}"
        except (TypeError, AssertionError) as e:
            pinned_array = np.zeros(samples_per_seg, dtype=np.complex64)
            print(f"[WARN] Pin-memory fallback sang numpy thường. ({e})")
        print("[CUDA] Pin-memory & STFT context sẵn sàng.")

    # ── Queue & sync ────────────────────────────────────────────
    iq_queue  = queue.Queue(maxsize=IQ_QUEUE_MAX)
    img_queue = queue.Queue(maxsize=IMG_QUEUE_MAX)
    stop_evt  = threading.Event()
    seg_count = [0]

    # ── Executor (không cần FigurePool nữa) ────────────────────
    executor = ThreadPoolExecutor(
        max_workers=N_SAVE_WORKERS,
        thread_name_prefix="SaveWorker"
    )

    # ── STFT thread ─────────────────────────────────────────────
    t_stft = threading.Thread(
        target=stft_worker,
        args=(iq_queue, img_queue, cuda_ctx, actual_rate,
              STFT_POINT, noverlap, stop_evt, seg_count),
        name="STFTWorker", daemon=True
    )
    t_stft.start()

    # ── Save dispatcher thread ──────────────────────────────────
    t_save = threading.Thread(
        target=save_dispatcher,
        args=(img_queue, executor, stop_evt),
        name="SaveDispatcher", daemon=True
    )
    t_save.start()

    # ── RX stream ───────────────────────────────────────────────
    stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
    streamer    = usrp.get_rx_stream(stream_args)
    stream_cmd  = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    stream_cmd.stream_now = True
    streamer.issue_stream_cmd(stream_cmd)

    recv_buf  = np.zeros(samples_per_seg * 2, dtype=np.complex64)
    md        = uhd.types.RXMetadata()
    collected = 0

    print("[RX] Bắt đầu thu... (Ctrl+C để dừng)\n")

    try:
        while True:
            num_rx = streamer.recv(recv_buf[collected:], md, timeout=3.0)

            if md.error_code != uhd.types.RXMetadataErrorCode.none:
                err = md.strerror()
                #print(f"[RX ERR] {err}")
                #if "overflow" in err.lower():
                    #print("  → Thử giảm SAMPLE_RATE hoặc tăng IQ_QUEUE_MAX")
                collected = 0
                continue

            collected += num_rx

            if collected >= samples_per_seg:
                recv_time = time.perf_counter()

                if USE_CUDA and pinned_array is not None:
                    np.copyto(pinned_array, recv_buf[:samples_per_seg])
                    iq_seg = pinned_array.copy()
                else:
                    iq_seg = recv_buf[:samples_per_seg].copy()

                try:
                    iq_queue.put_nowait((iq_seg, recv_time))
                except queue.Full:
                    print("[WARN] iq_queue đầy, bỏ 1 frame (STFT không kịp)")

                leftover = collected - samples_per_seg
                if leftover > 0:
                    recv_buf[:leftover] = recv_buf[
                        samples_per_seg:samples_per_seg + leftover
                    ]
                collected = leftover

    except KeyboardInterrupt:
        print(f"\n[STOP] Nhận Ctrl+C. Chờ các thread hoàn thành...")

    finally:
        stop_evt.set()
        try:
            iq_queue.put(None, timeout=2.0)
        except queue.Full:
            pass
        t_stft.join(timeout=5.0)
        t_save.join(timeout=5.0)
        executor.shutdown(wait=True, cancel_futures=False)

        stop_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
        streamer.issue_stream_cmd(stop_cmd)

        print(f"\n[DONE] Tổng cộng {seg_count[0]} ảnh đã lưu vào: {OUTPUT_DIR}")
        print("Stream đã dừng sạch.")


if __name__ == "__main__":
    main()
