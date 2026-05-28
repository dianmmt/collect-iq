# =============================================================
#  config.py — Tham số trung tâm cho toàn bộ dự án
#  Chỉnh sửa tại đây, tất cả file còn lại sẽ dùng theo.
# =============================================================

# ── B205 mini — SDR ──────────────────────────────────────────
CENTER_FREQ   = 2450e6      # Hz — tần số trung tâm
SAMPLE_RATE   = 50e6        # Hz — tốc độ lấy mẫu (tối đa ~56 MS/s)
GAIN          = 0           # dB — gain thu
BANDWIDTH     = None        # Hz — băng thông analog (None = tự động)
ANTENNA       = "RX2"       # cổng antenna: "RX2" hoặc "TX/RX"

# ── X310 — SDR ───────────────────────────────────────────────
X310_ADDR         = "192.168.40.2"
X310_SUBDEV       = "B:0"
X310_SAMPLE_RATE  = 100e6   # Hz
X310_BANDWIDTH    = 100e6   # Hz

# ── Thu thập IQ ──────────────────────────────────────────────
DURATION_SEC      = 0.1     # giây / chunk
OUTPUT_FMT        = "bin"   # fc32 | sc16 | npy | bin
IQ_OUTPUT_DIR     = "iq_capture"

# ── STFT / Spectrogram ───────────────────────────────────────
STFT_POINT        = 1024
OVERLAP           = 0.7     # tỉ lệ overlap (0.0 – <1.0)

# ── Ảnh đầu ra ───────────────────────────────────────────────
SPEC_OUTPUT_DIR   = "spectrogram_output"
CMAP              = "jet"   # jet | hot | viridis | inferno | grey | ...

# Định dạng lưu ảnh:
#   "webp_lossy"    → WebP lossy  (nhỏ + nhanh, khuyến nghị)
#   "webp_lossless" → WebP lossless (không mất dữ liệu màu)
#   "jpeg"          → JPEG
#   "png"           → PNG lossless
SAVE_FORMAT       = "webp_lossy"
WEBP_QUALITY      = 80      # 0-100, dùng khi SAVE_FORMAT="webp_lossy"
JPEG_QUALITY      = 85      # 0-95,  dùng khi SAVE_FORMAT="jpeg"
IMG_WIDTH         = 1000    # px chiều ngang ảnh
IMG_HEIGHT        = 700     # px chiều dọc   ảnh
N_SAVE_WORKERS    = 4       # số thread lưu ảnh song song

# ── Phân tích tín hiệu ───────────────────────────────────────
NF_FFT_SIZE           = 1024
NF_PERCENTILE         = 10.0
NF_METHOD             = "welch"   # welch | median | time | all
SIGNAL_THRESHOLD_FACTOR = 10.0     # threshold = noise_floor × factor
NPERSEG               = 1024      # độ dài segment Welch

# ── Realtime spectrum (bdts.py) ──────────────────────────────
BDTS_GAIN       = 0
BDTS_PATCH_SIZE = 1024
BDTS_YMIN       = -150      # dBFS — giới hạn trục Y dưới
BDTS_YMAX       = 0         # dBFS — giới hạn trục Y trên



POWER_DB_MIN   = -100.0   # dB → màu tối nhất
POWER_DB_MAX   =  -50.0   # dB → màu sáng nhất
AUTO_DB_RANGE  = False    # True để tự đo lần đầu
