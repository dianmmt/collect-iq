"""
analyze_iq_fft.py  (v2 — unified 10*log10 power dBFS)
------------------------------------------------------
Phân tích FFT từ file IQ thu bằng USRP B205 mini.

Thay đổi so với v1:
  - [FIX 1] Thống nhất công thức dBFS sang 10*log10(power) thay vì 20*log10(amp)
            → nhất quán với iq_awgn_inject.py v5
            → đường mid-line và noise floor đọc đúng giá trị power dBFS

  Công thức mới:
      power_per_bin = (|FFT(x * win)| / nfft)^2        ← bình phương biên độ normalized
      power_avg     = mean(power_per_bin_frame_1, ...)  ← averaging trên power
      amp_db        = 10 * log10(power_avg)             ← POWER dBFS

  Lưu ý: kết quả số sẽ khác v1 (power dB ≠ amplitude dB).
  Thang đo mới nhất quán với cách inject script tính noise floor.

Cách dùng:
  python analyze_iq_fft.py --file capture.bin --rate 50e6 --freq 2.45e9
  python analyze_iq_fft.py --file capture.sc16 --rate 50e6 --fmt sc16
  python analyze_iq_fft.py --file capture.npy  --rate 50e6 --save out.png

Yêu cầu: numpy, matplotlib  (pip install numpy matplotlib)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ──────────────────────────────────────────────
#  ĐỌC FILE IQ
# ──────────────────────────────────────────────

def load_iq(file_path: str, fmt: str = "auto") -> np.ndarray:
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

    if fmt == "auto":
        suffix = p.suffix.lower()
        if suffix in (".cf32", ".bin"):
            fmt = "fc32"
        elif suffix == ".sc16":
            fmt = "sc16"
        elif suffix == ".npy":
            fmt = "npy"
        else:
            fmt = "fc32"
            print(f"[WARN] Không nhận ra đuôi '{suffix}', thử đọc như fc32.")

    if fmt in ("fc32", "bin"):
        raw = np.fromfile(p, dtype=np.float32)
        if raw.size % 2 != 0:
            raw = raw[:-1]
        samples = (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)

    elif fmt == "sc16":
        raw = np.fromfile(p, dtype=np.int16)
        if raw.size % 2 != 0:
            raw = raw[:-1]
        samples = (
            (raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32))
            / 32767.0
        ).astype(np.complex64)

    elif fmt == "npy":
        samples = np.load(p).astype(np.complex64)

    else:
        raise ValueError(f"fmt không hợp lệ: '{fmt}'")

    print(f"[LOAD] {p.name}  |  {len(samples):,} samples  |  format={fmt}")
    return samples


# ──────────────────────────────────────────────
#  TÍNH FFT  — Welch averaging, 10*log10 POWER dBFS
# ──────────────────────────────────────────────

def compute_fft(
    samples: np.ndarray,
    sample_rate: float,
    center_freq: float = 0.0,
    nfft: int = 1024,
    window: str = "hann",
    overlap: float = 0.5,
    avg_count: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Welch averaging với công thức POWER dBFS (nhất quán với inject script):

        Mỗi frame:
            spec      = FFT(seg * win)
            power_bin = |spec / nfft|^2        ← normalized power per bin

        Trung bình trên POWER (không phải biên độ):
            power_avg = mean(power_frame_1, power_frame_2, ...)

        Kết quả:
            amp_db = 10 * log10(power_avg)     ← POWER dBFS

    So với v1 (20*log10 amplitude): kết quả số khác nhưng thang đo nhất quán
    với iq_awgn_inject.py — noise floor và mid-line đọc đúng power dBFS.
    """
    n = len(samples)
    if n < nfft:
        raise ValueError(f"File chỉ có {n} samples, cần ít nhất {nfft}.")

    wins = {
        "hann":        np.hanning(nfft),
        "hamming":     np.hamming(nfft),
        "blackman":    np.blackman(nfft),
        "rectangular": np.ones(nfft),
    }
    win  = wins.get(window, np.hanning(nfft)).astype(np.float32)
    step = max(1, int(nfft * (1 - overlap)))

    # Tích lũy POWER (không phải biên độ)
    power_sum = np.zeros(nfft, dtype=np.float64)
    count     = 0

    for start in range(0, n - nfft + 1, step):
        if count >= avg_count:
            break
        seg        = samples[start : start + nfft] * win
        spec       = np.fft.fftshift(np.fft.fft(seg))
        power_sum += (np.abs(spec) / nfft) ** 2    # power per bin (normalized)
        count     += 1

    if count == 0:
        raise ValueError(f"Không đủ samples ({n}) cho NFFT={nfft}.")

    power_avg = power_sum / count                   # average power
    amp_db    = 10.0 * np.log10(power_avg + 1e-30)  # POWER dBFS — nhất quán với inject

    freqs = (np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / sample_rate))
             + center_freq)

    print(f"[FFT]  NFFT={nfft}, window={window}, overlap={overlap}, avg={count} frames")
    return freqs, amp_db


# ──────────────────────────────────────────────
#  PHÂN TÍCH VÀ VẼ ĐỒ THỊ
# ──────────────────────────────────────────────

def analyze_and_plot(
    freqs: np.ndarray,
    amp_db: np.ndarray,
    sample_rate: float,
    center_freq: float = 0.0,
    title: str = "Spectrum",
    save_fig: str = None,
):
    idx_max  = int(np.argmax(amp_db))
    idx_min  = int(np.argmin(amp_db))
    amp_max  = float(amp_db[idx_max])
    amp_min  = float(amp_db[idx_min])
    freq_max = float(freqs[idx_max])
    freq_min = float(freqs[idx_min])
    amp_mid  = (amp_max + amp_min) / 2.0

    # Noise floor: median của bottom-40% bins (nhất quán với inject script)
    p40            = float(np.percentile(amp_db, 40))
    noise_floor_db = float(np.median(amp_db[amp_db < p40]))

    print("=" * 60)
    print(f"  MAX         : {amp_max:+.2f} dBFS  @  {freq_max/1e6:.6f} MHz")
    print(f"  MIN         : {amp_min:+.2f} dBFS  @  {freq_min/1e6:.6f} MHz")
    print(f"  MID         : {amp_mid:+.2f} dBFS")
    print(f"  Noise floor : {noise_floor_db:+.2f} dBFS  (median bottom-40% bins)")
    print(f"  [Thang đo   : 10*log10(power) — POWER dBFS]")
    print("=" * 60)

    freqs_mhz = freqs / 1e6
    fig, ax   = plt.subplots(figsize=(14, 6), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")

    ax.plot(freqs_mhz, amp_db, color="#00bfff", linewidth=0.8, alpha=0.9,
            label="Spectrum")

    # Mid line
    ax.axhline(amp_mid, color="red", linewidth=1.8, linestyle="--",
               label=f"Mid = {amp_mid:+.2f} dBFS", zorder=5)

    # Noise floor line
    ax.axhline(noise_floor_db, color="#aaffaa", linewidth=1.2, linestyle=":",
               label=f"Noise floor = {noise_floor_db:+.2f} dBFS", zorder=5)

    # MAX marker
    ax.plot(freq_max/1e6, amp_max, marker="^", color="#ffdd00", markersize=9,
            zorder=6,
            label=f"Max: {amp_max:+.2f} dBFS @ {freq_max/1e6:.4f} MHz")
    ax.annotate(
        f" MAX\n {freq_max/1e6:.4f} MHz\n {amp_max:+.2f} dBFS",
        xy=(freq_max/1e6, amp_max), xytext=(15, -5),
        textcoords="offset points", color="#ffdd00", fontsize=7.5,
        fontfamily="monospace",
    )

    # MIN marker
    ax.plot(freq_min/1e6, amp_min, marker="v", color="#ff6b6b", markersize=9,
            zorder=6,
            label=f"Min: {amp_min:+.2f} dBFS @ {freq_min/1e6:.4f} MHz")
    ax.annotate(
        f" MIN\n {freq_min/1e6:.4f} MHz\n {amp_min:+.2f} dBFS",
        xy=(freq_min/1e6, amp_min), xytext=(15, 5),
        textcoords="offset points", color="#ff6b6b", fontsize=7.5,
        fontfamily="monospace",
    )

    ax.set_xlabel("Frequency (MHz)", color="#c9d1d9", fontsize=11)
    ax.set_ylabel("Power (dBFS)",    color="#c9d1d9", fontsize=11)   # ← đổi label
    ax.set_title(title, color="white", fontsize=13, pad=12)
    ax.tick_params(colors="#8b949e", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.grid(True, color="#21262d", linewidth=0.6, linestyle="-")
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(5))
    ax.grid(True, which="minor", color="#161b22", linewidth=0.4)
    ax.legend(loc="upper right", fontsize=8,
              facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")

    info_str = (
        f"SR: {sample_rate/1e6:.2f} MS/s"
        + (f"  |  CF: {center_freq/1e6:.4f} MHz" if center_freq != 0 else "")
        + "  |  scale: 10·log₁₀(power)"
    )
    ax.text(0.01, 0.01, info_str, transform=ax.transAxes,
            color="#8b949e", fontsize=8, fontfamily="monospace",
            verticalalignment="bottom")

    plt.tight_layout()

    if save_fig:
        fig.savefig(save_fig, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[SAVE] {save_fig}")

    plt.show()

    return {
        "freq_max_hz":    freq_max,
        "amp_max_dbfs":   amp_max,
        "freq_min_hz":    freq_min,
        "amp_min_dbfs":   amp_min,
        "amp_mid_dbfs":   amp_mid,
        "noise_floor_db": noise_floor_db,
    }


# ──────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phân tích FFT file IQ từ USRP B205 mini (v2 — power dBFS)."
    )
    parser.add_argument("--file",    required=True,
                        help="File IQ (.cf32 .sc16 .npy .bin)")
    parser.add_argument("--rate",    type=float, required=True,
                        help="Sample rate Hz, vd: 50e6")
    parser.add_argument("--freq",    type=float, default=0.0,
                        help="Center freq Hz, vd: 2.45e9")
    parser.add_argument("--fmt",     type=str,   default="auto",
                        choices=["auto", "fc32", "sc16", "npy", "bin"])
    parser.add_argument("--nfft",    type=int,   default=1024,
                        help="Số điểm FFT (mặc định: 1024)")
    parser.add_argument("--window",  type=str,   default="hann",
                        choices=["hann", "hamming", "blackman", "rectangular"])
    parser.add_argument("--avg",     type=int,   default=50,
                        help="Số frame trung bình (mặc định: 50)")
    parser.add_argument("--overlap", type=float, default=0.5,
                        help="Độ phủ khung 0.0–0.9 (mặc định: 0.5)")
    parser.add_argument("--save",    type=str,   default=None,
                        help="Lưu PNG, vd: out.png")
    args = parser.parse_args()

    samples = load_iq(args.file, fmt=args.fmt)

    freqs, amp_db = compute_fft(
        samples,
        sample_rate=args.rate,
        center_freq=args.freq,
        nfft=args.nfft,
        window=args.window,
        overlap=args.overlap,
        avg_count=args.avg,
    )

    analyze_and_plot(
        freqs, amp_db,
        sample_rate=args.rate,
        center_freq=args.freq,
        title=f"IQ Spectrum  —  {Path(args.file).name}",
        save_fig=args.save,
    )


if __name__ == "__main__":
    main()
