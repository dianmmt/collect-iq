#!/usr/bin/env python3
"""
b205_realtime_spectrum.py  —  producer/consumer thread version
UHD chạy thread riêng → queue → main thread vẽ
Ctrl+C để dừng
"""

import signal
import sys
import time
import threading
import queue
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import uhd

from config import (
    CENTER_FREQ, SAMPLE_RATE,
    BDTS_GAIN as GAIN,
    BDTS_PATCH_SIZE as PATCH_SIZE,
    BDTS_YMIN as YMIN,
    BDTS_YMAX as YMAX,
)

# ─────────────────────────────────────────────
# THAM SỐ
# ─────────────────────────────────────────────
Q_MAXSIZE = 4   # tránh queue tràn khi vẽ chậm

# ─────────────────────────────────────────────
# STOP EVENT
# ─────────────────────────────────────────────
stop_event = threading.Event()

def handle_sigint(sig, frame):
    print("\n[INFO] Ctrl+C — đang dừng...")
    stop_event.set()

signal.signal(signal.SIGINT, handle_sigint)

# ─────────────────────────────────────────────
# KHỞI TẠO USRP (gọi trong UHD thread)
# ─────────────────────────────────────────────
def make_streamer():
    global PATCH_SIZE
    usrp = uhd.usrp.MultiUSRP("type=b200,num_recv_frames=1024,recv_frame_size=8200")
    usrp.set_rx_rate(SAMPLE_RATE, 0)
    usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(CENTER_FREQ), 0)
    usrp.set_rx_gain(GAIN, 0)
    usrp.set_rx_antenna("RX2", 0)
    time.sleep(0.5)

    st = uhd.usrp.StreamArgs("fc32", "sc16")
    st.channels = [0]
    rx = usrp.get_rx_stream(st)

    max_s = rx.get_max_num_samps()
    PATCH_SIZE = min(PATCH_SIZE, max_s)
    print(f"[INFO] max_num_samps={max_s}, PATCH_SIZE={PATCH_SIZE}")

    cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    cmd.stream_now = True
    rx.issue_stream_cmd(cmd)

    # flush
    buf = np.zeros((1, PATCH_SIZE), dtype=np.complex64)
    md  = uhd.types.RXMetadata()
    for _ in range(20):
        rx.recv(buf, md, timeout=2.0)

    return rx

# ─────────────────────────────────────────────
# UHD PRODUCER THREAD
# ─────────────────────────────────────────────
def uhd_producer(spec_queue: queue.Queue):
    """
    Chạy hoàn toàn trong thread riêng.
    Không đụng matplotlib.
    """
    try:
        rx = make_streamer()
        buf = np.zeros((1, PATCH_SIZE), dtype=np.complex64)
        md  = uhd.types.RXMetadata()
        win = np.hanning(PATCH_SIZE)

        while not stop_event.is_set():
            n = rx.recv(buf, md, timeout=2.0)

            if md.error_code not in (
                uhd.types.RXMetadataErrorCode.none,
                uhd.types.RXMetadataErrorCode.overflow,
            ):
                print(f"[WARN] {metadata.strerror()}")
                continue

            if n < PATCH_SIZE:
                continue

            iq   = buf[0, :PATCH_SIZE].copy()   # copy tránh race condition
            X    = np.fft.fftshift(np.fft.fft(iq * win))
            spec = 20 * np.log10(np.abs(X) / PATCH_SIZE + 1e-12)

            # non-blocking put — bỏ qua nếu queue đầy (consumer chậm)
            try:
                spec_queue.put_nowait(spec)
            except queue.Full:
                pass

        # dừng stream
        stop_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
        rx.issue_stream_cmd(stop_cmd)
        print("[INFO] UHD stream dừng.")

    except Exception as e:
        print(f"[ERROR] UHD thread: {e}")
        stop_event.set()

# ─────────────────────────────────────────────
# MAIN THREAD — chỉ vẽ
# ─────────────────────────────────────────────
def main():
    spec_queue = queue.Queue(maxsize=Q_MAXSIZE)

    # Khởi động UHD thread
    t = threading.Thread(target=uhd_producer, args=(spec_queue,), daemon=True)
    t.start()

    # Chờ PATCH_SIZE được set bởi UHD thread
    time.sleep(1.5)

    # Setup plot
    freqs = (np.fft.fftshift(np.fft.fftfreq(PATCH_SIZE, 1.0 / SAMPLE_RATE))
             + CENTER_FREQ) / 1e6

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    (line,) = ax.plot(freqs, np.full(PATCH_SIZE, YMIN),
                      color="#00d4aa", linewidth=0.8)

    ax.set_xlim(freqs[0], freqs[-1])
    ax.set_ylim(YMIN, YMAX)
    ax.set_xlabel("Tần số (MHz)", color="#aab4c2")
    ax.set_ylabel("Biên độ (dBFS)", color="#aab4c2")
    ax.set_title(
        f"Real-time Spectrum  {CENTER_FREQ/1e6:.1f} MHz | BW {SAMPLE_RATE/1e6:.1f} MHz",
        color="#e6edf3",
    )
    ax.tick_params(colors="#aab4c2")
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")
    ax.grid(True, color="#21262d", linewidth=0.5, linestyle="--")
    fig.tight_layout()
    fig.canvas.draw()

    print("[INFO] Vẽ phổ... Ctrl+C để dừng.")

    # Vòng lặp vẽ — main thread
    while not stop_event.is_set():
        try:
            spec = spec_queue.get(timeout=0.5)
            line.set_ydata(spec)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
        except queue.Empty:
            fig.canvas.flush_events()   # giữ GUI responsive khi chờ

        if not plt.fignum_exists(fig.number):
            stop_event.set()
            break

    plt.close("all")
    t.join(timeout=3)
    print("[INFO] Thoát.")

if __name__ == "__main__":
    main()
