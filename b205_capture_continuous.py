#!/usr/bin/env python3
"""
b205_continuous_capture.py
Thu IQ liên tục từng chunk 0.1s cho đến khi Ctrl+C.
"""

import signal
import time
import argparse
import numpy as np
import uhd
from pathlib import Path
from datetime import datetime

from config import (
    CENTER_FREQ, SAMPLE_RATE, GAIN, BANDWIDTH,
    DURATION_SEC, OUTPUT_FMT, IQ_OUTPUT_DIR,
)

# ─────────────────────────────────────────────
# STOP FLAG
# ─────────────────────────────────────────────
stop_flag = False

def handle_sigint(sig, frame):
    global stop_flag
    print("\n[INFO] Ctrl+C — dừng sau chunk hiện tại...")
    stop_flag = True

signal.signal(signal.SIGINT, handle_sigint)

# ─────────────────────────────────────────────
# CẤU HÌNH B205
# ─────────────────────────────────────────────
def set_sdr_b205(
    rx_freq=2450e6,
    sample_rate=10e6,
    gain=0,
    bandwidth=None,
    antenna="RX2",
    channel=0,
    clock_source="internal",
    time_source="internal",
):
    usrp = uhd.usrp.MultiUSRP("type=b200")
    usrp.set_rx_subdev_spec(uhd.usrp.SubdevSpec("A:A"))
    usrp.set_clock_source(clock_source)
    usrp.set_time_source(time_source)
    usrp.set_rx_rate(sample_rate, channel)
    usrp.set_rx_freq(uhd.types.TuneRequest(rx_freq), channel)
    usrp.set_rx_gain(gain, channel)
    usrp.set_rx_antenna(antenna, channel)
    if bandwidth is not None:
        usrp.set_rx_bandwidth(bandwidth, channel)
    usrp.set_time_now(uhd.types.TimeSpec(0.0))

    print("=== USRP B205 mini configured ===")
    print(f"RX freq     : {usrp.get_rx_freq(channel)/1e6:.6f} MHz")
    print(f"Sample rate : {usrp.get_rx_rate(channel)/1e6:.6f} MS/s")
    print(f"Gain        : {usrp.get_rx_gain(channel):.2f} dB")
    print(f"Bandwidth   : {usrp.get_rx_bandwidth(channel)/1e6:.6f} MHz")
    return usrp


# ─────────────────────────────────────────────
# TẠO STREAMER
# ─────────────────────────────────────────────
def get_streamer(usrp, channels=(0,), cpu_format="fc32", otw_format="sc16"):
    st_args = uhd.usrp.StreamArgs(cpu_format, otw_format)
    st_args.channels = list(channels)
    rx_streamer = usrp.get_rx_stream(st_args)
    rx_metadata = uhd.types.RXMetadata()
    info = {
        "num_channels":  rx_streamer.get_num_channels(),
        "max_num_samps": rx_streamer.get_max_num_samps(),
    }
    return rx_streamer, rx_metadata, info


# ─────────────────────────────────────────────
# THU MỘT CHUNK  ← FIX: không issue start_cont lại
# ─────────────────────────────────────────────
def recv_chunk(rx_streamer, rx_metadata, num_samps, timeout=3.0):
    """
    Thu đúng num_samps từ stream đang chạy sẵn.
    KHÔNG gọi issue_stream_cmd — stream đã được start từ bên ngoài.
    """
    max_samps    = rx_streamer.get_max_num_samps()
    num_channels = rx_streamer.get_num_channels()

    recv_buffer = np.zeros((num_channels, max_samps),  dtype=np.complex64)
    result      = np.zeros((num_channels, num_samps),  dtype=np.complex64)
    recv_count  = 0
    errors      = []

    while recv_count < num_samps:
        n = rx_streamer.recv(recv_buffer, rx_metadata, timeout)

        err = str(rx_metadata.error_code)
        if "none" not in err.lower():
            if "overflow" not in err.lower():          # overflow nhỏ → bỏ qua
                errors.append(err)
            if "timeout" in err.lower():
                continue

        if n > 0:
            n_copy = min(n, num_samps - recv_count)
            result[:, recv_count:recv_count + n_copy] = recv_buffer[:, :n_copy]
            recv_count += n_copy

    return result[:, :recv_count], errors


# ─────────────────────────────────────────────
# LƯU FILE IQ
# ─────────────────────────────────────────────
def save_iq(samples, output_path, fmt="bin", metadata=None):
    output_path = Path(output_path)
    saved_files = []

    for ch_idx in range(samples.shape[0]):
        ch_data = samples[ch_idx]
        suffix  = "" if samples.shape[0] == 1 else f"_ch{ch_idx}"

        if fmt in ("fc32", "bin"):
            ext = ".cf32" if fmt == "fc32" else ".bin"
            out_file = output_path.with_name(output_path.stem + suffix + ext)
            interleaved = np.empty(len(ch_data) * 2, dtype=np.float32)
            interleaved[0::2] = ch_data.real
            interleaved[1::2] = ch_data.imag
            interleaved.tofile(out_file)

        elif fmt == "sc16":
            out_file = output_path.with_name(output_path.stem + suffix + ".sc16")
            scale = 32767.0 / (np.max(np.abs(ch_data)) + 1e-12)
            interleaved = np.empty(len(ch_data) * 2, dtype=np.int16)
            interleaved[0::2] = (ch_data.real * scale).astype(np.int16)
            interleaved[1::2] = (ch_data.imag * scale).astype(np.int16)
            interleaved.tofile(out_file)

        elif fmt == "npy":
            out_file = output_path.with_name(output_path.stem + suffix + ".npy")
            np.save(out_file, ch_data)

        else:
            raise ValueError(f"fmt không hợp lệ: {fmt}")

        print(f"  [SAVE] {out_file}  ({out_file.stat().st_size/1e6:.2f} MB)")
        saved_files.append(out_file)

    if metadata:
        meta_file = output_path.with_name(output_path.stem + "_meta.txt")
        with open(meta_file, "w", encoding="utf-8") as f:
            f.write(f"# Generated: {datetime.now().isoformat()}\n")
            for k, v in metadata.items():
                f.write(f"{k} = {v}\n")
        saved_files.append(meta_file)

    return saved_files


# ─────────────────────────────────────────────
# CONTINUOUS CAPTURE
# ─────────────────────────────────────────────
def continuous_capture(freq, rate, gain, bandwidth, fmt, output_dir, duration_chunk=0.1):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cấu hình
    usrp = set_sdr_b205(rx_freq=freq, sample_rate=rate, gain=gain, bandwidth=bandwidth)

    # 2. Streamer
    rx_streamer, rx_md, info = get_streamer(usrp)
    print(f"[INFO] max_num_samps={info['max_num_samps']}")

    # 3. START stream một lần duy nhất ← FIX
    cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    cmd.stream_now = True
    rx_streamer.issue_stream_cmd(cmd)

    # Flush vài packet đầu
    flush_buf = np.zeros((1, info["max_num_samps"]), dtype=np.complex64)
    for _ in range(10):
        rx_streamer.recv(flush_buf, rx_md, timeout=2.0)

    num_samps = int(rate * duration_chunk)
    print(f"[INFO] Thu {num_samps} samps/chunk ({duration_chunk*1000:.0f}ms) — Ctrl+C để dừng\n")

    chunk_idx   = 0
    total_samps = 0
    t_start     = time.time()

    try:
        while not stop_flag:
            ts       = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            out_path = out_dir / f"iq_{ts}_chunk{chunk_idx:04d}"

            # 4. Thu chunk
            samples, errors = recv_chunk(rx_streamer, rx_md, num_samps)

            if errors:
                print(f"  [WARN] chunk {chunk_idx:04d}: {errors}")

            # 5. Lưu
            metadata = {
                "rx_freq_hz":     freq,
                "sample_rate_hz": rate,
                "gain_db":        gain,
                "chunk_index":    chunk_idx,
                "num_samples":    samples.shape[1],
                "format":         fmt,
                "capture_time":   datetime.now().isoformat(),
            }
            save_iq(samples, out_path, fmt=fmt, metadata=metadata)

            total_samps += samples.shape[1]
            chunk_idx   += 1
            elapsed      = time.time() - t_start
            print(f"  chunk {chunk_idx:04d} | {samples.shape[1]} samps | elapsed {elapsed:.1f}s")

    finally:
        # 6. STOP stream sạch dù thoát bằng Ctrl+C hay lỗi
        try:
            stop_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
            rx_streamer.issue_stream_cmd(stop_cmd)
        except Exception:
            pass

    print(f"\n[INFO] Xong: {chunk_idx} chunks | {total_samps/1e6:.2f} M samps | {time.time()-t_start:.1f}s")


# ─────────────────────────────────────────────
# MAIN  ← chỉ một khối duy nhất
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freq",   type=float, default=CENTER_FREQ)
    parser.add_argument("--rate",   type=float, default=SAMPLE_RATE)
    parser.add_argument("--gain",   type=float, default=GAIN)
    parser.add_argument("--bw",     type=float, default=BANDWIDTH)
    parser.add_argument("--fmt",    type=str,   default=OUTPUT_FMT,
                        choices=["fc32", "sc16", "npy", "bin"])
    parser.add_argument("--outdir", type=str,   default=IQ_OUTPUT_DIR)
    args = parser.parse_args()

    continuous_capture(
        freq=args.freq,
        rate=args.rate,
        gain=args.gain,
        bandwidth=args.bw,
        fmt=args.fmt,
        output_dir=args.outdir,
        duration_chunk=DURATION_SEC,
    )
