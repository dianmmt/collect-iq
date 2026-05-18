import uhd
import time
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime


def set_sdr_b205(
    rx_freq=2.437e9,
    sample_rate=25e6,
    gain=20,
    bandwidth=None,
    antenna="RX2",
    channel=0,
    clock_source="internal",
    time_source="internal",
):
    """
    Cau hinh USRP B205 mini (ket noi qua USB).

    B205 mini khac X310:
    - Khong can addr= (USB, tu detect)
    - Khong co subdev slot A/B (chi 1 card, subdev "A:A")
    - Sample rate toi da ~56 MS/s (master clock 56 MHz)
    """
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
    print(f"RX freq         : {usrp.get_rx_freq(channel)/1e6:.6f} MHz")
    print(f"Sample rate     : {usrp.get_rx_rate(channel)/1e6:.6f} MS/s")
    print(f"Gain            : {usrp.get_rx_gain(channel):.2f} dB")
    print(f"Antenna         : {usrp.get_rx_antenna(channel)}")
    print(f"Bandwidth       : {usrp.get_rx_bandwidth(channel)/1e6:.6f} MHz")
    print(f"Clock source    : {usrp.get_clock_source(0)}")
    print(f"Time source     : {usrp.get_time_source(0)}")

    return usrp


def get_streamer(usrp, channels=(0,), cpu_format="fc32", otw_format="sc16"):
    """Tao RX streamer."""
    try:
        st_args = uhd.usrp.StreamArgs(cpu_format, otw_format)
        st_args.channels = list(channels)

        rx_streamer = usrp.get_rx_stream(st_args)
        rx_metadata = uhd.types.RXMetadata()

        info = {
            "channels": list(channels),
            "cpu_format": cpu_format,
            "otw_format": otw_format,
            "num_channels": rx_streamer.get_num_channels(),
            "max_num_samps": rx_streamer.get_max_num_samps(),
        }
        return rx_streamer, rx_metadata, info

    except Exception as e:
        raise RuntimeError(f"Khong tao duoc RX streamer: {e}")


def _metadata_error_to_text(metadata):
    """Chuan hoa loi metadata thanh dict."""
    try:
        err_name = str(metadata.error_code)
    except Exception:
        err_name = "UNKNOWN"

    try:
        err_msg = metadata.strerror()
    except Exception:
        err_msg = "N/A"

    out = {
        "error_code": err_name,
        "message": err_msg,
        "has_time_spec": False,
        "time_spec": None,
        "out_of_sequence": None,
        "more_fragments": None,
        "fragment_offset": None,
        "start_of_burst": None,
        "end_of_burst": None,
    }

    for attr in [
        "has_time_spec",
        "out_of_sequence",
        "more_fragments",
        "fragment_offset",
        "start_of_burst",
        "end_of_burst",
    ]:
        if hasattr(metadata, attr):
            try:
                out[attr] = getattr(metadata, attr)
            except Exception:
                pass

    if hasattr(metadata, "has_time_spec") and metadata.has_time_spec:
        try:
            out["has_time_spec"] = True
            out["time_spec"] = metadata.time_spec.get_real_secs()
        except Exception:
            pass

    return out


def recv_samples(
    rx_streamer,
    rx_metadata,
    num_samps,
    timeout=3.0,
    start_now=True,
    start_time=None,
    stop_when_done=True,
    keep_partial_on_error=True,
):
    """
    Thu num_samps mau IQ. Tra ve (result, errors, stats).

    Parameters
    ----------
    rx_streamer : UHD RX streamer
    rx_metadata : UHD RXMetadata
    num_samps : int
        Tong so samples can thu
    timeout : float
        Timeout cho moi lan recv() — B205 USB can dai hon X310
    stop_when_done : bool
        Gui lenh stop_cont sau khi thu xong
    keep_partial_on_error : bool
        Neu co loi giu lai phan samples da thu duoc

    Returns
    -------
    result : np.ndarray, shape (num_channels, num_samps_thu_duoc), dtype complex64
    errors : list[dict]
    stats  : dict
    """
    max_samps = rx_streamer.get_max_num_samps()
    num_channels = rx_streamer.get_num_channels()

    recv_buffer = np.zeros((num_channels, max_samps), dtype=np.complex64)
    result = np.zeros((num_channels, num_samps), dtype=np.complex64)

    errors = []
    recv_count = 0
    recv_calls = 0
    t0 = time.time()

    try:
        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
        stream_cmd.stream_now = bool(start_now and start_time is None)

        if start_time is not None:
            stream_cmd.stream_now = False
            stream_cmd.time_spec = start_time

        rx_streamer.issue_stream_cmd(stream_cmd)

        while recv_count < num_samps:
            recv_calls += 1
            samps = rx_streamer.recv(recv_buffer, rx_metadata, timeout)

            md = _metadata_error_to_text(rx_metadata)

            if "none" not in md["error_code"].lower():
                md["recv_call"] = recv_calls
                md["recv_count_before_error"] = recv_count
                errors.append(md)

                if "timeout" in md["error_code"].lower():
                    continue

                if not keep_partial_on_error:
                    raise RuntimeError(f"Loi stream: {md['error_code']} - {md['message']}")

            if samps > 0:
                n_copy = min(samps, num_samps - recv_count)
                result[:, recv_count:recv_count + n_copy] = recv_buffer[:, :n_copy]
                recv_count += n_copy

    except Exception as e:
        errors.append({
            "error_code": "EXCEPTION",
            "message": str(e),
            "recv_call": recv_calls,
            "recv_count_before_error": recv_count,
        })

    finally:
        if stop_when_done:
            try:
                stop_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
                rx_streamer.issue_stream_cmd(stop_cmd)
            except Exception as e:
                errors.append({
                    "error_code": "STOP_STREAM_EXCEPTION",
                    "message": str(e),
                    "recv_call": recv_calls,
                    "recv_count_before_error": recv_count,
                })

    elapsed = time.time() - t0
    result = result[:, :recv_count]

    stats = {
        "requested_samps": int(num_samps),
        "received_samps": int(recv_count),
        "num_channels": int(num_channels),
        "recv_calls": int(recv_calls),
        "elapsed_sec": float(elapsed),
        "throughput_msps_per_chan": float(recv_count / elapsed / 1e6) if elapsed > 0 else 0.0,
        "num_errors": len(errors),
    }

    return result, errors, stats


def save_iq(samples, output_path, fmt="fc32", metadata=None):
    """
    Luu tin hieu IQ ra file.

    Parameters
    ----------
    samples : np.ndarray
        Shape (num_channels, num_samps), dtype complex64
    output_path : str or Path
        Duong dan file dau ra (khong can duoi, ham tu them)
    fmt : str
        "fc32" -> interleaved float32 I/Q (.cf32)
        "sc16" -> interleaved int16  I/Q (.sc16)
        "npy"  -> complex64 numpy array  (.npy)
        "bin"  -> interleaved float32 I/Q (.bin), giong fc32 khac duoi
    metadata : dict or None
        Neu truyen vao se luu them file _meta.txt

    Returns
    -------
    saved_files : list[Path]
    """
    output_path = Path(output_path)
    saved_files = []

    for ch_idx in range(samples.shape[0]):
        ch_data = samples[ch_idx]

        suffix = "" if samples.shape[0] == 1 else f"_ch{ch_idx}"

        if fmt == "fc32":
            out_file = output_path.with_name(output_path.stem + suffix + ".cf32")
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

        elif fmt == "bin":
            out_file = output_path.with_name(output_path.stem + suffix + ".bin")
            interleaved = np.empty(len(ch_data) * 2, dtype=np.float32)
            interleaved[0::2] = ch_data.real
            interleaved[1::2] = ch_data.imag
            interleaved.tofile(out_file)

        else:
            raise ValueError(f"fmt khong hop le: '{fmt}'. Chon 'fc32', 'sc16', 'npy', hoac 'bin'.")

        print(f"[SAVE] Channel {ch_idx} -> {out_file}  ({out_file.stat().st_size / 1e6:.2f} MB)")
        saved_files.append(out_file)

    if metadata is not None:
        meta_file = output_path.with_name(output_path.stem + "_meta.txt")
        with open(meta_file, "w", encoding="utf-8") as f:
            f.write("# IQ Capture Metadata\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n")
            for k, v in metadata.items():
                f.write(f"{k} = {v}\n")
        print(f"[SAVE] Metadata  -> {meta_file}")
        saved_files.append(meta_file)

    return saved_files


# ──────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Thu va luu IQ bang USRP B205 mini")
    parser.add_argument("--freq",     type=float, default=2450e6, help="Tan so trung tam (Hz)")
    parser.add_argument("--rate",     type=float, default=20e6,   help="Sample rate (S/s), toi da ~56e6")
    parser.add_argument("--gain",     type=float, default=20,     help="RX gain (dB)")
    parser.add_argument("--bw",       type=float, default=None,   help="Bandwidth analog (Hz)")
    parser.add_argument("--duration", type=float, default=0.1,    help="Thoi gian thu (giay)")
    parser.add_argument("--fmt",      type=str,   default="fc32",
                        choices=["fc32", "sc16", "npy", "bin"],   help="Dinh dang file luu")
    parser.add_argument("--output",   type=str,   default=None,   help="Duong dan file output (khong can duoi)")
    args = parser.parse_args()

    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path(f"iq_{ts}")
    else:
        out_path = Path(args.output)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Cau hinh B205 mini
    usrp = set_sdr_b205(
        rx_freq=args.freq,
        sample_rate=args.rate,
        gain=args.gain,
        bandwidth=args.bw,
        antenna="RX2",
    )

    # 2. Tao streamer
    rx_streamer, rx_md, info = get_streamer(usrp, channels=(0,))
    print("Streamer info:", info)

    # 3. Thu samples
    num_samps = int(args.rate * args.duration)
    print(f"\nThu {num_samps} samples ({args.duration * 1000:.1f} ms) @ {args.rate / 1e6:.1f} MS/s ...")

    samples, errors, stats = recv_samples(
        rx_streamer=rx_streamer,
        rx_metadata=rx_md,
        num_samps=num_samps,
        timeout=3.0,
    )

    print("Stats:", stats)
    if errors:
        print(f"[WARN] {len(errors)} loi/canh bao trong qua trinh stream:")
        for e in errors[:5]:
            print("  ", e)

    # 4. Luu file
    metadata = {
        "device":           "USRP B205 mini",
        "rx_freq_hz":       args.freq,
        "sample_rate_hz":   args.rate,
        "gain_db":          args.gain,
        "bandwidth_hz":     args.bw,
        "duration_sec":     args.duration,
        "num_samples":      stats["received_samps"],
        "format":           args.fmt,
        "antenna":          "RX2",
        "capture_time":     datetime.now().isoformat(),
    }

    saved = save_iq(samples, out_path, fmt=args.fmt, metadata=metadata)
    print(f"\nDa luu {len(saved)} file(s).")
