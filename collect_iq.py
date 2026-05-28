import uhd
import time
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime

from config import (
    CENTER_FREQ, SAMPLE_RATE, GAIN, BANDWIDTH, ANTENNA,
    DURATION_SEC, OUTPUT_FMT, IQ_OUTPUT_DIR,
    NF_FFT_SIZE, NF_PERCENTILE, NF_METHOD,
)


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


def compute_noise_floor(
    samples,
    sample_rate,
    fft_size=1024,
    percentile=10.0,
    method="welch",
    channel=0,
):
    """
    Tinh noise floor tu tin hieu IQ.

    Cac phuong phap uoc luong noise floor:
    - "welch"      : PSD trung binh (Welch) -> lay percentile thap nhat (default)
    - "median"     : Median cua PSD (chac chan hon khi co nhieu xung)
    - "time"       : Tinh thang trong mien thoi gian (20*log10 of RMS)
    - "all"        : Tra ve ca 3 phuong phap

    Parameters
    ----------
    samples : np.ndarray
        Shape (num_channels, num_samps), dtype complex64
    sample_rate : float
        Sample rate (Hz), can de tinh noise floor theo bandwidth
    fft_size : int
        So diem FFT. Nen chon la luy thua cua 2 (512, 1024, 2048).
        Lon hon -> phan giai tan so tot hon, nhung tinh cham hon.
    percentile : float
        Phan vi (%) dung cho phuong phap "welch".
        percentile=10 nghia la lay muc nang luong phan vi thu 10 cua PSD
        (phan lon la noise, khong phai tin).
    method : str
        "welch", "median", "time", hoac "all"
    channel : int
        Chi so kenh can tinh (default 0)

    Returns
    -------
    result : dict
        Cac key chinh:
        - "noise_floor_dbfs"    : Noise floor (dBFS, tuong doi full-scale)
        - "noise_floor_dbm"     : Noise floor (dBm, chi mang tinh tham khao)
        - "noise_floor_w_hz"    : Noise spectral density (W/Hz)
        - "rms_dbfs"            : RMS toan bo tin hieu (dBFS)
        - "peak_dbfs"           : Peak (dBFS)
        - "crest_factor_db"     : Crest factor = Peak - RMS (dB)
        - "dynamic_range_db"    : Dynamic range uoc tinh (Peak - Noise floor)
        - "snr_estimate_db"     : SNR uoc tinh (RMS - Noise floor)
        - "method"              : Phuong phap da dung
        - "fft_size"            : FFT size da dung
        - "num_segments"        : So segment FFT
        - "freq_resolution_hz"  : Do phan giai tan so (Hz/bin)

    Notes
    -----
    dBFS (decibels relative to full scale):
        0 dBFS la muc lon nhat co the bieu dien (|z| = 1.0 voi fc32).
        Noise floor thuong nam trong khoang -80 den -100 dBFS
        tuy thuoc vao gain va hardware.

    dBm (reference):
        Chuyen tu dBFS sang dBm can biet impedance va calibration.
        Cong thuc nay dung: P_dBm = P_dBFS + P_ref_dBm
        Voi B205: full-scale ~ +10 dBm (gan dung), nen:
            noise_floor_dbm ~ noise_floor_dbfs + 10
        Chi mang tinh tham khao — de chinh xac can do RF calibration.
    """
    if samples.ndim == 1:
        x = samples.astype(np.complex64)
    else:
        x = samples[channel].astype(np.complex64)

    n = len(x)
    if n == 0:
        raise ValueError("Mang samples rong.")

    fft_size = min(fft_size, n)
    num_segments = n // fft_size

    # ── 1. Cac chi so mien thoi gian ────────────────────────────────────────
    power_linear = np.mean(np.abs(x) ** 2)                     # cong suat trung binh
    rms_linear   = np.sqrt(power_linear)
    peak_linear  = np.max(np.abs(x))

    rms_dbfs    = 20.0 * np.log10(rms_linear + 1e-300)
    peak_dbfs   = 20.0 * np.log10(peak_linear + 1e-300)
    crest_db    = peak_dbfs - rms_dbfs                         # Crest factor

    # ── 2. Uoc luong noise floor ─────────────────────────────────────────────
    window = np.hanning(fft_size)
    win_power_correction = np.sum(window ** 2) / fft_size      # Hieu chinh nang luong cua so

    noise_floor_dbfs_welch  = None
    noise_floor_dbfs_median = None
    noise_floor_dbfs_time   = None

    if num_segments >= 1:
        # Tinh PSD bang phuong phap Welch (trung binh nhieu segment FFT)
        psd_segments = []
        for i in range(num_segments):
            seg = x[i * fft_size:(i + 1) * fft_size] * window
            spectrum = np.fft.fft(seg, n=fft_size)
            # Cong suat moi bin (da hieu chinh cho so)
            psd = (np.abs(spectrum) ** 2) / (fft_size * win_power_correction)
            psd_segments.append(psd)

        psd_avg    = np.mean(psd_segments, axis=0)             # Welch PSD
        psd_median = np.median(psd_segments, axis=0)           # Median PSD

        # Noise floor = percentile thap (phan lon la noise, khong phai tin hieu)
        noise_psd_welch  = np.percentile(psd_avg,    percentile)
        noise_psd_median = np.percentile(psd_median, percentile)

        noise_floor_dbfs_welch  = 10.0 * np.log10(noise_psd_welch  + 1e-300)
        noise_floor_dbfs_median = 10.0 * np.log10(noise_psd_median + 1e-300)

        # Noise spectral density (W/Hz) — can sample_rate
        freq_resolution = sample_rate / fft_size
        noise_w_hz_welch = noise_psd_welch / freq_resolution

    else:
        # Qua it samples de phan segment -> dung RMS thay the
        freq_resolution = sample_rate / fft_size
        noise_w_hz_welch = power_linear / sample_rate

    # Phuong phap mien thoi gian: RMS cua 10% samples co bien do nho nhat
    # (loai bo burst/tin hieu, chi giu phan noise)
    sorted_power = np.sort(np.abs(x) ** 2)
    cutoff_idx   = max(1, int(len(sorted_power) * 0.10))
    noise_power_time = np.mean(sorted_power[:cutoff_idx])
    noise_floor_dbfs_time = 10.0 * np.log10(noise_power_time + 1e-300)

    # ── 3. Chon noise floor chinh ────────────────────────────────────────────
    # Mac dinh dung Welch; fallback sang time domain neu it segment
    if noise_floor_dbfs_welch is not None:
        if method in ("welch", "all"):
            primary_nf = noise_floor_dbfs_welch
        elif method == "median":
            primary_nf = noise_floor_dbfs_median
        else:
            primary_nf = noise_floor_dbfs_time
    else:
        primary_nf = noise_floor_dbfs_time

    # ── 4. Chuyen sang dBm (tham khao, B205 full-scale ~ +10 dBm) ───────────
    B205_FULLSCALE_DBM = 10.0                                  # gan dung, can calibration
    noise_floor_dbm = primary_nf + B205_FULLSCALE_DBM

    # ── 5. Cac chi so dan xuat ───────────────────────────────────────────────
    dynamic_range_db = peak_dbfs - primary_nf
    snr_estimate_db  = rms_dbfs  - primary_nf

    result = {
        # Noise floor chinh
        "noise_floor_dbfs"         : float(primary_nf),
        "noise_floor_dbm"          : float(noise_floor_dbm),
        "noise_floor_w_hz"         : float(noise_w_hz_welch),
        # Phan tich tin hieu
        "rms_dbfs"                 : float(rms_dbfs),
        "peak_dbfs"                : float(peak_dbfs),
        "crest_factor_db"          : float(crest_db),
        "dynamic_range_db"         : float(dynamic_range_db),
        "snr_estimate_db"          : float(snr_estimate_db),
        # Thong so tinh toan
        "method"                   : method,
        "fft_size"                 : int(fft_size),
        "num_segments"             : int(num_segments),
        "freq_resolution_hz"       : float(freq_resolution),
        "percentile_used"          : float(percentile),
    }

    # Them ket qua tung phuong phap neu method="all"
    if method == "all":
        result["noise_floor_dbfs_welch"]  = float(noise_floor_dbfs_welch)  if noise_floor_dbfs_welch  is not None else None
        result["noise_floor_dbfs_median"] = float(noise_floor_dbfs_median) if noise_floor_dbfs_median is not None else None
        result["noise_floor_dbfs_time"]   = float(noise_floor_dbfs_time)

    return result


def print_noise_floor(nf_result):
    """In ket qua noise floor ra man hinh theo dinh dang dep."""
    print("\n=== Noise Floor Analysis ===")
    print(f"Method              : {nf_result['method']}  |  FFT size: {nf_result['fft_size']}  |  Segments: {nf_result['num_segments']}")
    print(f"Freq resolution     : {nf_result['freq_resolution_hz']/1e3:.3f} kHz/bin")
    print("─" * 44)
    print(f"Noise floor         : {nf_result['noise_floor_dbfs']:>8.2f} dBFS")
    print(f"Noise floor (ref)   : {nf_result['noise_floor_dbm']:>8.2f} dBm  (B205 full-scale ~ +10 dBm)")
    print(f"Noise spectral dens : {nf_result['noise_floor_w_hz']:.3e} W/Hz")
    print("─" * 44)
    print(f"RMS power           : {nf_result['rms_dbfs']:>8.2f} dBFS")
    print(f"Peak power          : {nf_result['peak_dbfs']:>8.2f} dBFS")
    print(f"Crest factor        : {nf_result['crest_factor_db']:>8.2f} dB")
    print(f"Dynamic range       : {nf_result['dynamic_range_db']:>8.2f} dB  (peak - noise floor)")
    print(f"SNR estimate        : {nf_result['snr_estimate_db']:>8.2f} dB  (RMS  - noise floor)")

    if nf_result["method"] == "all":
        print("─" * 44)
        print("Per-method comparison:")
        if nf_result.get("noise_floor_dbfs_welch") is not None:
            print(f"  Welch  (p={nf_result['percentile_used']:4.1f}%) : {nf_result['noise_floor_dbfs_welch']:>8.2f} dBFS")
        if nf_result.get("noise_floor_dbfs_median") is not None:
            print(f"  Median (p={nf_result['percentile_used']:4.1f}%) : {nf_result['noise_floor_dbfs_median']:>8.2f} dBFS")
        if nf_result.get("noise_floor_dbfs_time") is not None:
            print(f"  Time domain (10%) : {nf_result['noise_floor_dbfs_time']:>8.2f} dBFS")

    print("=" * 44)


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
    parser.add_argument("--freq",       type=float, default=CENTER_FREQ,  help="Tan so trung tam (Hz)")
    parser.add_argument("--rate",       type=float, default=SAMPLE_RATE,  help="Sample rate (S/s), toi da ~56e6")
    parser.add_argument("--gain",       type=float, default=GAIN,         help="RX gain (dB)")
    parser.add_argument("--bw",         type=float, default=BANDWIDTH,    help="Bandwidth analog (Hz)")
    parser.add_argument("--duration",   type=float, default=DURATION_SEC, help="Thoi gian thu (giay)")
    parser.add_argument("--fmt",        type=str,   default=OUTPUT_FMT,
                        choices=["fc32", "sc16", "npy", "bin"],            help="Dinh dang file luu")
    parser.add_argument("--output",     type=str,   default=None,          help="Duong dan file output (khong can duoi)")
    # Tham so noise floor
    parser.add_argument("--nf-fft",     type=int,   default=NF_FFT_SIZE,  help="FFT size cho noise floor")
    parser.add_argument("--nf-pct",     type=float, default=NF_PERCENTILE, help="Percentile cho Welch method")
    parser.add_argument("--nf-method",  type=str,   default=NF_METHOD,
                        choices=["welch", "median", "time", "all"],        help="Phuong phap tinh noise floor")
    args = parser.parse_args()

    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path(IQ_OUTPUT_DIR) / f"iq_{ts}"
    else:
        out_path = Path(args.output)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Cau hinh B205 mini
    usrp = set_sdr_b205(
        rx_freq=args.freq,
        sample_rate=args.rate,
        gain=args.gain,
        bandwidth=args.bw,
        antenna=ANTENNA,
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

    # 4. Tinh noise floor
    nf_result = compute_noise_floor(
        samples=samples,
        sample_rate=args.rate,
        fft_size=args.nf_fft,
        percentile=args.nf_pct,
        method=args.nf_method,
        channel=0,
    )
    print_noise_floor(nf_result)

    # 5. Luu file
    metadata = {
        "device":               "USRP B205 mini",
        "rx_freq_hz":           args.freq,
        "sample_rate_hz":       args.rate,
        "gain_db":              args.gain,
        "bandwidth_hz":         args.bw,
        "duration_sec":         args.duration,
        "num_samples":          stats["received_samps"],
        "format":               args.fmt,
        "antenna":              ANTENNA,
        "capture_time":         datetime.now().isoformat(),
        # Ket qua noise floor
        "noise_floor_dbfs":     nf_result["noise_floor_dbfs"],
        "noise_floor_dbm":      nf_result["noise_floor_dbm"],
        "noise_floor_w_hz":     nf_result["noise_floor_w_hz"],
        "rms_dbfs":             nf_result["rms_dbfs"],
        "peak_dbfs":            nf_result["peak_dbfs"],
        "snr_estimate_db":      nf_result["snr_estimate_db"],
        "dynamic_range_db":     nf_result["dynamic_range_db"],
        "nf_method":            args.nf_method,
        "nf_fft_size":          args.nf_fft,
    }

    saved = save_iq(samples, out_path, fmt=args.fmt, metadata=metadata)
    print(f"\nDa luu {len(saved)} file(s).")
