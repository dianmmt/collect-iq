import uhd
import time
import numpy as np
from pathlib import Path
from datetime import datetime

from config import (
    CENTER_FREQ, GAIN, ANTENNA,
    X310_ADDR, X310_SUBDEV, X310_SAMPLE_RATE, X310_BANDWIDTH,
    IQ_OUTPUT_DIR,
)

def set_sdr(
    addr="192.168.40.2",
    subdev="A:0",
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
    Cấu hình USRP X310 với UBX-160.

    Parameters
    ----------
    addr : str
        Địa chỉ IP của X310.
    subdev : str
        Subdevice muốn dùng, ví dụ:
        - "A:0" : card slot A, kênh 0
        - "B:0" : card slot B, kênh 0
    rx_freq : float
        Tần số thu trung tâm (Hz).
    sample_rate : float
        Tốc độ lấy mẫu (S/s).
    gain : float
        Gain thu (dB).
    bandwidth : float or None
        Băng thông analog frontend (Hz).
    antenna : str
        Cổng antenna, thường là "RX2" hoặc "TX/RX".
    channel : int
        Logical channel index, thường để 0 khi chỉ dùng 1 kênh.
    """

    # Tạo device args, qua cổng Ethernet/SFP+
    usrp = uhd.usrp.MultiUSRP(f"addr={addr}")

    # Chọn daughterboard/subdevice
    usrp.set_rx_subdev_spec(uhd.usrp.SubdevSpec(subdev))

    # Clock / time source
    usrp.set_clock_source(clock_source)
    usrp.set_time_source(time_source)

    # Cấu hình RX
    usrp.set_rx_rate(sample_rate, channel)
    usrp.set_rx_freq(uhd.types.TuneRequest(rx_freq), channel)
    usrp.set_rx_gain(gain, channel)
    usrp.set_rx_antenna(antenna, channel)

    if bandwidth is not None:
        usrp.set_rx_bandwidth(bandwidth, channel)

    # Reset time về 0
    usrp.set_time_now(uhd.types.TimeSpec(0.0))

    # In cấu hình thực tế
    print("=== USRP X310 configured ===")
    print(f"IP address      : {addr}")
    print(f"Subdevice       : {subdev}")
    print(f"Channel         : {channel}")
    print(f"RX freq         : {usrp.get_rx_freq(channel)/1e6:.6f} MHz")
    print(f"Sample rate     : {usrp.get_rx_rate(channel)/1e6:.6f} MS/s")
    print(f"Gain            : {usrp.get_rx_gain(channel):.2f} dB")
    print(f"Antenna         : {usrp.get_rx_antenna(channel)}")
    print(f"Bandwidth       : {usrp.get_rx_bandwidth(channel)/1e6:.6f} MHz")
    print(f"Clock source    : {usrp.get_clock_source(0)}")
    print(f"Time source     : {usrp.get_time_source(0)}")

    return usrp

def get_streamer(usrp, channels=(0,), cpu_format="fc32", otw_format="sc16"):
    """
    Tạo RX streamer cho USRP.

    Returns
    -------
    rx_streamer : UHD RX streamer
    rx_metadata : UHD RXMetadata object
    info : dict
        Thông tin phụ trợ về streamer
    """
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
        raise RuntimeError(f"Không tạo được RX streamer: {e}")


def _metadata_error_to_text(metadata):
    """
    Chuẩn hóa thông tin lỗi metadata thành dict dễ log/debug.
    UHD trả lỗi stream chủ yếu qua RXMetadata.error_code và strerror().
    """
    try:
        err_name = str(metadata.error_code)
    except Exception:
        err_name = "UNKNOWN"

    try:
        err_msg = metadata.strerror()
    except Exception:
        err_msg = "Không đọc được metadata.strerror()"

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
    timeout=1.0,
    start_now=True,
    start_time=None,
    stop_when_done=True,
    keep_partial_on_error=True,
):
    """
    Thu num_samps mẫu IQ từ RX streamer và trả về cả danh sách lỗi/cảnh báo.

    Parameters
    ----------
    rx_streamer : UHD RX streamer
    rx_metadata : UHD RXMetadata
    num_samps : int
        Tổng số samples cần thu trên mỗi channel
    timeout : float
        Timeout cho mỗi lần recv()
    start_now : bool
        True nếu muốn stream ngay
    start_time : uhd.types.TimeSpec or None
        Nếu muốn timed start thì truyền start_time, khi đó start_now phải False
    stop_when_done : bool
        Gửi lệnh stop_cont sau khi thu xong
    keep_partial_on_error : bool
        Nếu có lỗi giữa chừng, vẫn giữ lại phần samples đã thu được

    Returns
    -------
    result : np.ndarray
        Shape = (num_channels, num_samps_thu_duoc), dtype complex64
    errors : list[dict]
        Danh sách lỗi/cảnh báo trong quá trình stream
    stats : dict
        Thống kê stream
    """
    max_samps = rx_streamer.get_max_num_samps()
    print(f'[WARN] Max_samps is: {max_samps}')
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

                # timeout thường không cần quăng exception ngay
                if "timeout" in md["error_code"].lower():
                    continue

                # overflow, late command, broken chain... có thể quyết định dừng
                if not keep_partial_on_error:
                    raise RuntimeError(f"Lỗi stream: {md['error_code']} - {md['message']}")

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

usrp = set_sdr(
    addr=X310_ADDR,
    subdev=X310_SUBDEV,
    rx_freq=CENTER_FREQ,
    sample_rate=X310_SAMPLE_RATE,
    gain=GAIN,
    bandwidth=X310_BANDWIDTH,
    antenna=ANTENNA,
)

rx_streamer, rx_md, info = get_streamer(usrp, channels=(0,))
print(info)

samples, errors, stats = recv_samples(
    rx_streamer=rx_streamer,
    rx_metadata=rx_md,
    num_samps=int(X310_SAMPLE_RATE * 0.1),
    timeout=1.0,
)

print("stats =", stats)
print("errors =", errors[:10])
print("samples shape =", samples.shape)


# ── THÊM PHẦN NÀY ──────────────────────────────
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = Path(IQ_OUTPUT_DIR) / f"iq_x310_{ts}"

interleaved = samples[0].view(np.float32)   # fc32 → float32 interleaved I/Q
interleaved.tofile(str(out_path) + ".bin")

print(f"Đã lưu: {out_path}.bin  ({interleaved.nbytes / 1e6:.1f} MB)")
