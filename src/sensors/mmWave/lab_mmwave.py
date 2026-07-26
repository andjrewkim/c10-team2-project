"""mmWave radar (TI IWR6843 / IWR1843 / xWRL6432) utilities extracted and
consolidated from ``mmwave/labs/lab11-mmwave-lab/mmwave_lab/``.

Provides point-cloud TLV decoding, range-profile parsing, exponential
background subtraction, motion detection, CLI configuration helpers,
and velocity tracking — the same algorithms used in the COSMOS lab tools.

All cfg files live in ``config/`` (point_cloud.cfg, near_field_hand_50cm.cfg,
hand_distance.cfg).
"""

from __future__ import annotations

import math
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPEED_OF_LIGHT = 299_792_458.0

# The four uint16 words {0x0102, 0x0304, 0x0506, 0x0708}
# appear in little-endian byte order on UART.
MAGIC_WORD = b"\x02\x01\x04\x03\x06\x05\x08\x07"

POINT_CLOUD_FLOAT = 1
POINT_CLOUD_FIXED_TYPES = {301, 1020}
RANGE_PROFILE_MAJOR = 302
RANGE_PROFILE_MINOR = 303

CLI_OK_PATTERNS = ("done", "mmwdemo:", "skipped")
CLI_FAILURE_PATTERNS = ("error", "not recognized", "invalid", "failed")
OPTIONAL_UNSUPPORTED_COMMANDS = {"cfarScndPassCfg", "compressionCfg"}

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PointCloud:
    x: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    y: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    z: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    velocity: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))


@dataclass(frozen=True)
class RangeConfig:
    bin_spacing_m: float
    fft_size: int
    num_range_bins: int


def empty_point_cloud() -> PointCloud:
    return PointCloud()


# ---------------------------------------------------------------------------
# Config file loading (xWRL6432 / IWR cfg format)
# ---------------------------------------------------------------------------


def load_configuration(path: str | Path) -> list[str]:
    """Load TI mmWave demo configuration commands from a ``.cfg`` file.

    Strips blank lines, ``%`` / ``#`` comments, and validates that a
    ``sensorStart`` command is present.
    """
    commands: list[str] = []
    for raw_line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("%", "#")):
            continue
        commands.append(line)
    if not any(c.startswith("sensorStart") for c in commands):
        raise ValueError("Configuration file has no sensorStart command.")
    return commands


def parse_range_config(commands: list[str]) -> Optional[RangeConfig]:
    """Estimate range-bin spacing and FFT size from CLI configuration.

    Returns *None* if the required commands are not found.

    Fs = 100 MHz / DigOutputSampRate
    delta_r = c * Fs / (2 * slope * N_FFT)
    """
    sample_rate_divider: Optional[float] = None
    adc_samples: Optional[int] = None
    slope_mhz_per_us: Optional[float] = None

    for cmd in commands:
        fields = cmd.split()
        if fields[0] == "chirpComnCfg" and len(fields) >= 5:
            sample_rate_divider = float(fields[1])
            adc_samples = int(fields[4])
        elif fields[0] == "chirpTimingCfg" and len(fields) >= 5:
            slope_mhz_per_us = float(fields[4])

    if sample_rate_divider is None or adc_samples is None or slope_mhz_per_us is None or slope_mhz_per_us == 0:
        return None
    if sample_rate_divider <= 0 or adc_samples <= 0:
        return None

    sampling_rate_hz = 100e6 / sample_rate_divider
    fft_size = 1 << (adc_samples - 1).bit_length()
    slope_hz_per_second = abs(slope_mhz_per_us) * 1e12

    bin_spacing = (
        SPEED_OF_LIGHT * sampling_rate_hz / (2.0 * slope_hz_per_second * fft_size)
    )
    return RangeConfig(bin_spacing_m=bin_spacing, fft_size=fft_size, num_range_bins=fft_size // 2)


# ---------------------------------------------------------------------------
# Point cloud decoding
# ---------------------------------------------------------------------------


def decode_float_points(payload: bytes) -> PointCloud:
    """Decode a floating-point point-cloud TLV payload."""
    count = len(payload) // 16
    if count == 0:
        return PointCloud()
    values = np.frombuffer(payload[: count * 16], dtype="<f4").reshape(count, 4)
    return PointCloud(
        x=values[:, 0].astype(float),
        y=values[:, 1].astype(float),
        z=values[:, 2].astype(float),
        velocity=values[:, 3].astype(float),
    )


def decode_fixed_points(payload: bytes) -> PointCloud:
    """Decode a fixed-point (10-byte per point) point-cloud TLV payload."""
    if len(payload) < 20:
        return PointCloud()
    xyz_unit, doppler_unit, *_ = struct.unpack_from("<ffff", payload, 0)
    num_major_points, _ = struct.unpack_from("<HH", payload, 16)

    xs, ys, zs, vs = [], [], [], []
    offset = 20
    for _ in range(num_major_points):
        if offset + 10 > len(payload):
            break
        x, y, z, doppler, _, _ = struct.unpack_from("<hhhhBB", payload, offset)
        xs.append(x * xyz_unit)
        ys.append(y * xyz_unit)
        zs.append(z * xyz_unit)
        vs.append(doppler * doppler_unit)
        offset += 10

    return PointCloud(
        x=np.array(xs, dtype=float),
        y=np.array(ys, dtype=float),
        z=np.array(zs, dtype=float),
        velocity=np.array(vs, dtype=float),
    )


def point_cloud_from_tlvs(tlvs: list[tuple[int, bytes]]) -> PointCloud:
    """Extract the first point-cloud TLV from a list of TI TLV packets."""
    for tlv_type, payload in tlvs:
        if tlv_type == POINT_CLOUD_FLOAT:
            return decode_float_points(payload)
        if tlv_type in POINT_CLOUD_FIXED_TYPES:
            return decode_fixed_points(payload)
    return PointCloud()


def range_profile_from_tlvs(tlvs: list[tuple[int, bytes]]) -> Optional[np.ndarray]:
    """Extract the first range-profile TLV as a float array."""
    for tlv_type, payload in tlvs:
        if tlv_type in {RANGE_PROFILE_MAJOR, RANGE_PROFILE_MINOR}:
            return np.frombuffer(payload, dtype="<u4").astype(float)
    return None


# ---------------------------------------------------------------------------
# Frame reading (from serial)
# ---------------------------------------------------------------------------


def read_exact(port: Any, count: int, timeout_s: float) -> bytes:
    """Read exactly *count* bytes from a serial port."""
    output = bytearray()
    deadline = time.monotonic() + timeout_s
    while len(output) < count:
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Timed out reading {count} bytes ({len(output)} received)."
            )
        chunk = port.read(count - len(output))
        if chunk:
            output.extend(chunk)
    return bytes(output)


def read_frame(port: Any, timeout_s: float, expected_range_profile_bytes: Optional[int] = None) -> tuple[int, list[tuple[int, bytes]]]:
    """Read one mmWave radar frame from the serial port.

    Returns ``(frame_number, list_of_tlv_packets)``.
    """
    _wait_for_magic_word(port, timeout_s)
    header = MAGIC_WORD + read_exact(port, 32, timeout_s)
    (
        _version,
        total_packet_length,
        _platform,
        frame_number,
        _cpu_cycles,
        _num_objects,
        num_tlvs,
        _subframe,
    ) = struct.unpack_from("<8I", header, 8)

    if not 40 <= total_packet_length <= 2_000_000:
        raise ValueError(f"Implausible packet length: {total_packet_length}")
    if num_tlvs > 64:
        raise ValueError(f"Implausible TLV count: {num_tlvs}")

    packet = header + read_exact(port, total_packet_length - 40, timeout_s)
    tlvs = _parse_tlvs(packet, num_tlvs, expected_range_profile_bytes)
    return frame_number, tlvs


def read_range_profile_frame(
    port: Any,
    timeout_s: float,
    expected_range_profile_bytes: int,
) -> tuple[int, np.ndarray]:
    """Convenience: read frames until one contains a range profile."""
    while True:
        frame_number, tlvs = read_frame(port, timeout_s, expected_range_profile_bytes)
        profile = range_profile_from_tlvs(tlvs)
        if profile is not None:
            return frame_number, profile


def read_point_cloud_frame(
    port: Any,
    timeout_s: float,
) -> tuple[int, PointCloud]:
    """Convenience: read a frame and extract the point cloud."""
    frame_number, tlvs = read_frame(port, timeout_s)
    return frame_number, point_cloud_from_tlvs(tlvs)


def _wait_for_magic_word(port: Any, timeout_s: float) -> None:
    """Discard ASCII data until the binary magic word appears."""
    window = bytearray()
    discarded = bytearray()
    deadline = time.monotonic() + timeout_s
    while True:
        if time.monotonic() > deadline:
            text = bytes(discarded[-512:]).decode("ascii", errors="ignore").strip()
            msg = "Timed out waiting for frame magic word."
            if text:
                msg += f" Last UART text: {text}"
            raise TimeoutError(msg)
        b = port.read(1)
        if not b:
            continue
        window.extend(b)
        discarded.extend(b)
        if len(window) > len(MAGIC_WORD):
            del window[0]
        if bytes(window) == MAGIC_WORD:
            return


def _parse_tlvs(packet: bytes, num_tlvs: int, expected_range_profile_bytes: Optional[int] = None) -> list[tuple[int, bytes]]:
    """Parse TLV entries from a frame packet.

    Tries both 40-byte and 52-byte header offsets.
    """
    candidates: list[tuple[int, list[tuple[int, bytes]]]] = []
    for header_size in (40, 52):
        pos = header_size
        tlvs: list[tuple[int, bytes]] = []
        ok = True
        for _ in range(num_tlvs):
            if pos + 8 > len(packet):
                ok = False
                break
            tlv_type, tlv_length = struct.unpack_from("<II", packet, pos)
            pos += 8
            if tlv_type == 0 or tlv_length > len(packet) - pos:
                ok = False
                break
            tlvs.append((tlv_type, packet[pos : pos + tlv_length]))
            pos += tlv_length
        if ok:
            score = 0
            if header_size == 40:
                score += 1
            if pos == len(packet):
                score += 4
            if expected_range_profile_bytes is not None:
                for t, p in tlvs:
                    if t in {RANGE_PROFILE_MAJOR, RANGE_PROFILE_MINOR} and len(p) == expected_range_profile_bytes:
                        score += 16
                        break
            candidates.append((score, tlvs))
    if candidates:
        return max(candidates, key=lambda x: x[0])[1]
    raise ValueError("Could not identify TLV start position.")


# ---------------------------------------------------------------------------
# CLI helpers (sending commands, reading responses)
# ---------------------------------------------------------------------------


def write_cli_command(port: Any, command: str) -> None:
    """Write a CLI command (with CRLF) to the serial port."""
    port.write((command + "\r\n").encode("ascii"))
    port.flush()


def read_text_until_quiet(port: Any, quiet_time: float = 0.15, max_time: float = 2.0) -> str:
    """Read UART text until there has been no input for *quiet_time* seconds."""
    start = time.monotonic()
    last_rx = start
    chunks: list[bytes] = []
    while time.monotonic() - start < max_time:
        waiting = port.in_waiting
        if waiting:
            chunks.append(port.read(waiting))
            last_rx = time.monotonic()
            continue
        if time.monotonic() - last_rx >= quiet_time:
            break
        time.sleep(0.01)
    return b"".join(chunks).decode("ascii", errors="ignore")


def cli_response_failed(reply: str) -> bool:
    reply_lower = reply.lower()
    return any(pattern in reply_lower for pattern in CLI_FAILURE_PATTERNS)


def cli_response_ok(reply: str) -> bool:
    reply_lower = reply.lower()
    return any(pattern in reply_lower for pattern in CLI_OK_PATTERNS)


def require_plausible_cli_response(command: str, reply: str) -> None:
    """Raise if *reply* is unrecognizable (wrong baud / not in CLI mode)."""
    if cli_response_ok(reply) or cli_response_failed(reply):
        return
    if reply.strip():
        raise RuntimeError(
            f"Unexpected CLI response while sending {command!r}. "
            "This usually means the UART baud rate is wrong or the sensor "
            "was already streaming binary frames. Reset or power-cycle the "
            "EVM to return the demo CLI to 115200."
        )
    raise RuntimeError(
        f"No CLI response while sending {command!r}. "
        "Check the serial port and baud rate."
    )


def send_configuration(
    port: Any,
    commands: list[str],
    use_cfg_baud_rate: bool = False,
) -> None:
    """Send a list of cfg commands to the radar, handling baud rate changes.

    Skips ``sensorStart`` (sent last) and optionally skips ``baudRate``.
    """
    start_command: Optional[str] = None
    for command in commands:
        if command.startswith("sensorStart"):
            start_command = command
            continue

        command_name = command.split()[0]

        if command_name == "baudRate" and not use_cfg_baud_rate:
            continue

        write_cli_command(port, command)

        if command_name == "baudRate":
            time.sleep(0.15)  # let the command leave the UART buffer
            fields = command.split()
            if len(fields) == 2:
                port.baudrate = int(fields[1])
            reply = read_text_until_quiet(port, max_time=1.0)
            if cli_response_failed(reply):
                raise RuntimeError(f"Device returned error for: {command}")
            continue

        reply = read_text_until_quiet(port)
        if reply and (cli_response_ok(reply) or cli_response_failed(reply)):
            pass  # printed by caller
        require_plausible_cli_response(command, reply)

        if "not recognized" in reply.lower() and command_name in OPTIONAL_UNSUPPORTED_COMMANDS:
            continue
        if cli_response_failed(reply):
            raise RuntimeError(f"Device returned error for: {command}")

    if start_command is None:
        raise ValueError("No sensorStart command found.")
    port.reset_input_buffer()
    write_cli_command(port, start_command)


# ---------------------------------------------------------------------------
# Radar lifecycle helpers
# ---------------------------------------------------------------------------


def stop_and_drain(port: Any, timeout_s: float = 4.0) -> None:
    """Recover the CLI even if binary frames are still streaming.

    Repeatedly sends ``sensorStop 0`` until the CLI prompt appears.
    """
    port.reset_input_buffer()
    deadline = time.monotonic() + timeout_s
    last_rx = time.monotonic()
    next_stop = 0.0
    text_tail = ""
    while time.monotonic() < deadline:
        now = time.monotonic()
        if now >= next_stop:
            write_cli_command(port, "sensorStop 0")
            next_stop = now + 0.25
        waiting = port.in_waiting
        if waiting:
            data = port.read(waiting)
            text_tail = (text_tail + data.decode("ascii", errors="ignore"))[-512:]
            last_rx = time.monotonic()
        elif "done" in text_tail.lower() and time.monotonic() - last_rx > 0.25:
            break
        elif "mmwdemo:/>" in text_tail.lower() and time.monotonic() - last_rx > 0.5:
            break
        else:
            time.sleep(0.02)
    port.reset_input_buffer()


def warm_reset_demo(port: Any, timeout_s: float = 8.0) -> None:
    """Reload the flashed demo so the UART is back at a clean CLI prompt.

    Sends ``sensorWarmRst`` and waits for the ``mmwDemo:/>`` prompt.
    """
    port.reset_input_buffer()
    write_cli_command(port, "sensorWarmRst")
    chunks: list[bytes] = []
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        waiting = port.in_waiting
        if waiting:
            chunks.append(port.read(waiting))
            text = b"".join(chunks[-8:]).decode("ascii", errors="ignore").lower()
            if "mmwdemo:/>" in text:
                port.reset_input_buffer()
                return
        else:
            time.sleep(0.05)
    text = b"".join(chunks).decode("ascii", errors="ignore")
    if "not supported" in text.lower() or "error" in text.lower():
        raise RuntimeError(f"sensorWarmRst failed: {text.strip()}")
    raise RuntimeError("Timed out waiting for mmwDemo prompt after sensorWarmRst.")


def remove_leading_sensor_stop(commands: list[str]) -> list[str]:
    """Strip a leading ``sensorStop`` command (useful after fresh start)."""
    if commands and commands[0].split()[0] == "sensorStop":
        return commands[1:]
    return commands


def configure_radar(port: Any, cfg_path: str | Path, use_cfg_baud: bool = False) -> None:
    """Full radar configuration pipeline: load cfg, stop, warm reset, send."""
    commands = load_configuration(cfg_path)
    stop_and_drain(port)
    warm_reset_demo(port)
    send_configuration(port, remove_leading_sensor_stop(commands), use_cfg_baud)


# ---------------------------------------------------------------------------
# Velocity tracking (point-cloud frame-to-frame)
# ---------------------------------------------------------------------------


def xy_points(cloud: PointCloud) -> np.ndarray:
    """Return (N, 2) array of x/y coordinates."""
    if len(cloud.x) == 0:
        return np.empty((0, 2), dtype=float)
    return np.column_stack((cloud.x, cloud.y))


def radial_ranges(cloud: PointCloud) -> np.ndarray:
    """Return per-point radial range (sqrt(x^2 + y^2))."""
    return np.sqrt(cloud.x**2 + cloud.y**2)


def estimate_tracked_velocity(
    cloud: PointCloud,
    previous_cloud: Optional[PointCloud],
    previous_time: Optional[float],
    now: float,
    max_match_distance_m: float = 0.25,
) -> np.ndarray:
    """Estimate radial velocity by nearest-neighbour frame-to-frame tracking.

    Returns an array of velocities (NaN for unmatched points).
    """
    tracked = np.full(len(cloud.x), np.nan, dtype=float)
    if previous_cloud is None or previous_time is None or len(cloud.x) == 0 or len(previous_cloud.x) == 0:
        return tracked
    dt = now - previous_time
    if dt <= 0:
        return tracked
    current_xy = xy_points(cloud)
    previous_xy = xy_points(previous_cloud)
    deltas = current_xy[:, np.newaxis, :] - previous_xy[np.newaxis, :, :]
    distances = np.sqrt(np.sum(deltas * deltas, axis=2))
    nearest_indices = np.argmin(distances, axis=1)
    nearest_distances = distances[np.arange(len(cloud.x)), nearest_indices]
    matched = nearest_distances <= max_match_distance_m
    current_ranges = radial_ranges(cloud)
    previous_ranges = radial_ranges(previous_cloud)
    tracked[matched] = (
        current_ranges[matched] - previous_ranges[nearest_indices[matched]]
    ) / dt
    return tracked


# ---------------------------------------------------------------------------
# Point-cloud filtering
# ---------------------------------------------------------------------------


def filter_front_roi(
    cloud: PointCloud,
    x_limit_m: float = 0.15,
    y_max_m: float = 0.50,
) -> PointCloud:
    """Keep only points within a front-facing region of interest."""
    if len(cloud.x) == 0:
        return empty_point_cloud()
    mask = (
        (np.abs(cloud.x) <= x_limit_m)
        & (cloud.y >= 0.0)
        & (cloud.y <= y_max_m)
    )
    return PointCloud(
        x=cloud.x[mask],
        y=cloud.y[mask],
        z=cloud.z[mask],
        velocity=cloud.velocity[mask],
    )


def filter_point_slice(
    cloud: PointCloud,
    point_distance_m: float,
    point_window_m: float,
    x_limit_m: Optional[float] = None,
    _z_limit_m: Optional[float] = None,
) -> np.ndarray:
    """Return an (N, 3) array of points near a specific down-range distance.

    Used for box-content analysis where a narrow y-slice is selected.
    """
    if len(cloud.x) == 0:
        return np.empty((0, 3), dtype=float)
    xyz = np.column_stack((cloud.x, cloud.y, cloud.z))
    mask = np.abs(xyz[:, 1] - point_distance_m) <= point_window_m
    if x_limit_m is not None and x_limit_m > 0:
        mask &= np.abs(xyz[:, 0]) <= x_limit_m
    return xyz[mask]


def pack_slice_points(frames: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Pack a list of per-frame (N,3) point arrays into a NaN-padded matrix."""
    frame_count = len(frames)
    counts = np.array([len(frame) for frame in frames], dtype=np.uint16)
    max_points = int(np.max(counts)) if len(counts) else 0
    points = np.full((frame_count, max_points, 3), np.nan, dtype=float)
    for index, frame_points in enumerate(frames):
        count = len(frame_points)
        if count:
            points[index, :count, :] = frame_points
    return counts, points


# ---------------------------------------------------------------------------
# Motion detection (background subtraction)
# ---------------------------------------------------------------------------


class ExponentialBackgroundSubtractor:
    """Online exponential-moving-average background subtractor.

    Tracks a running background estimate and returns the residual
    (current profile minus background) after each update.
    """

    def __init__(self, alpha: float = 0.01, init_frames: int = 20) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.init_frames = init_frames
        self._background: Optional[np.ndarray] = None
        self._seed_count = 0
        self._seed_sum: Optional[np.ndarray] = None

    def update(self, profile: np.ndarray) -> np.ndarray:
        profile = np.asarray(profile, dtype=float)
        if self._background is None or self._seed_count < self.init_frames:
            if self._seed_sum is None:
                self._seed_sum = profile.copy()
            else:
                self._seed_sum += profile
            self._seed_count += 1
            if self._seed_count >= self.init_frames:
                self._background = self._seed_sum / self._seed_count
                self._seed_sum = None
            return np.zeros_like(profile)
        residual = profile - self._background
        self._background = (1.0 - self.alpha) * self._background + self.alpha * profile
        return residual

    @property
    def background(self) -> Optional[np.ndarray]:
        return self._background

    def reset(self) -> None:
        self._background = None
        self._seed_count = 0
        self._seed_sum = None


def track_motion_target(
    motion: np.ndarray,
    bin_spacing_m: float,
    min_range_m: float = 0.15,
    max_range_m: float = 2.0,
    peak_ratio: float = 3.0,
) -> tuple[Optional[float], float]:
    """Find the range of the strongest moving object in a motion profile.

    Parameters
    ----------
    motion : ndarray
        Per-bin motion residual (absolute or positive).
    bin_spacing_m : float
        Range per FFT bin.
    min_range_m, max_range_m : float
        Range window to search within.

    Returns
    -------
    (distance_m, peak_strength) or (None, peak_strength) if no clear peak.
    """
    start = max(1, int(math.ceil(min_range_m / bin_spacing_m)))
    stop = min(len(motion), int(math.floor(max_range_m / bin_spacing_m)) + 1)
    window = motion[start:stop]
    if len(window) == 0:
        return None, 0.0

    peak_idx = start + int(np.argmax(window))
    peak_strength = float(motion[peak_idx])
    typical = float(np.median(window)) + 1.0
    if peak_strength < peak_ratio * typical:
        return None, peak_strength

    # Sub-bin refinement via weighted centroid.
    left = max(start, peak_idx - 1)
    right = min(stop, peak_idx + 2)
    indices = np.arange(left, right, dtype=float)
    weights = motion[left:right].astype(float)
    ws = float(np.sum(weights))
    refined = float(np.sum(indices * weights) / ws) if ws > 0 else float(peak_idx)
    return refined * bin_spacing_m, peak_strength


def motion_residual(profile: np.ndarray, background: np.ndarray, residual_mode: str = "absolute") -> np.ndarray:
    """Compute the motion residual between a profile and its background."""
    residual = profile - background
    if residual_mode == "positive":
        return np.maximum(residual, 0.0)
    return np.abs(residual)


def update_ema_background(background: np.ndarray, profile: np.ndarray, alpha: float) -> np.ndarray:
    """Update a background estimate with an exponential moving average."""
    return (1.0 - alpha) * background + alpha * profile


# ---------------------------------------------------------------------------
# Helpers shared by box / posture feature extraction
# ---------------------------------------------------------------------------


def load_trial_data(npz_path: Path) -> dict[str, np.ndarray]:
    """Load a trial_data.npz recording into a plain dict."""
    with np.load(npz_path) as npz:
        return {key: npz[key] for key in npz.files}


def db_scale(values: np.ndarray) -> np.ndarray:
    """Log (dB-like) scaling for range-profile values."""
    return 10.0 * np.log10(np.maximum(values, 0.0) + 1.0)


def fill_nan_series(values: np.ndarray, fallback: float = 0.0) -> np.ndarray:
    """Linearly interpolate NaN gaps in a 1-D array."""
    output = np.asarray(values, dtype=float).copy()
    if output.size == 0:
        return output
    finite = np.isfinite(output)
    if not finite.any():
        output[:] = fallback
        return output
    if finite.all():
        return output
    indices = np.arange(output.size)
    output[~finite] = np.interp(indices[~finite], indices[finite], output[finite])
    return output


def resample_vector(values: np.ndarray, target_count: int) -> np.ndarray:
    """Resample a 1-D vector to *target_count* evenly spaced points."""
    values = fill_nan_series(np.asarray(values, dtype=float))
    if target_count <= 0:
        raise ValueError("target_count must be positive")
    if values.size == 0:
        return np.zeros(target_count, dtype=float)
    if values.size == 1:
        return np.full(target_count, float(values[0]), dtype=float)
    source_x = np.linspace(0.0, 1.0, values.size)
    target_x = np.linspace(0.0, 1.0, target_count)
    return np.interp(target_x, source_x, values)


def resample_matrix(matrix: np.ndarray, target_rows: int, target_cols: int) -> np.ndarray:
    """Resample a 2-D matrix to *target_rows* x *target_cols*."""
    matrix = np.asarray(matrix, dtype=float)
    if target_rows <= 0 or target_cols <= 0:
        raise ValueError("target_rows and target_cols must be positive")
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        return np.zeros((target_rows, target_cols), dtype=float)
    time_resampled = np.column_stack(
        [resample_vector(matrix[:, col], target_rows) for col in range(matrix.shape[1])]
    )
    if matrix.shape[1] == 1:
        return np.repeat(time_resampled, target_cols, axis=1)
    source_x = np.linspace(0.0, 1.0, matrix.shape[1])
    target_x = np.linspace(0.0, 1.0, target_cols)
    return np.vstack([np.interp(target_x, source_x, row) for row in time_resampled])


def robust_normalize(values: np.ndarray) -> np.ndarray:
    """Robust z-score-like normalization using 95th percentile."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=float)
    scale = float(np.percentile(np.abs(finite), 95))
    if not math.isfinite(scale) or scale <= 1e-9:
        scale = float(np.std(finite))
    if not math.isfinite(scale) or scale <= 1e-9:
        scale = 1.0
    return np.clip(values / scale, -6.0, 6.0)


def time_window_segments(
    time_s: np.ndarray,
    window_seconds: float,
    overlap: float,
    min_frames: int,
) -> list[tuple[int, int]]:
    """Split a time series into overlapping windows.

    Returns ``[(start_idx, end_idx), ...]`` slices of *time_s*.
    """
    time_s = np.asarray(time_s, dtype=float)
    if time_s.size < min_frames:
        return []
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")
    step_seconds = window_seconds * (1.0 - overlap)
    first_time = float(time_s[0])
    last_time = float(time_s[-1])
    segments: list[tuple[int, int]] = []
    start_time = first_time
    while start_time <= last_time:
        end_time = start_time + window_seconds
        start_index = int(np.searchsorted(time_s, start_time, side="left"))
        end_index = int(np.searchsorted(time_s, end_time, side="right"))
        if end_index - start_index >= min_frames:
            segments.append((start_index, end_index))
        if end_time >= last_time:
            break
        start_time += step_seconds
    return segments


def read_box_data_frame(
    port: Any,
    frame_timeout_s: float,
    expected_range_profile_bytes: int,
) -> tuple[int, np.ndarray, PointCloud]:
    """Read a frame that contains both a range profile and point cloud."""
    while True:
        frame_number, tlvs = read_frame(port, frame_timeout_s, expected_range_profile_bytes)
        profile = range_profile_from_tlvs(tlvs)
        if profile is not None:
            return frame_number, profile, point_cloud_from_tlvs(tlvs)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

__all__ = [
    "SPEED_OF_LIGHT",
    "MAGIC_WORD",
    "PointCloud",
    "RangeConfig",
    "empty_point_cloud",
    "load_configuration",
    "parse_range_config",
    "decode_float_points",
    "decode_fixed_points",
    "point_cloud_from_tlvs",
    "range_profile_from_tlvs",
    "read_exact",
    "read_frame",
    "read_range_profile_frame",
    "read_point_cloud_frame",
    "write_cli_command",
    "read_text_until_quiet",
    "cli_response_failed",
    "cli_response_ok",
    "require_plausible_cli_response",
    "send_configuration",
    "stop_and_drain",
    "warm_reset_demo",
    "remove_leading_sensor_stop",
    "configure_radar",
    "xy_points",
    "radial_ranges",
    "estimate_tracked_velocity",
    "filter_front_roi",
    "filter_point_slice",
    "pack_slice_points",
    "ExponentialBackgroundSubtractor",
    "track_motion_target",
    "motion_residual",
    "update_ema_background",
    "load_trial_data",
    "db_scale",
    "fill_nan_series",
    "resample_vector",
    "resample_matrix",
    "robust_normalize",
    "time_window_segments",
    "read_box_data_frame",
]
