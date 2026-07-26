"""mmWave radar driver (TI IWR6843 / IWR1843 / xWRL6432, UART).

Production-grade streaming driver.  Configures the radar ONCE (first
``read()``), then streams frames continuously at the radar's native
frame rate (~10 Hz / 100ms).

Uses the lab's proven ``configure_radar()`` pipeline for reliable
initialisation instead of a bespoke config loop.

Usage:
    sensor = MmWaveRadarSensor(sensor_id="r", mode="serial",
                                serial_port="/dev/cu.usbserial-xxx",
                                cfg_path="config/point_cloud.cfg")
    sensor.start()          # open + configure (blocks ~4 s)
    for _ in range(100):
        obs = sensor.read()  # returns one frame (~100 ms)
    sensor.stop()
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.sensors.base import BaseSensor, SensorObservation

log = logging.getLogger("drivers.mmwave")
_LOG_INTERVAL = 100  # log every Nth frame in serial mode

# ---------------------------------------------------------------------------
# Mock point cloud — per-instance state, smooth trajectory
# ---------------------------------------------------------------------------


def _make_mock_point_cloud(state: dict[str, Any]) -> dict[str, Any]:
    state["frame"] += 1
    f = state["frame"]
    state["pos"] += 0.02 * state["dir"]
    if abs(state["pos"]) > 1.0:
        state["dir"] *= -1
    cx = state["pos"]
    cy = 1.5 + 0.3 * np.sin(f * 0.05)
    n_points = 12 + int(4 * np.sin(f * 0.1))
    objects = []
    for i in range(n_points):
        angle = np.random.uniform(-0.3, 0.3)
        dist = np.random.uniform(0.1, 0.4)
        doppler = 0.2 * np.sin(f * 0.1 + i * 0.5)
        objects.append({
            "x": float(cx + dist * np.sin(angle)),
            "y": float(cy + dist * np.cos(angle) - 0.2),
            "z": float(np.random.uniform(-0.3, 0.5)),
            "doppler": float(doppler),
            "snr": float(np.random.uniform(15, 35)),
        })
    return {
        "num_detected_obj": n_points,
        "objects": objects,
        "range_profile": None,
        "motion_score": float(abs(doppler) * 2),
    }


# ---------------------------------------------------------------------------
# MmWaveRadarSensor
# ---------------------------------------------------------------------------


class MmWaveRadarSensor(BaseSensor):
    """Production streaming mmWave radar driver.

    Parameters
    ----------
    sensor_id : str
        Unique identifier for this radar unit.
    serial_port : str
        UART device path (data output from the radar).
    baudrate_data : int
        Baud rate (115200 for IWRL6432).
    mode : str
        ``\"mock\"`` or ``\"serial\"``.
    cfg_path : str | None
        Path to TI ``.cfg`` file.
    """

    def __init__(
        self,
        sensor_id: str,
        serial_port: str = "/dev/ttyUSB0",
        baudrate_data: int = 115200,
        mode: str = "mock",
        cfg_path: str | None = None,
    ) -> None:
        super().__init__(sensor_id=sensor_id, sensor_type="mmwave")
        self.serial_port = serial_port
        self.baudrate_data = baudrate_data
        self.mode = mode
        self.cfg_path = cfg_path

        # Serial state
        self._port: Any = None
        self._lock = threading.Lock()
        self._running = False
        self._frame_count = 0
        self._bg_subtractor: Any = None
        self._range_bin_spacing = 0.05
        self._reconnect_delay = 0.5  # initial backoff for reconnection

        # Mock state
        self._mock_state: dict[str, float] = {"pos": 0.0, "dir": 1.0, "frame": 0.0}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the serial port and configure the radar.

        Blocks until streaming begins (~4 s).  Safe to call multiple
        times (no-op if already running).
        """
        if self._running:
            return
        if self.mode != "serial":
            return

        from src.sensors.mmWave.lab_mmwave import configure_radar, ExponentialBackgroundSubtractor
        import serial as pyserial

        with self._lock:
            t0 = time.monotonic()
            log.info(f"Opening {self.serial_port} @ {self.baudrate_data} baud")
            self._port = pyserial.Serial(
                self.serial_port, baudrate=self.baudrate_data, timeout=0.3
            )
            time.sleep(0.2)
            self._port.reset_input_buffer()

            if self.cfg_path:
                log.info(f"Configuring radar with {self.cfg_path}")
                try:
                    configure_radar(self._port, self.cfg_path)
                except Exception as exc:
                    log.error(f"Configuration failed: {exc}")
                    self._port.close()
                    self._port = None
                    raise
            else:
                # Minimal config
                from src.sensors.mmWave.lab_mmwave import write_cli_command
                write_cli_command(self._port, "sensorStop 0")
                time.sleep(0.1)
                self._port.reset_input_buffer()
                commands = [
                    "channelCfg 7 3 0",
                    "chirpComnCfg 8 0 0 256 4 24.3 3",
                    "chirpTimingCfg 28 37 0 160 58",
                    "frameCfg 64 0 4000 1 100 0",
                    "guiMonitor 2 0 0 0 0 0 0 0 0 0 0",
                    "sigProcChainCfg 16 2 1 0 0 0 0 0",
                ]
                for cmd in commands:
                    write_cli_command(self._port, cmd)
                    time.sleep(0.05)
                self._port.reset_input_buffer()
                write_cli_command(self._port, "sensorStart 0 0 0 0")

            self._bg_subtractor = ExponentialBackgroundSubtractor(alpha=0.02, init_frames=5)
            self._frame_count = 0
            self._reconnect_delay = 0.5
            self._running = True
            elapsed = time.monotonic() - t0
            log.info(f"Radar configured and streaming in {elapsed:.1f}s")

    def stop(self) -> None:
        """Stop the radar and close the port."""
        with self._lock:
            if self._port is not None:
                try:
                    from src.sensors.mmWave.lab_mmwave import write_cli_command
                    write_cli_command(self._port, "sensorStop 0")
                    time.sleep(0.05)
                except Exception:
                    pass
                try:
                    self._port.close()
                except Exception:
                    pass
                self._port = None
            self._running = False
            self._frame_count = 0

    def __enter__(self) -> MmWaveRadarSensor:
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()

    def __del__(self) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read(self) -> list[SensorObservation]:
        if self.mode == "serial":
            return self._read_serial()
        return self._read_mock()

    # ------------------------------------------------------------------
    # Mock mode
    # ------------------------------------------------------------------

    def _read_mock(self) -> list[SensorObservation]:
        obs = _make_mock_point_cloud(self._mock_state)
        n = obs["num_detected_obj"]
        return [
            SensorObservation(
                sensor_id=self.sensor_id,
                sensor_type=self.sensor_type,
                timestamp=datetime.now(timezone.utc),
                observation=obs,
                confidence=0.7 if n > 5 else 0.3,
                metadata={"mode": "mock", "num_objects": n},
            )
        ]

    # ------------------------------------------------------------------
    # Serial mode — configure once, then stream
    # ------------------------------------------------------------------

    def _read_serial(self) -> list[SensorObservation]:
        from src.sensors.mmWave.lab_mmwave import (
            point_cloud_from_tlvs,
            range_profile_from_tlvs,
            read_frame,
        )

        # Start the radar on first call, or reconnect if it died
        if not self._running:
            try:
                self.start()
            except Exception as exc:
                delay = self._reconnect_delay
                log.warning(f"Start failed ({exc}), retrying in {delay:.1f}s")
                self._reconnect_delay = min(self._reconnect_delay * 2, 5.0)
                time.sleep(delay)
                return [
                    SensorObservation(
                        sensor_id=self.sensor_id,
                        sensor_type=self.sensor_type,
                        timestamp=datetime.now(timezone.utc),
                        observation={"error": f"Reconnecting: {exc}"},
                        confidence=0.0,
                        metadata={"mode": "serial", "reconnect_delay": delay},
                    )
                ]

        # Read one frame
        try:
            with self._lock:
                if self._port is None or not self._port.is_open:
                    raise RuntimeError("Port not open")
                frame_number, tlvs = read_frame(self._port, timeout_s=0.5)
        except Exception as exc:
            log.warning(f"Frame read failed: {exc}")
            self._running = False  # trigger reconnect on next call
            self._reconnect_delay = min(self._reconnect_delay * 2, 5.0)
            self.stop()
            return [
                SensorObservation(
                    sensor_id=self.sensor_id,
                    sensor_type=self.sensor_type,
                    timestamp=datetime.now(timezone.utc),
                    observation={"error": f"Frame lost: {exc}"},
                    confidence=0.0,
                    metadata={"mode": "serial", "reconnect_delay": self._reconnect_delay},
                )
            ]

        # Successful read — reset backoff
        self._reconnect_delay = 0.5
        self._frame_count += 1

        # Decode
        cloud = point_cloud_from_tlvs(tlvs)
        n = len(cloud.x)
        objects_list = [
            {"x": float(cloud.x[i]), "y": float(cloud.y[i]),
             "z": float(cloud.z[i]), "doppler": float(cloud.velocity[i]), "snr": 0.0}
            for i in range(n)
        ]

        motion_score = 0.0
        range_data = None
        rp = range_profile_from_tlvs(tlvs)
        if rp is not None and self._bg_subtractor is not None:
            range_data = rp.tolist()
            mp = self._bg_subtractor.update(rp)
            motion_score = float(np.mean(np.abs(mp)))
        elif n > 0 and self._bg_subtractor is not None:
            ranges = np.sqrt(cloud.x**2 + cloud.y**2)
            if len(ranges):
                try:
                    mp = self._bg_subtractor.update(ranges)
                    motion_score = float(np.mean(np.abs(mp)))
                except ValueError:
                    pass

        log_this = (self._frame_count % _LOG_INTERVAL) == 0
        if log_this:
            log.info(f"Frame {self._frame_count}: {n} objects, motion={motion_score:.3f}")

        obs = {
            "num_detected_obj": n,
            "objects": objects_list,
            "range_profile": range_data,
            "motion_score": motion_score,
            "frame_number": frame_number,
        }

        return [
            SensorObservation(
                sensor_id=self.sensor_id,
                sensor_type=self.sensor_type,
                timestamp=datetime.now(timezone.utc),
                observation=obs,
                confidence=0.8 if n > 0 else 0.3,
                metadata={
                    "mode": "serial",
                    "serial_port": self.serial_port,
                    "num_objects": n,
                    "frame_number": frame_number,
                },
            )
        ]


__all__ = ["MmWaveRadarSensor"]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="mmWave radar self-test")
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--mode", choices=["mock", "serial"], default="mock")
    parser.add_argument("--cfg")
    args = parser.parse_args()

    sensor = MmWaveRadarSensor(
        sensor_id="mmwave-self-test",
        serial_port=args.port,
        mode=args.mode,
        cfg_path=args.cfg,
    )
    try:
        if args.mode == "serial":
            sensor.start()
        print(f"Reading from {sensor.sensor_id} ({args.mode})")
        for _ in range(50):
            for o in sensor.read():
                obs = o.observation
                err = obs.get("error", "")
                if err:
                    print(f"  ⚠ {err}")
                else:
                    n = obs.get("num_detected_obj", 0)
                    m = obs.get("motion_score", 0)
                    print(f"  frame={obs.get('frame_number', '?')}  objs={n}  motion={m:.3f}  conf={o.confidence:.2f}")
            if args.mode == "mock":
                time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        sensor.stop()
        print("Stopped.")
