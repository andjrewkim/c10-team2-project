"""IMU sensor driver (accelerometer + gyroscope, I2C).

Supports two modes:

- **mock** (default): synthetic accelerometer / gyroscope values.
- **serial**: real BMI270 serial stream parsed with the lab's IMU parser
  and optionally processed through the lab's quaternion-based dead-reckoned
  trajectory integrator.

The serial parsing + trajectory math lives in
``sensors/lab_integration.imu``.
"""

from __future__ import annotations

import argparse
import random
import time
from datetime import datetime, timezone
from typing import Any

from src.sensors.base import BaseSensor, SensorObservation


class ImuSensor(BaseSensor):
    """Driver for an I2C or serial IMU (e.g. BNO055, MPU6050, ICM-20948, BMI270).

    Parameters
    ----------
    sensor_id : str
        Unique identifier for this IMU instance.
    mode : str
        ``"mock"`` or ``"serial"``.
    serial_port : str | None
        UART port for serial mode (e.g. ``/dev/cu.usbserial-xxx``).
    i2c_bus : int
        I2C bus number (used in mock mode metadata).
    i2c_address : int
        7-bit I2C device address (used in mock mode metadata).
    baudrate : int
        Serial baud rate for serial mode.
    """

    def __init__(
        self,
        sensor_id: str,
        mode: str = "mock",
        serial_port: str | None = None,
        i2c_bus: int = 1,
        i2c_address: int = 0x28,
        baudrate: int = 115200,
    ) -> None:
        super().__init__(sensor_id=sensor_id, sensor_type="imu")
        self.mode = mode
        self.serial_port = serial_port
        self.i2c_bus = i2c_bus
        self.i2c_address = i2c_address
        self.baudrate = baudrate
        self._gyro_bias: tuple[float, float, float] | None = None
        self._accel_bias: tuple[float, float, float] | None = None
        self._serial_conn: Any = None  # persistent serial connection

    def read(self) -> list[SensorObservation]:
        if self.mode == "serial":
            return self._read_serial()
        return self._read_mock()

    # ------------------------------------------------------------------
    # Bias calibration (zeroing)
    # ------------------------------------------------------------------

    def zero(self, num_samples: int = 50) -> dict[str, tuple[float, float, float]]:
        """Calibrate sensor biases by sampling while stationary.

        Collects *num_samples* readings (must be stationary), computes the
        mean gyroscope and accelerometer values, and stores them as bias
        offsets.  Subsequent ``read()`` calls subtract these offsets.

        Returns
        -------
        dict with keys ``"gyro_bias"`` and ``"accel_bias"`` (each ``(x, y, z)``).
        """
        samples: list[dict[str, float]] = []
        print(f"  Keeping IMU stationary, sampling {num_samples} readings...")
        for i in range(num_samples):
            obs_list = self.read_raw()
            if obs_list:
                obs = obs_list[0].observation or {}
                samples.append(obs)
            time.sleep(0.01)

        if not samples:
            print("  Warning: no samples collected, bias unchanged")
            return {"gyro_bias": self._gyro_bias, "accel_bias": self._accel_bias}

        gyro_x = [s.get("gyro_x", 0) for s in samples]
        gyro_y = [s.get("gyro_y", 0) for s in samples]
        gyro_z = [s.get("gyro_z", 0) for s in samples]
        accel_x = [s.get("accel_x", 0) for s in samples]
        accel_y = [s.get("accel_y", 0) for s in samples]
        accel_z = [s.get("accel_z", 0) for s in samples]

        import statistics
        self._gyro_bias = (
            statistics.mean(gyro_x),
            statistics.mean(gyro_y),
            statistics.mean(gyro_z),
        )
        self._accel_bias = (
            statistics.mean(accel_x),
            statistics.mean(accel_y),
            statistics.mean(accel_z),
        )
        print(f"  Gyro bias set to:  x={self._gyro_bias[0]:.4f}  y={self._gyro_bias[1]:.4f}  z={self._gyro_bias[2]:.4f} dps")
        print(f"  Accel bias set to: x={self._accel_bias[0]:.4f}  y={self._accel_bias[1]:.4f}  z={self._accel_bias[2]:.4f} g")
        return {"gyro_bias": self._gyro_bias, "accel_bias": self._accel_bias}

    def reset_bias(self) -> None:
        """Clear stored bias offsets."""
        self._gyro_bias = None
        self._accel_bias = None
        print("  Bias offsets cleared")

    def read_raw(self) -> list[SensorObservation]:
        """Read without applying bias correction (used internally by ``zero()``)."""
        if self.mode == "serial":
            return self._read_serial_raw()
        return self._read_mock_raw()

    # ------------------------------------------------------------------
    # Mock mode
    # ------------------------------------------------------------------

    def _read_mock(self) -> list[SensorObservation]:
        raw = self._read_mock_raw()
        return [self._apply_bias(obs) for obs in raw]

    def _apply_bias(self, obs: SensorObservation) -> SensorObservation:
        if self._gyro_bias is None and self._accel_bias is None:
            return obs
        o = dict(obs.observation or {})
        if self._gyro_bias is not None:
            o["gyro_x"] = o.get("gyro_x", 0) - self._gyro_bias[0]
            o["gyro_y"] = o.get("gyro_y", 0) - self._gyro_bias[1]
            o["gyro_z"] = o.get("gyro_z", 0) - self._gyro_bias[2]
        if self._accel_bias is not None:
            o["accel_x"] = o.get("accel_x", 0) - self._accel_bias[0]
            o["accel_y"] = o.get("accel_y", 0) - self._accel_bias[1]
            o["accel_z"] = o.get("accel_z", 0) - self._accel_bias[2]
        return SensorObservation(
            sensor_id=obs.sensor_id, sensor_type=obs.sensor_type,
            timestamp=obs.timestamp, observation=o,
            confidence=obs.confidence, metadata=obs.metadata,
            position=obs.position, tag_id=obs.tag_id,
        )

    def _read_mock_raw(self) -> list[SensorObservation]:
        obs = {
            "accel_x": random.uniform(-2.0, 2.0),
            "accel_y": random.uniform(-2.0, 2.0),
            "accel_z": random.uniform(-2.0, 2.0),
            "gyro_x": random.uniform(-0.5, 0.5),
            "gyro_y": random.uniform(-0.5, 0.5),
            "gyro_z": random.uniform(-0.5, 0.5),
        }
        return [
            SensorObservation(
                sensor_id=self.sensor_id,
                sensor_type=self.sensor_type,
                timestamp=datetime.now(timezone.utc),
                observation=obs,
                confidence=0.95,
                metadata={"mode": "mock", "i2c_bus": self.i2c_bus,
                          "i2c_address": hex(self.i2c_address)},
            )
        ]

    # ------------------------------------------------------------------
    # Serial mode (BMI270 ESP32 format)
    # ------------------------------------------------------------------

    def _read_serial(self) -> list[SensorObservation]:
        return [self._apply_bias(obs) for obs in self._read_serial_raw()]

    def _read_serial_raw(self) -> list[SensorObservation]:
        if self.serial_port is None:
            return []

        try:
            from src.sensors.lab_integration.imu import parse_imu_line
        except ImportError:
            return []

        try:
            import serial as pyserial
        except ImportError:
            return []

        if self._serial_conn is None:
            try:
                self._serial_conn = pyserial.Serial(
                    self.serial_port, baudrate=self.baudrate, timeout=0.1,
                )
                self._serial_conn.reset_input_buffer()
                time.sleep(0.05)
            except Exception as e:
                print(f"  Warning: could not open {self.serial_port} ({e})")
                return []

        observations: list[SensorObservation] = []
        try:
            ser = self._serial_conn
            ser.reset_input_buffer()
            for _ in range(5):
                raw = ser.readline()
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                parsed = parse_imu_line(line)
                if parsed is None:
                    continue
                ax, ay, az, gx, gy, gz = parsed
                obs = {
                    "accel_x": ax,
                    "accel_y": ay,
                    "accel_z": az,
                    "gyro_x": gx,
                    "gyro_y": gy,
                    "gyro_z": gz,
                }
                observations.append(
                    SensorObservation(
                        sensor_id=self.sensor_id,
                        sensor_type=self.sensor_type,
                        timestamp=datetime.now(timezone.utc),
                        observation=obs,
                        confidence=0.95,
                        metadata={"mode": "serial",
                                  "serial_port": self.serial_port,
                                  "baudrate": self.baudrate},
                    )
                )
                break
            if not observations:
                self.close()
        except Exception as e:
            self.close()
        return observations

    def close(self) -> None:
        if self._serial_conn is not None:
            try:
                self._serial_conn.close()
            except Exception:
                pass
            self._serial_conn = None


__all__ = ["ImuSensor"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IMU sensor self-test & calibration")
    parser.add_argument("--mode", choices=["mock", "serial"], default="mock")
    parser.add_argument("--serial-port")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--zero", action="store_true",
                        help="Calibrate gyro/accel bias (keep stationary)")
    parser.add_argument("--reset-bias", action="store_true",
                        help="Clear stored bias offsets")
    parser.add_argument("--num-samples", type=int, default=50,
                        help="Number of samples for calibration (default 50)")
    args = parser.parse_args()
    sensor = ImuSensor(sensor_id="imu-self-test", mode=args.mode,
                       serial_port=args.serial_port, baudrate=args.baud)

    if args.reset_bias:
        sensor.reset_bias()

    if args.zero:
        sensor.zero(num_samples=args.num_samples)

    print(f"Reading from {sensor.sensor_id} ({args.mode})")
    if args.zero or args.reset_bias:
        print("  Gyro bias:" + (
            f" ({sensor._gyro_bias[0]:.4f}, {sensor._gyro_bias[1]:.4f}, {sensor._gyro_bias[2]:.4f})"
            if sensor._gyro_bias else " None"
        ))
        print("  Accel bias:" + (
            f" ({sensor._accel_bias[0]:.4f}, {sensor._accel_bias[1]:.4f}, {sensor._accel_bias[2]:.4f})"
            if sensor._accel_bias else " None"
        ))
    for o in sensor.read():
        obs = o.observation
        print(f"[{o.timestamp.isoformat()}] {o.sensor_type}/{o.sensor_id}  "
              f"accel=({obs.get('accel_x', 0):.3f}, {obs.get('accel_y', 0):.3f}, "
              f"{obs.get('accel_z', 0):.3f}) g  "
              f"gyro=({obs.get('gyro_x', 0):.3f}, {obs.get('gyro_y', 0):.3f}, "
              f"{obs.get('gyro_z', 0):.3f}) dps  conf={o.confidence}")
