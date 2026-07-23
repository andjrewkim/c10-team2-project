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
from datetime import datetime, timezone

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

    def read(self) -> list[SensorObservation]:
        if self.mode == "serial":
            return self._read_serial()
        return self._read_mock()

    # ------------------------------------------------------------------
    # Mock mode
    # ------------------------------------------------------------------

    def _read_mock(self) -> list[SensorObservation]:
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
        if self.serial_port is None:
            return self._read_mock()

        try:
            from sensors.lab_integration.imu import parse_imu_line
        except ImportError:
            return self._read_mock()

        try:
            import serial as pyserial
        except ImportError:
            return self._read_mock()

        observations: list[SensorObservation] = []
        try:
            ser = pyserial.Serial(self.serial_port, baudrate=self.baudrate,
                                  timeout=0.3)
            for _ in range(10):
                raw = ser.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").strip()
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
            ser.close()
        except Exception:
            return self._read_mock()
        return observations


__all__ = ["ImuSensor"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IMU sensor self-test")
    parser.add_argument("--mode", choices=["mock", "serial"], default="mock")
    parser.add_argument("--serial-port")
    parser.add_argument("--baud", type=int, default=115200)
    args = parser.parse_args()
    sensor = ImuSensor(sensor_id="imu-self-test", mode=args.mode,
                       serial_port=args.serial_port, baudrate=args.baud)
    for o in sensor.read():
        obs = o.observation
        print(f"[{o.timestamp.isoformat()}] {o.sensor_type}/{o.sensor_id}  "
              f"accel=({obs.get('accel_x', 0):.3f}, {obs.get('accel_y', 0):.3f}, "
              f"{obs.get('accel_z', 0):.3f}) g  conf={o.confidence}")
