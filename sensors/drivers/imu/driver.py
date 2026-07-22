"""IMU sensor driver stub (accelerometer + gyroscope, I2C).

TODO: Replace the stub body with real I2C reads using a library such as
      smbus2, adafruit-circuitpython-bno055, or the manufacturer SDK.
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timezone
from typing import Any

from sensors.base import BaseSensor, SensorObservation


class ImuSensor(BaseSensor):
    """Driver for an I2C IMU (e.g. BNO055, MPU6050, ICM-20948).

    Parameters
    ----------
    sensor_id : str
        Unique identifier for this IMU instance.
    i2c_bus : int
        I2C bus number (typically 0 or 1 on Raspberry Pi).
    i2c_address : int
        7-bit I2C device address (e.g. 0x28 for BNO055).
    """

    def __init__(
        self,
        sensor_id: str,
        i2c_bus: int = 1,
        i2c_address: int = 0x28,
    ) -> None:
        super().__init__(sensor_id=sensor_id, sensor_type="imu")
        self.i2c_bus = i2c_bus
        self.i2c_address = i2c_address
        # TODO: Open I2C bus — e.g.
        #   self._bus = smbus2.SMBus(self.i2c_bus)
        #   self._device = self._bus.open(self.i2c_address)

    def read(self) -> list[SensorObservation]:
        """Blocking read from the IMU.

        Expected observation format (dict):
            accel_x  : float  (m/s²)
            accel_y  : float  (m/s²)
            accel_z  : float  (m/s²)
            gyro_x   : float  (rad/s)
            gyro_y   : float  (rad/s)
            gyro_z   : float  (rad/s)
            (optional) mag_x, mag_y, mag_z, temperature, quaternion...

        Returns
        -------
        list[SensorObservation]
            One observation per read cycle.
        """
        # TODO: Replace with real I2C register reads.
        #   For BNO055:
        #     accel = self._read_registers(ACCEL_DATA_START, 6)
        #     gyro  = self._read_registers(GYRO_DATA_START, 6)
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
                metadata={"i2c_bus": self.i2c_bus, "i2c_address": hex(self.i2c_address)},
            )
        ]

    # ------------------------------------------------------------------
    # TODO: add private helpers for I2C register access
    # ------------------------------------------------------------------


def _test_main() -> None:
    parser = argparse.ArgumentParser(description="IMU sensor self-test")
    parser.add_argument("--bus", type=int, default=1)
    parser.add_argument("--address", type=lambda x: int(x, 0), default=0x28)
    args = parser.parse_args()

    sensor = ImuSensor(sensor_id="imu-self-test", i2c_bus=args.bus, i2c_address=args.address)
    obs_list = sensor.read()
    for obs in obs_list:
        print(f"[{obs.timestamp.isoformat()}] {obs.sensor_type}/{obs.sensor_id}")
        print(f"  observation: {obs.observation}")
        print(f"  confidence:  {obs.confidence}")


if __name__ == "__main__":
    _test_main()
