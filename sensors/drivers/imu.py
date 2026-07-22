"""IMU sensor driver (accelerometer + gyroscope, I2C).

Expected hardware: I2C inertial measurement unit (BNO055, MPU6050, ICM-20948).
Wiring: VCC→3.3V, GND→GND, SDA→GPIO2, SCL→GPIO3 (Raspberry Pi).

TODO: Replace stub body with real I2C reads using smbus2 or adafruit-circuitpython-bno055.
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timezone

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

    def read(self) -> list[SensorObservation]:
        """Blocking read from the IMU.

        Expected observation dict:
            accel_x, accel_y, accel_z : float (m/s²)
            gyro_x, gyro_y, gyro_z   : float (rad/s)
            (optional) mag, temperature, quaternion
        """
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


__all__ = ["ImuSensor"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IMU sensor self-test")
    parser.add_argument("--bus", type=int, default=1)
    parser.add_argument("--address", type=lambda x: int(x, 0), default=0x28)
    args = parser.parse_args()
    sensor = ImuSensor(sensor_id="imu-self-test", i2c_bus=args.bus, i2c_address=args.address)
    for o in sensor.read():
        print(f"[{o.timestamp.isoformat()}] {o.sensor_type}/{o.sensor_id}  obs={o.observation}  conf={o.confidence}")
