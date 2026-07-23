from __future__ import annotations

from src.sensors.base_reader import BaseReader, Reading
from src.sensors.drivers.imu import ImuSensor


class ImuReader(BaseReader):
    def __init__(
        self,
        sensor_id: str = "imu-0",
        mode: str = "mock",
        serial_port: str | None = None,
    ):
        super().__init__(sensor_id=sensor_id, sensor_type="imu")
        self._sensor = ImuSensor(
            sensor_id=sensor_id,
            mode=mode,
            serial_port=serial_port,
        )
        self._started = False

    def start(self) -> None:
        self._started = True

    def read(self) -> Reading:
        obs_list = self._sensor.read()
        if not obs_list:
            return Reading(
                sensor_id=self.sensor_id,
                sensor_type=self.sensor_type,
                data={"accel": [0, 0, 0], "gyro": [0, 0, 0]},
                confidence=0.0,
            )
        obs = obs_list[0]
        o = obs.observation or {}
        return Reading(
            sensor_id=self.sensor_id,
            sensor_type=self.sensor_type,
            timestamp=obs.timestamp,
            data={
                "accel": [
                    o.get("accel_x", 0),
                    o.get("accel_y", 0),
                    o.get("accel_z", 0),
                ],
                "gyro": [
                    o.get("gyro_x", 0),
                    o.get("gyro_y", 0),
                    o.get("gyro_z", 0),
                ],
                "quat": o.get("quat"),
                "trajectory": o.get("trajectory"),
            },
            confidence=obs.confidence,
        )

    def zero(self, num_samples: int = 50) -> dict[str, tuple[float, float, float]]:
        return self._sensor.zero(num_samples=num_samples)

    def reset_bias(self) -> None:
        self._sensor.reset_bias()

    def stop(self) -> None:
        self._started = False
