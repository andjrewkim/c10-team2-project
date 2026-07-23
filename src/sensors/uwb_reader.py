from __future__ import annotations

from src.sensors.base_reader import BaseReader, Reading
from src.sensors.drivers.uwb import UwbAnchorSensor


class UwbReader(BaseReader):
    def __init__(
        self,
        sensor_id: str = "uwb-0",
        mode: str = "mock",
        serial_port: str = "/dev/ttyACM0",
        baudrate: int = 115200,
        position: dict[str, float] | None = None,
    ):
        super().__init__(sensor_id=sensor_id, sensor_type="uwb")
        self._sensor = UwbAnchorSensor(
            sensor_id=sensor_id,
            serial_port=serial_port,
            baudrate=baudrate,
            position=position,
            mode=mode,
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
                data={"ranges_cm": [], "position": None},
                confidence=0.0,
            )
        obs = obs_list[0]
        o = obs.observation or {}
        return Reading(
            sensor_id=self.sensor_id,
            sensor_type=self.sensor_type,
            timestamp=obs.timestamp,
            data={
                "ranges_cm": o.get("ranges_cm", []),
                "position": o.get("position"),
                "raw_ranges": o.get("raw_ranges", []),
            },
            confidence=obs.confidence,
        )

    def stop(self) -> None:
        self._started = False
