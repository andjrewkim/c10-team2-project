from __future__ import annotations

from src.sensors.base_reader import BaseReader, Reading
from src.sensors.drivers.rfid import RfidReaderSensor


class RfidReader(BaseReader):
    def __init__(
        self,
        sensor_id: str = "rfid-0",
        host: str = "192.168.1.100",
        port: int = 5084,
        mode: str = "mock",
        selected_epcs: list[str] | None = None,
    ):
        super().__init__(sensor_id=sensor_id, sensor_type="rfid")
        self._sensor = RfidReaderSensor(
            sensor_id=sensor_id,
            host=host,
            port=port,
            mode=mode,
            selected_epcs=selected_epcs,
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
                data={"tags": [], "touch": False},
                confidence=0.0,
            )
        obs = obs_list[0]
        o = obs.observation or {}
        return Reading(
            sensor_id=self.sensor_id,
            sensor_type=self.sensor_type,
            timestamp=obs.timestamp,
            data={
                "tags": o.get("tags", []),
                "touch": o.get("touch", False),
                "antenna": o.get("antenna"),
            },
            confidence=obs.confidence,
        )

    def stop(self) -> None:
        self._started = False
