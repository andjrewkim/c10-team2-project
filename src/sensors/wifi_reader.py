from __future__ import annotations

from src.sensors.base_reader import BaseReader, Reading
from src.sensors.drivers.wifi import WiFiSensor


class WiFiReader(BaseReader):
    def __init__(
        self,
        sensor_id: str = "wifi-0",
        interface: str = "wlan0",
        mode: str = "mock",
        bssids: list[str] | None = None,
        serial_port: str | None = None,
    ):
        super().__init__(sensor_id=sensor_id, sensor_type="wifi")
        self._sensor = WiFiSensor(
            sensor_id=sensor_id,
            interface=interface,
            mode=mode,
            bssids=bssids,
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
                data={"rssi": {}, "csi": None},
                confidence=0.0,
            )
        obs = obs_list[0]
        o = obs.observation or {}
        return Reading(
            sensor_id=self.sensor_id,
            sensor_type=self.sensor_type,
            timestamp=obs.timestamp,
            data={
                "rssi": o.get("rssi", {}),
                "csi": o.get("csi"),
                "amplitudes": o.get("amplitudes"),
            },
            confidence=obs.confidence,
        )

    def stop(self) -> None:
        self._started = False
