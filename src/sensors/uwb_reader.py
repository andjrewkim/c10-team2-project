from __future__ import annotations

import numpy as np

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

        ranges_cm: list[float] = []
        position: dict[str, float] | None = None
        confidences: list[float] = []

        for obs in obs_list:
            o = obs.observation or {}
            confidences.append(obs.confidence)

            if o.get("ranges_cm"):
                ranges_cm.extend(o["ranges_cm"])
            elif o.get("range_m") is not None:
                ranges_cm.append(float(o["range_m"]) * 100.0)

            if o.get("position") and position is None:
                position = o["position"]
            if obs.position and position is None:
                position = obs.position

        return Reading(
            sensor_id=self.sensor_id,
            sensor_type=self.sensor_type,
            timestamp=obs_list[0].timestamp,
            data={
                "ranges_cm": ranges_cm,
                "position": position,
            },
            confidence=float(np.mean(confidences)) if confidences else 0.0,
        )

    def stop(self) -> None:
        self._started = False
