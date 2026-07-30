from __future__ import annotations

from src.sensors.base_reader import BaseReader, Reading
from src.sensors.drivers.mmwave import MmWaveRadarSensor


class MmWaveReader(BaseReader):
    def __init__(
        self,
        sensor_id: str = "mmwave-0",
        mode: str = "mock",
        serial_port: str = "/dev/cu.usbserial-BH00LUQT",
        cfg_path: str = "config/point_cloud.cfg",
    ):
        super().__init__(sensor_id=sensor_id, sensor_type="mmwave")
        self._sensor = MmWaveRadarSensor(
            sensor_id=sensor_id,
            mode=mode,
            serial_port=serial_port,
            cfg_path=cfg_path,
        )
        self._started = False
        # Cache last successful reading so we can detect genuine
        # disconnection (same object returned) vs. empty detections
        # (fresh object with zero data).  The staleness checker in
        # the demo uses id(reading) to differentiate these cases.
        self._last_reading: Reading | None = None

    def start(self) -> None:
        self._sensor.start()
        self._started = True

    def read(self) -> Reading:
        obs_list = self._sensor.read()
        if not obs_list:
            if self._last_reading is not None:
                return self._last_reading
            return Reading(
                sensor_id=self.sensor_id,
                sensor_type=self.sensor_type,
                data={"points": [], "num_points": 0, "range_profile": None, "motion_score": 0.0},
                confidence=0.0,
            )
        obs = obs_list[0]
        observation = obs.observation or {}
        points = observation.get("objects", [])
        self._last_reading = Reading(
            sensor_id=self.sensor_id,
            sensor_type=self.sensor_type,
            timestamp=obs.timestamp,
            data={
                "points": [
                    {
                        "x": float(p.get("x", 0)),
                        "y": float(p.get("y", 0)),
                        "z": float(p.get("z", 0)),
                        "velocity": float(p.get("doppler", p.get("velocity", 0))),
                        "snr": float(p.get("snr", 0)),
                    }
                    for p in points
                ],
                "num_points": len(points),
                "range_profile": observation.get("range_profile"),
                "motion_score": observation.get("motion_score", 0.0),
            },
            confidence=obs.confidence,
        )
        return self._last_reading

    def stop(self) -> None:
        if self._started:
            self._sensor.stop()
            self._started = False
