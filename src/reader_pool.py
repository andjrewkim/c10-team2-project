"""Sensor reader pool — shared background-thread polling for all sensor readers.

Every tool (``realtime_demo.py``, ``collect.py``, ``dashboard/server.py``)
can instantiate a :class:`ReaderPool` with the same sensor config and get
continuous readings on a callback.

Usage
-----
    pool = ReaderPool(sensors_config_path="config/sensors.example.yaml")
    pool.set_callback(my_callback)         # called on every new reading
    pool.start()
    ...
    pool.stop()
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any, Optional

from sensors.base import BaseSensor, SensorObservation


def _import_sensor(class_path: str, sensor_id: str, params: dict[str, Any]) -> BaseSensor:
    """Dynamically import and instantiate a sensor class by dotted path."""
    import importlib

    module_path, _, class_name = class_path.rpartition(".")
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls(sensor_id=sensor_id, **params)


def load_sensors_from_config(path: str) -> list[BaseSensor]:
    """Load a list of instantiated sensors from a YAML config file."""
    import yaml

    with open(path) as f:
        raw = yaml.safe_load(f)

    sensors: list[BaseSensor] = []
    for entry in raw.get("sensors", []):
        sid: str = entry["id"]
        class_path: str = entry["class"]
        params: dict[str, Any] = dict(entry.get("params", {}))
        # Remove position from params if present (it's handled by the sensor's constructor)
        sensors.append(_import_sensor(class_path, sid, params))
    return sensors


class ReaderPool:
    """Manages a set of background threads, one per sensor, that poll ``read()``.

    Each thread calls the sensor's ``read()`` method at its natural rate
    (subject to a configurable minimum interval).  New observations are
    delivered to the registered callback — the same callback is invoked
    from *multiple* threads, so it must be thread-safe.
    """

    def __init__(
        self,
        sensors: list[BaseSensor],
        min_interval: float = 0.05,  # 50 ms minimum between polls
    ) -> None:
        self._sensors = sensors
        self._min_interval = min_interval
        self._callback: Optional[Callable[[SensorObservation], None]] = None
        self._threads: list[threading.Thread] = []
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_callback(self, cb: Callable[[SensorObservation], None]) -> None:
        self._callback = cb

    def start(self) -> None:
        self._stop_event.clear()
        self._threads = []
        for sensor in self._sensors:
            t = threading.Thread(
                target=_poll_loop,
                args=(sensor, self._min_interval, self._stop_event, self._callback),
                daemon=True,
                name=f"reader-{sensor.sensor_id}",
            )
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=3)

    @property
    def sensors(self) -> list[BaseSensor]:
        return list(self._sensors)


def _poll_loop(
    sensor: BaseSensor,
    min_interval: float,
    stop_event: threading.Event,
    callback: Optional[Callable[[SensorObservation], None]],
) -> None:
    """Background loop: call ``sensor.read()`` and invoke callback."""
    while not stop_event.is_set():
        t0 = time.monotonic()
        try:
            observations = sensor.read()
        except Exception:
            observations = []
        if callback is not None:
            for obs in observations:
                try:
                    callback(obs)
                except Exception:
                    pass
        elapsed = time.monotonic() - t0
        remaining = min_interval - elapsed
        if remaining > 0:
            stop_event.wait(remaining)
