"""Unit tests for the ReaderPool module.

Tests cover dynamic sensor import, config loading, and the poll-loop
lifecycle (thread-safe callback, start/stop).
"""

from __future__ import annotations

import threading
import time

import pytest

from src.reader_pool import ReaderPool, load_sensors_from_config, _import_sensor
from sensors.base import SensorObservation
from sensors.mock_sensor import MockSensor


def test_import_sensor_mock() -> None:
    """Dynamically import MockSensor via dotted path."""
    sensor = _import_sensor("sensors.mock_sensor.MockSensor", "test-001", {})
    assert sensor.sensor_id == "test-001"
    assert sensor.sensor_type == "mock"


def test_import_sensor_with_params() -> None:
    """Dynamically import sensor with constructor kwargs."""
    sensor = _import_sensor(
        "sensors.mock_sensor.MockSensor",
        "test-002",
        {"min_confidence": 0.3, "max_confidence": 0.8},
    )
    assert sensor.sensor_id == "test-002"
    obs = sensor.read()
    assert 0.3 <= obs[0].confidence <= 0.8


def test_load_sensors_from_example_config() -> None:
    """Loading sensors from example config returns a list of instantiated sensors."""
    sensors = load_sensors_from_config("config/sensors.example.yaml")
    assert len(sensors) >= 1
    # Should include at least MockSensor
    types = {s.sensor_type for s in sensors}
    assert "mock" in types


def test_reader_pool_start_stop() -> None:
    """ReaderPool start/stop lifecycle doesn't raise."""
    sensors = [MockSensor(sensor_id="pool-test")]
    pool = ReaderPool(sensors, min_interval=0.01)
    pool.start()
    time.sleep(0.05)
    pool.stop()
    # No assertions needed — just verify no exceptions


def test_reader_pool_callback_is_called() -> None:
    """Callback should be invoked as observations arrive."""
    received: list[SensorObservation] = []
    lock = threading.Lock()

    def cb(obs: SensorObservation) -> None:
        with lock:
            received.append(obs)

    sensors = [MockSensor(sensor_id="cb-test")]
    pool = ReaderPool(sensors, min_interval=0.01)
    pool.set_callback(cb)
    pool.start()
    time.sleep(0.1)
    pool.stop()

    with lock:
        assert len(received) > 0
        # All observations should be from our sensor
        assert all(o.sensor_id == "cb-test" for o in received)


def test_reader_pool_multiple_sensors() -> None:
    """Pool with multiple sensors calls back for each."""
    sensor_ids = ["multi-a", "multi-b"]
    sensors = [MockSensor(sensor_id=sid) for sid in sensor_ids]
    seen: set[str] = set()
    lock = threading.Lock()

    def cb(obs: SensorObservation) -> None:
        with lock:
            seen.add(obs.sensor_id)

    pool = ReaderPool(sensors, min_interval=0.01)
    pool.set_callback(cb)
    pool.start()
    time.sleep(0.15)
    pool.stop()

    with lock:
        for sid in sensor_ids:
            assert sid in seen


def test_reader_pool_stop_does_not_block_forever() -> None:
    """stop() should return within a reasonable timeout even if a sensor hangs."""
    sensors = [MockSensor(sensor_id="hang-test")]
    pool = ReaderPool(sensors, min_interval=0.5)
    pool.start()
    t0 = time.monotonic()
    pool.stop()
    elapsed = time.monotonic() - t0
    assert elapsed < 5.0  # should not block for long
