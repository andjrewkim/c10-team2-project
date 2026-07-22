from sensors.base import SensorObservation, BaseSensor
from sensors.mock_sensor import MockSensor

from datetime import datetime, timezone


def test_sensor_observation_contract() -> None:
    obs = SensorObservation(
        sensor_id="test-01",
        sensor_type="mock",
        timestamp=datetime.now(timezone.utc),
        observation={"temp": 22.5},
        confidence=0.85,
        metadata={"fw": "1.0"},
    )
    assert obs.sensor_id == "test-01"
    assert obs.sensor_type == "mock"
    assert 0.0 <= obs.confidence <= 1.0


def test_mock_sensor_produces_valid_observations() -> None:
    sensor: BaseSensor = MockSensor(sensor_id="mock-001")
    obs_list = sensor.read()
    assert len(obs_list) == 1
    obs = obs_list[0]
    assert obs.sensor_id == "mock-001"
    assert obs.sensor_type == "mock"
    assert 0.0 <= obs.confidence <= 1.0
    assert isinstance(obs.timestamp, datetime)
    assert obs.metadata.get("mock") is True


def test_mock_sensor_randomness() -> None:
    sensor = MockSensor()
    confidences = {sensor.read()[0].confidence for _ in range(50)}
    assert len(confidences) > 1
