import pytest

from config.schema import (
    SensorsConfig,
    FusionConfig,
    MqttConfigModel,
    SensorEntry,
)


def test_sensors_config_valid() -> None:
    raw = {
        "sensors": [
            {
                "id": "s1",
                "type": "mock",
                "class": "sensors.mock_sensor.MockSensor",
                "params": {"min_confidence": 0.2},
            }
        ]
    }
    cfg = SensorsConfig.model_validate(raw)
    assert len(cfg.sensors) == 1
    assert cfg.sensors[0].id == "s1"
    assert cfg.sensors[0].params["min_confidence"] == 0.2


def test_sensors_config_empty() -> None:
    cfg = SensorsConfig.model_validate({"sensors": []})
    assert len(cfg.sensors) == 0


def test_fusion_config_defaults() -> None:
    cfg = FusionConfig.model_validate({})
    assert cfg.strategy == "fusion.weighted_average.WeightedAverageFusion"
    assert cfg.type_weights == {}
    assert cfg.cycle_seconds == 1.0


def test_fusion_config_custom_weights() -> None:
    raw = {"strategy": "custom.Strategy", "type_weights": {"a": 2.0, "b": 0.5}}
    cfg = FusionConfig.model_validate(raw)
    assert cfg.strategy == "custom.Strategy"
    assert cfg.type_weights["a"] == 2.0


def test_sensor_entry_with_position() -> None:
    raw = {
        "id": "uwb-1",
        "type": "uwb",
        "class": "sensors.drivers.uwb.UwbAnchorSensor",
        "params": {"serial_port": "/dev/ttyACM0", "position": {"x": 1.0, "y": 2.0, "z": 0.0}},
        "position": {"x": 1.0, "y": 2.0, "z": 0.0},
    }
    entry = SensorEntry.model_validate(raw)
    assert entry.position is not None
    assert entry.position.x == 1.0
    assert entry.position.y == 2.0
    assert entry.position.z == 0.0


def test_sensor_entry_position_optional() -> None:
    raw = {
        "id": "mock-1",
        "type": "mock",
        "class": "sensors.mock_sensor.MockSensor",
    }
    entry = SensorEntry.model_validate(raw)
    assert entry.position is None


def test_mqtt_config_model() -> None:
    cfg = MqttConfigModel.model_validate({})
    assert cfg.host == "localhost"
    assert cfg.port == 1883
    assert cfg.tls_enabled is False

    cfg2 = MqttConfigModel.model_validate({"host": "10.0.0.1", "tls_enabled": True})
    assert cfg2.host == "10.0.0.1"
    assert cfg2.tls_enabled is True
