"""Pydantic models for config file validation.

These models are used to validate YAML config files at startup so
that misconfigurations fail fast with a clear error message.
"""

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class SensorEntry(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    type: str
    class_: str = Field(alias="class")
    params: dict[str, Any] = Field(default_factory=dict)
    location: str = ""


class SensorsConfig(BaseModel):
    sensors: list[SensorEntry] = Field(default_factory=list)


class FusionConfig(BaseModel):
    strategy: str = "fusion.weighted_average.WeightedAverageFusion"
    type_weights: dict[str, float] = Field(default_factory=dict)
    cycle_seconds: float = 1.0


class MqttConfigModel(BaseModel):
    host: str = "localhost"
    port: int = 1883
    keepalive: int = 60
    username: Optional[str] = None
    password: Optional[str] = None
    tls_enabled: bool = False
    client_id: str = "iot-fusion-node"
    topic_prefix: str = ""
    qos: int = 1


def load_sensors_config(path: str) -> SensorsConfig:
    import yaml
    with open(path) as f:
        raw = yaml.safe_load(f)
    return SensorsConfig.model_validate(raw)


def load_fusion_config(path: str) -> FusionConfig:
    import yaml
    with open(path) as f:
        raw = yaml.safe_load(f)
    return FusionConfig.model_validate(raw)


def load_mqtt_config(path: str) -> MqttConfigModel:
    import yaml
    with open(path) as f:
        raw = yaml.safe_load(f)
    return MqttConfigModel.model_validate(raw)
