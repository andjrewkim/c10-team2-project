import json
from dataclasses import dataclass, field
from typing import Callable, Optional

try:
    import paho.mqtt.client as mqtt
except ImportError:
    mqtt = None  # type: ignore[assignment]

from sensors.base import SensorObservation

# ---------------------------------------------------------------------------
# Topic naming convention
# ---------------------------------------------------------------------------
#   {location}/{sensor_type}/{sensor_id}/observation
#
# Examples:
#   living-room/mock/mock-001/observation
#   factory-floor-3/mmwave/radar-07/observation
#
# All levels are lower-case, hyphen-separated tokens.
# ---------------------------------------------------------------------------


def observation_topic(
    location: str,
    sensor_type: str,
    sensor_id: str,
) -> str:
    return f"{location}/{sensor_type}/{sensor_id}/observation"


@dataclass
class MqttConfig:
    """Config-driven broker settings.

    Every field has a default so the system can start without a config
    file in development, but production deployments MUST override via
    environment variables or a YAML config.
    """

    host: str = "localhost"
    port: int = 1883
    keepalive: int = 60
    username: Optional[str] = None
    password: Optional[str] = None
    tls_enabled: bool = False
    client_id: str = "iot-fusion-node"
    topic_prefix: str = ""
    qos: int = 1

    # Allow injecting extra kwargs for paho-mqtt
    extra: dict = field(default_factory=dict)


class MqttClient:
    """Thin wrapper around paho-mqtt.

    Usage
    -----
        config = MqttConfig(host="mqtt.eclipseprojects.io")
        client = MqttClient(config)
        client.connect()
        client.publish(obs)
        client.disconnect()
    """

    def __init__(self, config: MqttConfig) -> None:
        if mqtt is None:
            raise RuntimeError(
                "paho-mqtt is not installed. Install it with: pip install paho-mqtt"
            )
        self._config = config
        self._client: mqtt.Client = mqtt.Client(
            client_id=config.client_id,
            protocol=mqtt.MQTTv311,
        )
        if config.username:
            self._client.username_pw_set(config.username, config.password)
        self._on_message: Optional[Callable] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def connect(self) -> None:
        if self._config.tls_enabled:
            self._client.tls_set()
        self._client.connect(
            self._config.host,
            self._config.port,
            self._config.keepalive,
        )
        self._client.loop_start()

    def disconnect(self) -> None:
        self._client.loop_stop()
        self._client.disconnect()

    def publish(self, observation: SensorObservation, location: str = "") -> None:
        topic = self._resolve_topic(observation, location)
        payload = self._serialize(observation)
        self._client.publish(topic, payload, qos=self._config.qos)

    def subscribe(
        self,
        topic_filter: str,
        callback: Callable[[str, bytes], None],
    ) -> None:
        self._client.message_callback_add(topic_filter, _wrap_callback(callback))
        self._client.subscribe(topic_filter, qos=self._config.qos)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_topic(self, observation: SensorObservation, location: str) -> str:
        base = observation_topic(
            location=location or self._config.topic_prefix or "_",
            sensor_type=observation.sensor_type,
            sensor_id=observation.sensor_id,
        )
        return base

    @staticmethod
    def _serialize(observation: SensorObservation) -> str:
        return json.dumps(
            {
                "sensor_id": observation.sensor_id,
                "sensor_type": observation.sensor_type,
                "timestamp": observation.timestamp.isoformat(),
                "observation": observation.observation,
                "confidence": observation.confidence,
                "metadata": observation.metadata,
            },
            default=str,
        )


def _wrap_callback(user_cb: Callable[[str, bytes], None]) -> Callable:
    def _on_mqtt_message(_client, _userdata, msg) -> None:
        user_cb(msg.topic, msg.payload)
    return _on_mqtt_message
