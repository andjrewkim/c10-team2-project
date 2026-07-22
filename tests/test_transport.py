from transport.mqtt_client import observation_topic, MqttConfig


def test_observation_topic_convention() -> None:
    topic = observation_topic("living-room", "mock", "mock-001")
    assert topic == "living-room/mock/mock-001/observation"
    assert topic.count("/") == 3


def test_mqtt_config_defaults() -> None:
    config = MqttConfig()
    assert config.host == "localhost"
    assert config.port == 1883
    assert config.qos == 1
    assert config.tls_enabled is False


def test_mqtt_config_custom_values() -> None:
    config = MqttConfig(host="mqtt.example.com", port=8883, username="user", password="pass")
    assert config.host == "mqtt.example.com"
    assert config.port == 8883
    assert config.username == "user"
