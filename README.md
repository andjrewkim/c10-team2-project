# IoT Activity Detection

Pluggable multi-sensor activity-detection framework. Sensors feed
observations into a transport layer (MQTT reference implementation),
which a fusion strategy combines into a single activity estimate,
which action triggers consume.

```
Sensor  ──▶ Transport ──▶ Fusion ──▶ Action
(read)      (MQTT)       (strategy)  (trigger)
```

## Quickstart

```bash
pip install -e ".[dev]"
python run_demo.py
```

This runs a mock sensor → weighted-average fusion → console-log action
pipeline. No hardware or network required.

## Structure

| Directory   | Purpose                                   |
|-------------|-------------------------------------------|
| `sensors/`  | Sensor base class + mock implementation   |
| `transport/`| MQTT client wrapper                       |
| `fusion/`   | Fusion strategy interface + reference     |
| `actions/`  | Action trigger interface + example        |
| `config/`   | YAML example configs + Pydantic validation |
| `tests/`    | pytest suite                              |
| `docs/`     | Architecture & contributing guides        |

## Guiding principles

- **Sensor-agnostic** — adding a sensor type never touches fusion or action code.
- **Fusion-agnostic** — swap strategies behind a common interface.
- **Config over hardcoding** — weights, thresholds, topics in YAML.
- **Everything typed** — type hints throughout.

## License

MIT
