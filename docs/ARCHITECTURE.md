# Architecture

## Data flow

```
  ┌──────────┐    ┌───────────┐    ┌─────────────┐    ┌──────────┐
  │  Sensor  │───▶│ Transport │───▶│   Fusion    │───▶│  Action  │
  │  (read)  │    │  (MQTT)   │    │  (strategy) │    │ (trigger)│
  └──────────┘    └───────────┘    └─────────────┘    └──────────┘
```

1. **Sensor** — calls `read()` on each registered sensor, producing
   `SensorObservation` instances.  The sensor knows nothing about
   transport, fusion, or actions.

2. **Transport** — serialises each observation and publishes it to a
   topic per naming convention `{location}/{type}/{id}/observation`.
   The reference transport is MQTT (paho-mqtt).

3. **Fusion** — subscribes to sensor topics (or receives observations
   in-process), collects a batch, and runs them through a
   `FusionStrategy`.  The output is a `FusedResult` with an activity
   label and aggregate confidence.

4. **Action** — passes the fused result through a chain of
   `ActionTrigger` instances via `evaluate()`.  Each trigger may
   fire independently (e.g. console log, alert, home automation).

## Key design rules

| Layer     | Imports from            | Is replaced by  |
|-----------|-------------------------|-----------------|
| Sensor    | nothing framework-y     | subclassing     |
| Transport | sensors (observations)  | new wrapper     |
| Fusion    | sensors (observations)  | new strategy    |
| Action    | fusion (FusedResult)    | new trigger     |

- **No cross-layer imports** of concrete implementations.
- Config files wire everything together at startup.
- Every extension point is `abstractmethod` or a typed protocol.

## Config-driven wiring

At startup the system reads:

- `config/sensors.yaml` — which sensors to instantiate
- `config/fusion.yaml` — which fusion strategy and its parameters
- `config/mqtt.yaml` — MQTT broker connection

All three have `.example.yaml` counterparts with documented fields.
