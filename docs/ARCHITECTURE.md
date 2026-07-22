# Architecture

## Data flow

```
                                        ┌─────────────┐    ┌──────────┐
                                   ┌───▶│   Fusion    │───▶│  Action  │
                                   │    │  (strategy) │    │ (trigger)│
                                   │    └─────────────┘    └──────────┘
  ┌──────────┐    ┌───────────┐   │
  │  Sensor  │───▶│ Transport │───┤
  │  (read)  │    │  (MQTT)   │   │    ┌──────────────────┐
  └──────────┘    └───────────┘   └───▶│ RecordingSession  │───▶ /data/raw/
                                        │  (session logger) │
                                        └──────────────────┘
```

1. **Sensor** — calls `read()` on each registered sensor, producing
   `SensorObservation` instances.  The sensor knows nothing about
   transport, fusion, or actions.

2. **Transport** — serialises each observation and publishes it to a
   topic per naming convention `{location}/{type}/{id}/observation`.
   The reference transport is MQTT (paho-mqtt).

3. **Live path (Fusion + Action)** — subscribes to sensor topics,
   collects a batch, and runs them through a `FusionStrategy`.  The
   output `FusedResult` is evaluated by `ActionTrigger` instances.
   This path is optional — if no fusion is configured, sensors can
   still be recorded.

4. **Recording path** — `RecordingSession` subscribes to all sensor
   topics (``+/+/+/observation``) and writes every observation raw
   to a CSV file per session, tagged with label and participant ID.
   This data is used for offline training of ML fusion models.
   Multiple recording sessions can run with different sensor subsets.

## Key design rules

| Layer     | Imports from            | Is replaced by  |
|-----------|-------------------------|-----------------|
| Sensor    | nothing framework-y     | subclassing     |
| Transport | sensors (observations)  | new wrapper     |
| Fusion    | sensors (observations)  | new strategy    |
| Action    | fusion (FusedResult)    | new trigger     |
| Recording | transport (MQTT client) | N/A (standalone)|

- **No cross-layer imports** of concrete implementations.
- Config files wire everything together at startup.
- Every extension point is `abstractmethod` or a typed protocol.

## Config-driven wiring

At startup the system reads:

- `config/sensors.yaml` — which sensors to instantiate
- `config/fusion.yaml` — which fusion strategy and its parameters
- `config/mqtt.yaml` — MQTT broker connection

All three have `.example.yaml` counterparts with documented fields.
