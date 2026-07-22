# Recording Module

This module captures labeled sensor data for later training.  It runs
alongside the live fusion pipeline or as a standalone tool.

## Quickstart

```bash
# Ensure sensors are connected and the MQTT broker is running.
python -m recording.cli --label "walking" --participant alice --duration 30
```

The script:
1. Prints a 5-second countdown.
2. Records all sensor observations arriving via MQTT for 30 seconds.
3. Writes a CSV file to `data/raw/`.

## CLI reference

```
python -m recording.cli \
    --label "walking" \
    --participant alice \
    --duration 60 \
    --mqtt-config config/mqtt.example.yaml \
    --output-dir data/raw
```

Required: ``--label``, ``--participant``
Optional: ``--duration`` (default 30), ``--mqtt-config``, ``--output-dir``

If ``--mqtt-config`` is omitted, the script reads environment variables
``MQTT_HOST``, ``MQTT_PORT``, ``MQTT_USERNAME``, ``MQTT_PASSWORD``.

## Not all sensors need to be connected

The recording session subscribes to ``+/+/+/observation`` and writes
whatever it receives.  If only a subset of sensors are connected,
only those will appear in the output file.  The ``sensor_id`` and
``sensor_type`` columns identify which sensor generated each row.

## Output format

CSV with columns:

| Column           | Type    | Description                                |
|------------------|---------|--------------------------------------------|
| session_id       | str     | Unique session identifier                  |
| label            | str     | Activity label                             |
| participant_id   | str     | Person performing the activity             |
| sensor_id        | str     | Sensor that generated this observation     |
| sensor_type      | str     | Sensor type (imu, uwb, mmwave, rfid, wifi) |
| tag_id           | str|None | Tag/entity identifier (RFID, WiFi, UWB)    |
| timestamp        | str     | ISO-8601 UTC                               |
| observation      | str     | JSON-encoded sensor-specific payload       |
| confidence       | float   | Sensor confidence (0–1)                    |
| position_x/y/z   | float|None | Fixed sensor position, if known            |
| raw_metadata     | str     | JSON-encoded metadata bag                  |

## TODO: dataset sharing

Recorded CSV files are gitignored.  The team should periodically sync
the ``data/raw/`` folder to a shared drive, cloud bucket, or DVC remote
(e.g. Google Drive, S3, or a NAS).  Update this note once the location
is agreed upon.
