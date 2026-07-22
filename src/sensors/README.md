# Sensors

Every sensor node publishes observations conforming to the
`SensorObservation` dataclass contract defined in `base.py`.

## Adding a new sensor

1. Create a new file in this directory (e.g. `pir_sensor.py`).
2. Subclass `BaseSensor` and implement `read()`.
3. Store any hardware-specific configuration in the instance (or accept
   it via `__init__`).  Do **not** hardcode thresholds, topic names, or
   activity labels.
4. Register the new sensor in `__init__.py` if you want it importable
   from the package root.

**Rule of thumb**: fusion and action modules should never `import` a
concrete sensor class.  They work against `SensorObservation` only.
