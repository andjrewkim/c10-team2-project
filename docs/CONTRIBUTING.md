# Contributing

## Adding a new sensor type

1. Create `sensors/your_sensor.py`.
2. Subclass `BaseSensor` and implement `read()` → `list[SensorObservation]`.
3. Add a config entry to the `sensors` list in your config file.
4. Done — no fusion or action changes needed.

**Do not** hardcode topic strings, activity labels, or fusion weights
in the sensor class.  Those belong in config files.

## Adding a new fusion strategy

1. Create `fusion/your_strategy.py`.
2. Subclass `FusionStrategy` and implement `fuse()` and `reset()`.
3. Update `config/fusion.yaml` to point `strategy` at your new class.
4. Done — no sensor or action code changes needed.

## Adding a new action

1. Create `actions/your_action.py`.
2. Subclass `ActionTrigger` and implement `evaluate()`.
3. Wire it into the pipeline in `run_demo.py` (or your orchestrator).
4. Done — no sensor or fusion code changes needed.

## Before submitting

- Run `pytest tests/` — all tests must pass.
- Run `mypy .` if your change touches type annotations.
- No secrets, hardcoded IPs, or credentials in any committed file.

## Style

- Type hints everywhere.
- Dataclasses for data containers; ABCs for pluggable interfaces.
- Config values in YAML; env vars for secrets.
