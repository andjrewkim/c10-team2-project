# IoT Activity Detection

Multi-sensor activity-detection system with ML fusion — a university
semester project demonstrating a complete pipeline from raw sensor
data to activity prediction.

## What it does

Sensors (IMU, UWB, mmWave radar, RFID, WiFi) publish observations
over MQTT. The system can:

1. **Record** labelled sessions to disk (`recording/cli.py`).
2. **Train** an ML fusion model from those recordings (`training/`).
3. **Run live**: fuse sensor streams into an activity prediction
   and trigger actions (`run_demo.py`).

## Architecture

```ascii
Sensor ──▶ MQTT ──▶ RecordingSession ──▶ CSV on disk
(read)     bus                               │
                                             ▼
                                     training/prepare_dataset.py
                                             │
                                             ▼
                                     training/train_fusion_model.py
                                             │
                                             ▼
                                     fusion/model.pkl
                                             │
                      ┌──────────────────────┘
                      ▼
Live: sensor ──▶ MlFusion ──▶ Action
                (or WeightedAverage fallback)
```

## Quickstart (mock pipeline, no hardware)

```bash
pip install -e ".[dev]"
python run_demo.py
```

This runs a mock sensor → weighted-average fusion → console action.
No MQTT broker, no sensors needed.

## Full pipeline (real hardware)

```bash
# 1. Record sessions (requires MQTT broker + sensors)
python -m recording.cli --label "walking" --participant alice --duration 30
python -m recording.cli --label "sitting" --participant alice --duration 30

# 2. Prepare feature matrix for ML
pip install -e ".[train]"
python -m training.prepare_dataset --data-dir data/raw --output-dir data/processed

# 3. Train and compare models
python -m training.train_fusion_model --dataset data/processed/dataset.npz --output fusion/model.pkl

# 4. Run live with the trained model
python run_demo.py
```

After training, `run_demo.py` automatically loads `fusion/model.pkl`
via `MlFusion`. If no model file exists, it falls back to
`WeightedAverageFusion`.

## Live UI — Sensor Monitoring

```bash
pip install -e ".[ui]"
uvicorn ui.server:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000/ for a web dashboard with live sensor
cards and recording controls.

## Live UI — Localization Dashboard

Shows a floor-plan heatmap of the estimated person position in the
room, combining UWB trilateration with RFID, mmWave, WiFi, and IMU
belief adjustments. Sensor poller threads and the localisation engine
are shared via ``src/reader_pool.py`` and ``src/localization.py``:

```bash
pip install -e ".[ui]"
uvicorn src.dashboard.server:app --host 0.0.0.0 --port 8001
```

Open http://localhost:8001/ for a live floor-plan view with heatmap
overlay, position marker, and connected-sensor list.

## Project structure

| Directory         | Purpose                                              |
|-------------------|------------------------------------------------------|
| `sensors/`        | Base sensor + drivers (IMU, UWB, mmWave, RFID, WiFi) |
| `transport/`      | MQTT client wrapper                                  |
| `fusion/`         | Fusion strategies (WeightedAverage, MlFusion)        |
| `actions/`        | Action triggers (ConsoleAction)                      |
| `recording/`      | Labelled session capture → CSV                       |
| `training/`       | Dataset preparation + ML model training              |
| `ui/`             | FastAPI sensor-control dashboard                     |
| `src/`            | Shared reader pool, localisation engine, live dashboard |
| `config/`         | YAML config files (including localization.example.yaml) |
| `tests/`          | pytest suite                                         |

## Evaluation (for grading)

The ML fusion module (`fusion/ml_fusion.py`) compares logistic
regression, k-nearest neighbours, and a small MLP on real recorded
data, selecting the best model by test accuracy. The training script
prints accuracy + confusion matrix for each. See `training/README.md`.
