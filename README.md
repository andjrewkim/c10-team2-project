# Gesture Recognition System

Collect labelled gesture data from mmWave radar / IMU / UWB / WiFi / RFID,
extract sliding-window features, train classifiers, and run real-time inference.

## Quick Start (mock mode — no hardware)

```bash
# 1. Collect synthetic gesture data
python -m src.collect \
    --gestures push pull left right \
    --duration 3 --trials 3

# 2. Merge recordings
python -m src.combine_datasets

# 3. Extract features
python -m src.extract_features --window 5 --stride 3

# 4. Train classifiers
python -m src.train --classifiers random_forest knn svm_rbf

# 5. Evaluate (confusion matrix → results/figures/)
python -m src.evaluate

# 6. Live demo
python -m src.realtime_demo
```

## Pipeline

```
collect → combine_datasets → extract_features → train → evaluate → realtime_demo
```

| Step | Output |
|------|--------|
| `collect.py` | `data/raw/*.jsonl` |
| `combine_datasets.py` | `data/processed/combined_dataset.json` |
| `extract_features.py` | `data/processed/features.npz` |
| `train.py` | `models/*.pkl`, `models/train_results.json` |
| `evaluate.py` | `results/evaluation_results.json`, `results/figures/*.png` |
| `realtime_demo.py` | Live terminal predictions |

## With Real Sensors

```bash
python -m src.collect \
    --sensors mmwave imu \
    --mode serial \
    --gestures push pull clockwise anticlockwise \
    --duration 6 --trials 5
```

## Sensor Fusion

Use `--sensors mmwave imu` (or any combination) to capture multiple modalities.
The feature extractator automatically concatenates features from all present
sensors. Train single-sensor baselines separately, then compare against
the fused model.

## Gestures

`pull`, `push`, `clockwise`, `anticlockwise`, `right`, `left`, `bye-bye`,
`one-arm-boxing`, `clapping`, `two-arm-boxing`, `t-arm`, `raise-arms`,
`soli`, `making-fist-and-open`, `palm-up-down`

## Structure

```
src/
├── collect.py
├── combine_datasets.py
├── extract_features.py
├── train.py
├── evaluate.py
├── realtime_demo.py
└── sensors/
    ├── base_reader.py      # Abstract reader
    ├── mmwave_reader.py    # mmWave radar
    ├── imu_reader.py       # IMU
    ├── uwb_reader.py       # UWB ranging
    ├── wifi_reader.py      # WiFi RSSI/CSI
    ├── rfid_reader.py      # RFID
    ├── drivers/            # Hardware drivers
    └── lab_integration/    # Signal processing
data/
├── raw/                    # Recordings (.jsonl)
└── processed/              # Features (.npz)
models/                     # Trained models (.pkl)
results/figures/            # Confusion matrices (.png)
config/                     # Radar config (.cfg)
```

## Requirements

```
pip install -e .
```

For serial sensors: `pip install -e ".[serial]"`
For plots: `pip install -e ".[viz]"`
For everything: `pip install -e ".[all]"`
