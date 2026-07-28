# Gesture Recognition System

Collect labelled gesture data from mmWave radar / IMU / UWB / WiFi / RFID,
extract sliding-window features, train classifiers, and run real-time inference.

## Quick Start (mock mode — no hardware)

```bash
# 1. Collect synthetic gesture data
python -m src.collect \
    --gestures push pull left right \
    --duration 3 --trials 3

# 2. Inspect the data (plots centroid, velocity, point count)
python -m src.visualize --mode mmwave
python -m src.visualize --mode overlay --field mean_y

# 3. Merge recordings
python -m src.combine_datasets

# 4. Extract features
python -m src.extract_features --window 5 --stride 3

# 5. Train classifiers
python -m src.train --classifiers random_forest knn svm_rbf

# 6. Evaluate (confusion matrix → results/figures/)
python -m src.evaluate

# 7. Run real-time demo (classifies mock data)
python -m src.realtime_demo --mode mock
```

## Live Radar Viewer

```bash
# With mock data (no radar)
python -m src.live_view --mode mock

# With real mmWave radar
python -m src.live_view --mode serial
```

Opens a matplotlib window with 3D point cloud, top-down view, centroid/velocity time series, and point count graph.

## Pipeline

```
collect → visualize → combine_datasets → extract_features → train → evaluate → realtime_demo
```

| Script | What it does | Output |
|--------|-------------|--------|
| `live_view.py` | Live radar visualization | Plot window |
| `collect.py` | Record labelled gesture trials | `data/raw/*.jsonl` |
| `visualize.py` | Plot recorded data for inspection | Plot window |
| `combine_datasets.py` | Merge recordings into one dataset | `data/processed/combined_dataset.json` |
| `extract_features.py` | Sliding-window feature extraction | `data/processed/features.npz` |
| `train.py` | Train and compare classifiers | `models/*.pkl`, `models/train_results.json` |
| `evaluate.py` | Accuracy, confusion matrix, plots | `results/evaluation_results.json`, `results/figures/*.png` |
| `realtime_demo.py` | Live terminal classification | Terminal predictions |

## With Real Sensors

```bash
# Collect gesture data from mmWave radar
python -m src.collect \
    --sensors mmwave \
    --mode serial \
    --gestures push pull clockwise anticlockwise \
    --duration 6 --trials 5

# Collect from multiple sensors (fusion)
python -m src.collect \
    --sensors mmwave imu \
    --mode serial \
    --gestures push pull \
    --duration 5 --trials 5
```

## Sensor Fusion

Use `--sensors mmwave imu` (or any combination) to capture multiple modalities.
The feature extractor automatically concatenates features from all present
sensors. Train single-sensor baselines separately, then compare against
the fused model.

## Gestures

`pull`, `push`, `clockwise`, `anticlockwise`, `right`, `left`, `bye-bye`,
`one-arm-boxing`, `clapping`, `two-arm-boxing`, `t-arm`, `raise-arms`,
`soli`, `making-fist-open`, `palm-up-down`

## Structure

```
src/
├── live_view.py            # Live radar visualization
├── collect.py              # Gesture data collection
├── combine_datasets.py     # Merge recordings
├── extract_features.py     # Feature engineering
├── train.py                # Model training
├── evaluate.py             # Evaluation + confusion matrix
├── realtime_demo.py        # Live classification
├── visualize.py            # Plot recorded data
└── sensors/
    ├── base_reader.py      # Abstract reader
    ├── mmwave_reader.py    # mmWave radar
    ├── imu_reader.py       # IMU
    ├── uwb_reader.py       # UWB ranging
    ├── wifi_reader.py      # WiFi RSSI/CSI
    ├── rfid_reader.py      # RFID
    ├── mock_sensor.py      # Mock sensor for testing
    ├── reader_pool.py      # Background reader pool
    ├── base.py             # Base sensor ABC
    ├── drivers/            # Hardware drivers (serial/mock)
    └── lab_integration/    # COSMOS signal processing
data/
├── raw/                    # Recordings (.jsonl)
└── processed/              # Feature matrices (.npz)
models/                     # Trained models (.pkl)
results/figures/            # Confusion matrices (.png)
config/                     # Radar CFG + sensor YAML
```

## Requirements

```
pip install -e .
```

For serial sensors: `pip install -e ".[serial]"`
For plots: `pip install -e ".[viz]"`
For everything: `pip install -e ".[all]"`
