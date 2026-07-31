# Gesture Recognition System

Collect labeled gesture data from mmWave radar + IMU,
extract sliding-window features, train classifiers, and run real-time inference.

## Gestures

`pull`, `push`, `clockwise`, `anticlockwise`, `right`, `left`, `bye-bye`,
`one-arm-boxing`, `clapping`, `two-arm-boxing`, `t-arm`, `raise-arms`,
`soli`, `making-fist-open`, `palm-up-down`

## Sensor Fusion

Use `--sensors mmwave imu` to capture both modalities.
The feature extractor automatically concatenates features from all present
sensors. Train single-sensor baselines separately, then compare against
the fused model.

## Quick Start

```bash
# 1. Collect gesture data
python -m src.collect \
    --gestures push pull left right \
    --mode serial \
    --sensors imu mmwave --output data/multi_raw \
    --imu_port COM16 --mmwave_port COM12 \
    --duration 2 --trials 3

# 2. Inspect the data (per sensor)
python -m src.visualize --input data/multi_raw --sensor imu --mode stats --no-details
python -m src.visualize --input data/multi_raw --sensor imu --mode overlay
python -m src.visualize --input data/multi_raw --sensor imu --mode compare
python -m src.visualize --input data/multi_raw --sensor imu --mode consistency

# 3. Merge recordings
python -m src.combine_datasets --input data/multi_raw --output data/multi_combined

# 4. Extract features
python -m src.extract_features \
    --input data/multi_combined/ --output data/multi_processed \
    --window 2 --stride 3

# 5. Train classifiers
python -m src.train \
    --input data/multi_processed --output models/multisensor \
    --classifiers random_forest knn svm_rbf \
    --sensors imu mmwave

# 6. Evaluate (confusion matrix → results/figures/)
python -m src.evaluate \
    --model models/multisensor/best_model.pkl --features data/multi_processed/features_*.npz \
    --output results/figures \
    --window 10 --stride 5

# 7. Run real-time demo (classifies mock data)
python -m src.realtime_demo \
    --model models/multisensor/best_model.pkl --features data/multi_processed/features_*.npz \
    --sensors imu mmwave --imu-port * --mmwave-port * \
    --mode serial \
    --window 5 \
    --idle-threshold 0.25 \
    --gyro-gain 1.0 \
    --gyro-deadband 0.8 \
    --accel-gain 1.0 \
    --min-conf 0.25 \
    --change-frames 8 --smooth 5 --min-vote 4 \
    --gesture-conf push=0.85 soli=0.90 \
    --gesture-min-movement push=0.15 soli=0.20 \
    --gui
```

## Pipeline

```
collect → visualize → combine_datasets → extract_features → train → evaluate → realtime_demo
```

| Script | What it does | Output |
|--------|-------------|--------|
| `collect.py` | Record labeled gesture trials | `data/raw/*.csv` |
| `visualize.py` | Plot recorded data for inspection | Plot window, `results/*_figures/*.png` |
| `combine_datasets.py` | Merge recordings into one dataset | `data/combined/combined_*.csv` |
| `extract_features.py` | Sliding-window feature extraction | `data/processed/features_*.npz` |
| `train.py` | Train and compare classifiers | `models/train_*/*.pkl` |
| `evaluate.py` | Accuracy, confusion matrix, plots | `results/figures/evaluate_*.json`, `results/figures/*.png` || `realtime_demo.py`        | Live terminal classification, GUI window | Terminal predictions |

### Per-Gesture Thresholds

When some gestures activate too easily (e.g. `push`/`pull` triggering on small
rotations), use per-gesture overrides instead of just raising the global
`--min-conf` or `--idle-threshold` which would affect all gestures equally.

**`--gesture-conf <gesture>=<threshold>`** — require higher prediction
confidence for specific gestures:
```bash
# push needs >= 85% confidence, soli >= 90%, others keep --min-conf
--gesture-conf push=0.85 soli=0.90
```

**`--gesture-min-movement <gesture>=<threshold>`** — require a minimum
movement score for specific gestures, preventing activation on tiny
jitter or rotation:
```bash
# push needs at least 0.15 movement; boxing gestures need 0.3
--gesture-min-movement push=0.15 pull=0.15 one-arm-boxing=0.3 two-arm-boxing=0.3
```

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
    ├── mock_sensor.py      # Mock sensor for testing
    ├── reader_pool.py      # Background reader pool
    ├── base.py             # Base sensor ABC
    └── drivers/            # Hardware drivers (serial/mock)
data/
├── combined/               # Combined recordings (.csv)
├── multi_combined/         # Combined recordings on multi-sensor data (.csv)
├── multi_processed/        # Feature matrices on multi-sensor data (.csv)
├── multi_raw/              # Recordings (.csv)
└── processed/              # Feature matrices (.npz)
models/                     # Trained models (.pkl)
└── multisensor/            # Trained models on multi-sensor data (.pkl)
results/                    
├── figures/                # Confusion matrices from multi-sensor data (.png)
├── imu_figures/            # Confusion matrices from IMU-only data (.png)
└── mmwave_figures/         # Confusion matrices from mmWave-only data (.png)
config/                     # Radar CFG + sensor YAML
```

## Requirements

```
pip install -e .
```

For serial sensors: `pip install -e ".[serial]"`
For plots: `pip install -e ".[viz]"`
For everything: `pip install -e ".[all]"`