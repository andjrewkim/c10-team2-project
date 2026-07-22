# Training Module

Trains ML fusion models from recorded sensor sessions.

## Full pipeline

```bash
# 1. Record some sessions (requires MQTT broker + sensors)
python -m recording.cli --label "walking"   --participant alice --duration 30
python -m recording.cli --label "sitting"   --participant alice --duration 30
python -m recording.cli --label "walking"   --participant bob   --duration 30
python -m recording.cli --label "sitting"   --participant bob   --duration 30

# 2. Prepare feature matrix
python -m training.prepare_dataset --data-dir data/raw --output-dir data/processed

# 3. Train and compare models, save the best one
python -m training.train_fusion_model --dataset data/processed/dataset.npz --output fusion/model.pkl
```

After step 3, `run_demo.py` will automatically use the trained model
(via `fusion/ml_fusion.py`). If no model file exists it falls back to
`WeightedAverageFusion`.

## Scripts

| Script | Purpose |
|--------|---------|
| `prepare_dataset.py` | Loads session CSVs, aligns observations into time windows, creates train/test split |
| `train_fusion_model.py` | Compares LogisticRegression, kNN, and MLP; saves best to `fusion/model.pkl` |

## Requirements

- `numpy`, `scikit-learn` (installed via `pip install -e ".[train]"`)
- At least 5 labelled sessions across ≥2 activity classes for meaningful training
