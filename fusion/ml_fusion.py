"""ML-based fusion strategy using a pre-trained sklearn model.

Loads ``fusion/model.pkl`` at startup and uses the trained model to
produce a fused prediction + confidence from live sensor observations.

If no model file exists yet, gracefully falls back to
``WeightedAverageFusion`` so the pipeline still runs before training.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

from fusion.base import FusedResult, FusionStrategy
from fusion.weighted_average import WeightedAverageFusion
from sensors.base import SensorObservation

MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"


def _load_model_data() -> dict[str, Any] | None:
    if not MODEL_PATH.exists():
        return None
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except Exception as exc:
        print(f"[ml_fusion] Failed to load model: {exc}", file=sys.stderr)
        return None


class MlFusion(FusionStrategy):
    """Fusion strategy powered by a pre-trained sklearn model.

    Falls back to ``WeightedAverageFusion`` if the model file is absent,
    making the pipeline runnable before any training has occurred.
    """

    def __init__(self, type_weights: dict[str, float] | None = None) -> None:
        data = _load_model_data()
        if data is None:
            print("[ml_fusion] No model.pkl — falling back to WeightedAverageFusion")
            self._fallback = WeightedAverageFusion(type_weights=type_weights)
            self._model = None
            self._label_names: list[str] = []
            self._col_names: list[str] = []
            self._model_name: str = "fallback"
        else:
            self._model = data["model"]
            self._label_names = data.get("label_names", [])
            self._col_names = data.get("col_names", [])
            self._model_name = data.get("model_name", "unknown")
            self._fallback = None
            cls_count = len(self._label_names)
            print(f"[ml_fusion] Loaded model: {self._model_name} ({cls_count} classes)")

    def fuse(self, observations: list[SensorObservation]) -> FusedResult:
        if self._model is None or self._fallback is not None:
            return self._fallback.fuse(observations)  # type: ignore[union-attr]

        # Build feature vector matching training columns
        sensor_conf: dict[str, float] = {}
        for obs in observations:
            sensor_conf[obs.sensor_id] = obs.confidence

        feat = []
        missing: list[str] = []
        for col in self._col_names:
            if col.endswith("_confidence"):
                sid = col[: -len("_confidence")]
                feat.append(sensor_conf.get(sid, 0.0))
            elif col.endswith("_missing"):
                sid = col[: -len("_missing")]
                is_missing = 1.0 if sid not in sensor_conf else 0.0
                feat.append(is_missing)
                if is_missing:
                    missing.append(sid)

        if not feat:
            return FusedResult()

        x = np.array([feat], dtype=np.float64)
        probs = self._model.predict_proba(x)[0]  # type: ignore[union-attr]
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])

        label = (
            self._label_names[pred_class]
            if pred_class < len(self._label_names)
            else f"class-{pred_class}"
        )

        class_probs = {
            self._label_names[i] if i < len(self._label_names) else str(i): float(p)
            for i, p in enumerate(probs)
        }

        return FusedResult(
            activity_label=label,
            confidence=confidence,
            contributing_sensors=[obs.sensor_id for obs in observations],
            meta={
                "strategy": "ml_fusion",
                "model": self._model_name,
                "n_observations": len(observations),
                "n_missing_sensors": len(missing),
                "class_probs": class_probs,
            },
        )

    def reset(self) -> None:
        if self._fallback is not None:
            self._fallback.reset()
