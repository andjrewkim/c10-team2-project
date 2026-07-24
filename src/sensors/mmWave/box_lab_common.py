"""Feature extraction and classification helpers for mmWave box-content
and box-presence analysis.

Ported from the COSMOS ``box_lab_common.py`` lab module.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.sensors.mmWave.lab_mmwave import (
    RANGE_PROFILE_MAJOR,
    RANGE_PROFILE_MINOR,
    db_scale,
    fill_nan_series,
    read_box_data_frame,
    resample_matrix,
    resample_vector,
    robust_normalize,
)
from src.sensors.mmWave.lab_mmwave import point_cloud_from_tlvs, read_frame


INPUT_TYPE = "mmwave_box_contents_2d"
PRESENCE_INPUT_TYPE = "mmwave_box_presence"


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def safe_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return label.strip("_") or "unknown"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_manifest(dataset_dir: Path) -> list[dict[str, str]]:
    """Read ``trials.csv`` or ``trial_metadata.json`` files from a dataset dir."""
    manifest = dataset_dir / "trials.csv"
    if manifest.exists():
        with manifest.open(newline="") as f:
            return list(csv.DictReader(f))
    rows: list[dict[str, str]] = []
    for metadata_path in sorted(dataset_dir.rglob("trial_metadata.json")):
        try:
            metadata = json.loads(metadata_path.read_text())
        except json.JSONDecodeError:
            continue
        rows.append({k: "" if v is None else str(v) for k, v in metadata.items()})
    return rows


def source_dataset_name(dataset_dir: Path) -> str:
    meta = dataset_dir / "dataset_metadata.json"
    if meta.exists():
        try:
            return str(json.loads(meta.read_text()).get("dataset_name", dataset_dir.name))
        except json.JSONDecodeError:
            pass
    return dataset_dir.name


def read_manifests(dataset_dirs: list[Path]) -> tuple[list[dict[str, str]], list[str]]:
    """Read manifests from multiple dataset directories."""
    rows: list[dict[str, str]] = []
    missing: list[str] = []
    for dataset_dir in dataset_dirs:
        dataset_rows = read_manifest(dataset_dir)
        if not dataset_rows:
            missing.append(str(dataset_dir))
            continue
        source_name = source_dataset_name(dataset_dir)
        for row in dataset_rows:
            item = dict(row)
            item["_dataset_dir"] = str(dataset_dir)
            item["_source_dataset"] = source_name
            rows.append(item)
    return rows, missing


def resolve_npz_path(row: dict[str, str]) -> Path:
    """Resolve the trial_data.npz path from a manifest row."""
    value = row.get("npz_path", "")
    if value:
        p = Path(value)
        if p.is_absolute():
            return p
        candidate = (Path(row.get("_dataset_dir", ".")) / p).resolve()
        if candidate.exists():
            return candidate
    session_dir = Path(row.get("session_dir", ""))
    if not session_dir.is_absolute():
        session_dir = Path(row.get("_dataset_dir", ".")) / session_dir
    return session_dir.resolve() / "trial_data.npz"


# ---------------------------------------------------------------------------
# Box presence feature extraction
# ---------------------------------------------------------------------------


def box_presence_motion_matrix(
    data: dict[str, np.ndarray],
    feature_params: dict[str, Any],
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Compute the background-subtracted range-profile motion matrix."""
    if "range_profile" not in data or "range_m" not in data:
        return None
    profile = np.asarray(data["range_profile"], dtype=float)
    range_m = np.asarray(data["range_m"], dtype=float)
    if profile.ndim != 2 or range_m.ndim != 1 or profile.shape[0] < 2:
        return None

    cols = min(profile.shape[1], range_m.size)
    profile = profile[:, :cols]
    range_m = range_m[:cols]

    background = None
    if "range_background" in data:
        candidate = np.asarray(data["range_background"], dtype=float)
        if candidate.ndim == 1 and candidate.size >= cols:
            background = candidate[:cols]

    min_range = float(feature_params.get("min_range_m", 0.0))
    max_range = float(feature_params.get("max_range_m", 0.80))
    mask = (range_m >= min_range) & (range_m <= max_range)
    if not np.any(mask):
        return None
    profile = profile[:, mask]
    range_m = range_m[mask]
    if background is not None:
        background = background[mask]

    if bool(feature_params.get("db", True)):
        profile = db_scale(profile)
        if background is not None:
            background = db_scale(background)

    if background is not None:
        motion = profile - background[np.newaxis, :]
    else:
        fallback = max(1, int(feature_params.get("fallback_background_frames", 5)))
        count = min(fallback, profile.shape[0])
        motion = profile - np.median(profile[:count], axis=0)[np.newaxis, :]

    time_s = np.asarray(data.get("time_s", np.linspace(0, profile.shape[0] - 1, profile.shape[0])), dtype=float)
    if time_s.size != profile.shape[0]:
        time_s = np.linspace(0, profile.shape[0] - 1, profile.shape[0])
    return range_m, time_s, motion


def extract_box_presence_feature_vector(
    data: dict[str, np.ndarray],
    feature_params: dict[str, Any],
) -> tuple[Optional[list[float]], list[str]]:
    """Extract a feature vector from a box-presence motion matrix."""
    payload = box_presence_motion_matrix(data, feature_params)
    if payload is None:
        return None, []
    _range_m, _time_s, motion = payload
    if bool(feature_params.get("normalize", False)):
        motion = robust_normalize(motion)
    frames = int(feature_params.get("resample_frames", 24))
    bins = int(feature_params.get("resample_bins", 32))
    image = resample_matrix(motion, frames, bins)
    features = [float(v) for v in image.ravel()]
    names = [f"presence_range_image_t{r:02d}_r{c:02d}" for r in range(frames) for c in range(bins)]
    return features, names


def slice_presence_segment(data: dict, start_index: int, end_index: int) -> dict[str, np.ndarray]:
    """Slice a presence recording segment by frame indices."""
    seg: dict[str, np.ndarray] = {
        "range_m": np.asarray(data["range_m"]),
        "range_profile": np.asarray(data["range_profile"])[start_index:end_index],
    }
    if "range_background" in data:
        seg["range_background"] = np.asarray(data["range_background"])
    if "time_s" in data:
        t = np.asarray(data["time_s"], dtype=float)[start_index:end_index]
        seg["time_s"] = t - t[0] if t.size else t
    if "point_count" in data:
        seg["point_count"] = np.asarray(data["point_count"])[start_index:end_index]
    if "points_xyz" in data:
        seg["points_xyz"] = np.asarray(data["points_xyz"])[start_index:end_index]
    return seg


# ---------------------------------------------------------------------------
# Box content (static) feature extraction
# ---------------------------------------------------------------------------


def range_profile_features(
    data: dict[str, np.ndarray],
    feature_params: dict[str, Any],
) -> tuple[Optional[np.ndarray], list[str]]:
    """Extract a resampled (background-subtracted) mean range profile."""
    if "range_m" not in data:
        return None, []
    range_m = np.asarray(data["range_m"], dtype=float)
    if "mean_range_profile" in data:
        profile = np.asarray(data["mean_range_profile"], dtype=float)
    elif "range_profile" in data:
        profile = np.mean(np.asarray(data["range_profile"], dtype=float), axis=0)
    else:
        return None, []
    cols = min(range_m.size, profile.size)
    range_m = range_m[:cols]
    profile = profile[:cols]
    background = None
    if "range_background" in data:
        candidate = np.asarray(data["range_background"], dtype=float)
        if candidate.ndim == 1 and candidate.size >= cols:
            background = candidate[:cols]
    min_range = float(feature_params.get("min_range_m", 0.0))
    max_range = float(feature_params.get("max_range_m", 0.60))
    mask = (range_m >= min_range) & (range_m <= max_range)
    if not np.any(mask):
        return None, []
    profile = profile[mask]
    if background is not None:
        background = background[mask]
    if bool(feature_params.get("db", True)):
        profile = db_scale(profile)
        if background is not None:
            background = db_scale(background)
    if background is not None:
        profile = profile - background
    bins = int(feature_params.get("range_bins", 64))
    features = resample_vector(profile, bins)
    names = [f"box_delta_range_profile_b{i:02d}" for i in range(bins)]
    return features, names


def point_slice_features(data: dict, feature_params: dict) -> tuple[np.ndarray, list[str]]:
    """Extract point-cloud slice statistics."""
    names = [
        "slice_count_mean", "slice_count_std", "slice_count_max", "slice_count_total",
        "slice_centroid_x", "slice_centroid_y", "slice_spread_x", "slice_spread_y",
        "slice_min_x", "slice_max_x", "slice_min_y", "slice_max_y",
    ]
    count = np.asarray(data.get("slice_point_count", []), dtype=float)
    xyz = np.asarray(data.get("slice_points_xyz", []), dtype=float)
    if count.size == 0:
        return np.zeros(len(names), dtype=float), names
    count_stats = [float(np.mean(count)), float(np.std(count)), float(np.max(count)), float(np.sum(count))]
    points: list[np.ndarray] = []
    if xyz.ndim == 3:
        usable = min(xyz.shape[0], count.size)
        for fi in range(usable):
            pc = max(0, min(int(count[fi]), xyz.shape[1]))
            if pc == 0:
                continue
            fp = xyz[fi, :pc, :]
            fp = fp[np.all(np.isfinite(fp), axis=1)]
            if len(fp):
                points.append(fp)
    if not points:
        return np.array(count_stats + [0.0] * 8, dtype=float), names
    all_points = np.vstack(points)[:, :2]
    centroid = np.mean(all_points, axis=0)
    spread = np.std(all_points, axis=0)
    extras = [
        float(centroid[0]), float(centroid[1]),
        float(spread[0]), float(spread[1]),
        float(np.min(all_points[:, 0])), float(np.max(all_points[:, 0])),
        float(np.min(all_points[:, 1])), float(np.max(all_points[:, 1])),
    ]
    return np.array(count_stats + extras, dtype=float), names


def extract_box_feature_vector(
    data: dict[str, np.ndarray],
    feature_params: dict[str, Any],
) -> tuple[Optional[list[float]], list[str]]:
    """Extract a combined range-profile + point-cloud feature vector."""
    profile, profile_names = range_profile_features(data, feature_params)
    if profile is None:
        return None, []
    features = list(float(v) for v in profile)
    names = list(profile_names)
    if bool(feature_params.get("include_points", True)):
        pf, pn = point_slice_features(data, feature_params)
        features.extend(float(v) for v in pf)
        names.extend(pn)
    return features, names


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------


def estimator_classes(model: Any) -> Optional[list[Any]]:
    classes = getattr(model, "classes_", None)
    if classes is not None:
        return list(classes)
    if hasattr(model, "steps") and model.steps:
        final = model.steps[-1][1]
        classes = getattr(final, "classes_", None)
        if classes is not None:
            return list(classes)
    return None


def prediction_confidence(model: Any, features: list[float], prediction: Any) -> Optional[float]:
    if not hasattr(model, "predict_proba"):
        return None
    probs = model.predict_proba([features])[0]
    classes = estimator_classes(model)
    if classes is None:
        return float(max(probs))
    if prediction not in classes:
        return None
    return float(probs[classes.index(prediction)])


def prediction_scores(model: Any, features: list[float]) -> list[tuple[str, float]]:
    if not hasattr(model, "predict_proba"):
        return []
    probs = model.predict_proba([features])[0]
    classes = estimator_classes(model) or list(range(len(probs)))
    scores = [(str(label), float(score)) for label, score in zip(classes, probs)]
    return sorted(scores, key=lambda x: x[1], reverse=True)


def majority_vote(predictions: list[Any]) -> tuple[Optional[Any], Optional[float], dict[str, int]]:
    if not predictions:
        return None, None, {}
    counts = Counter(predictions)
    max_count = max(counts.values())
    tied = {l for l, c in counts.items() if c == max_count}
    voted = None
    for label in reversed(predictions):
        if label in tied:
            voted = label
            break
    fraction = max_count / len(predictions)
    return voted, fraction, {str(k): int(v) for k, v in counts.items()}
