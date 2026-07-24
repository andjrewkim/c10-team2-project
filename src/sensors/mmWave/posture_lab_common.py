"""Feature extraction for posture classification from mmWave point clouds.

Ported from the COSMOS ``posture_lab_common.py`` lab module.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.sensors.mmWave.lab_mmwave import fill_nan_series, resample_vector

INPUT_TYPE = "mmwave_posture_point_cloud_2d"
STAT_NAMES = ("mean", "std", "min", "max", "p10", "p50", "p90", "start", "end", "delta", "slope")


def filter_posture_points(xyz: np.ndarray, feature_params: dict[str, Any]) -> np.ndarray:
    """Filter point cloud points to a front-facing ROI for posture analysis."""
    xyz = np.asarray(xyz, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.size == 0:
        return np.empty((0, 3), dtype=float)
    finite = np.all(np.isfinite(xyz), axis=1)
    xyz = xyz[finite]
    if xyz.size == 0:
        return np.empty((0, 3), dtype=float)
    x_limit = float(feature_params.get("x_limit_m", 2.0))
    min_range = float(feature_params.get("min_range_m", 0.2))
    max_range = float(feature_params.get("max_range_m", 5.0))
    mask = (np.abs(xyz[:, 0]) <= x_limit) & (xyz[:, 1] >= min_range) & (xyz[:, 1] <= max_range)
    return xyz[mask]


def series_stats(values: np.ndarray, time_s: np.ndarray) -> list[float]:
    """Compute summary statistics for a time series."""
    values = fill_nan_series(np.asarray(values, dtype=float))
    if values.size == 0:
        return [0.0] * len(STAT_NAMES)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        values = np.zeros_like(values, dtype=float)
        finite = values
    if time_s.size != values.size:
        time_s = np.linspace(0, values.size - 1, values.size)
    duration = max(float(time_s[-1] - time_s[0]), 1e-9) if values.size > 1 else 1.0
    slope = float((values[-1] - values[0]) / duration)
    return [
        float(np.mean(values)), float(np.std(values)), float(np.min(values)), float(np.max(values)),
        float(np.percentile(finite, 10)), float(np.percentile(finite, 50)), float(np.percentile(finite, 90)),
        float(values[0]), float(values[-1]), float(values[-1] - values[0]), slope,
    ]


def append_series_features(features: list[float], names: list[str], prefix: str, values: np.ndarray, time_s: np.ndarray, trajectory_points: int = 0) -> None:
    """Extract summary stats + optional trajectory for a time series."""
    features.extend(series_stats(values, time_s))
    names.extend(f"{prefix}_{stat}" for stat in STAT_NAMES)
    if trajectory_points > 0:
        trajectory = resample_vector(values, trajectory_points)
        features.extend(float(v) for v in trajectory)
        names.extend(f"{prefix}_t{i:02d}" for i in range(trajectory_points))


def normalized_histogram(values: np.ndarray, bins: int, value_range: tuple[float, float]) -> np.ndarray:
    """Compute a normalized 1D histogram."""
    if values.size == 0:
        return np.zeros(bins, dtype=float)
    hist, _ = np.histogram(values, bins=bins, range=value_range)
    total = float(np.sum(hist))
    return hist.astype(float) / total if total > 0 else hist.astype(float)


def normalized_histogram2d(x_values: np.ndarray, y_values: np.ndarray, x_bins: int, y_bins: int, x_range: tuple[float, float], y_range: tuple[float, float]) -> np.ndarray:
    """Compute a normalized 2D histogram."""
    if x_values.size == 0 or y_values.size == 0:
        return np.zeros((x_bins, y_bins), dtype=float)
    hist, _, _ = np.histogram2d(x_values, y_values, bins=(x_bins, y_bins), range=(x_range, y_range))
    total = float(np.sum(hist))
    return hist.astype(float) / total if total > 0 else hist.astype(float)


def extract_posture_feature_vector(data: dict[str, np.ndarray], feature_params: dict[str, Any]) -> tuple[Optional[list[float]], list[str]]:
    """Extract a feature vector from a posture point-cloud recording segment."""
    if "points_xyz" not in data:
        return None, []
    points_xyz = np.asarray(data["points_xyz"], dtype=float)
    if points_xyz.ndim != 3 or points_xyz.shape[2] != 3 or points_xyz.shape[0] < 2:
        return None, []

    frame_count = points_xyz.shape[0]
    time_s = np.asarray(data.get("time_s", np.linspace(0, frame_count - 1, frame_count)), dtype=float)
    if time_s.size != frame_count:
        time_s = np.linspace(0, frame_count - 1, frame_count)

    counts = np.zeros(frame_count, dtype=float)
    centroid = np.full((frame_count, 2), np.nan, dtype=float)
    spread = np.full((frame_count, 2), np.nan, dtype=float)
    minimum = np.full((frame_count, 2), np.nan, dtype=float)
    maximum = np.full((frame_count, 2), np.nan, dtype=float)
    all_points: list[np.ndarray] = []

    for fi in range(frame_count):
        fp = filter_posture_points(points_xyz[fi], feature_params)
        counts[fi] = len(fp)
        if len(fp) == 0:
            continue
        xy = fp[:, :2]
        all_points.append(xy)
        centroid[fi] = np.mean(xy, axis=0)
        spread[fi] = np.std(xy, axis=0)
        minimum[fi] = np.min(xy, axis=0)
        maximum[fi] = np.max(xy, axis=0)

    width = maximum[:, 0] - minimum[:, 0]
    depth = maximum[:, 1] - minimum[:, 1]
    active_fraction = float(np.mean(counts > 0.0))
    trajectory_points = int(feature_params.get("trajectory_points", 20))

    features: list[float] = [active_fraction]
    names: list[str] = ["active_frame_fraction"]

    append_series_features(features, names, "point_count", counts, time_s, trajectory_points)
    for ai, an in enumerate(("x", "y")):
        append_series_features(features, names, f"centroid_{an}", centroid[:, ai], time_s, trajectory_points)
        append_series_features(features, names, f"spread_{an}", spread[:, ai], time_s)
        append_series_features(features, names, f"min_{an}", minimum[:, ai], time_s)
        append_series_features(features, names, f"max_{an}", maximum[:, ai], time_s)
    append_series_features(features, names, "body_width_x", width, time_s)
    append_series_features(features, names, "body_depth_y", depth, time_s)

    stacked = np.vstack(all_points) if all_points else np.empty((0, 2), dtype=float)

    x_limit = float(feature_params.get("x_limit_m", 2.0))
    min_range = float(feature_params.get("min_range_m", 0.2))
    max_range = float(feature_params.get("max_range_m", 5.0))
    x_bins = int(feature_params.get("x_bins", 8))
    y_bins = int(feature_params.get("y_bins", 12))
    xy_x_bins = int(feature_params.get("xy_x_bins", 8))
    xy_y_bins = int(feature_params.get("xy_y_bins", 12))

    for prefix, vals, bins, vrange in [
        ("x_hist", stacked[:, 0] if len(stacked) else np.empty(0), x_bins, (-x_limit, x_limit)),
        ("y_hist", stacked[:, 1] if len(stacked) else np.empty(0), y_bins, (min_range, max_range)),
    ]:
        hist = normalized_histogram(vals, bins, vrange)
        features.extend(float(v) for v in hist)
        names.extend(f"{prefix}_b{i:02d}" for i in range(bins))

    xy_hist = normalized_histogram2d(
        stacked[:, 0] if len(stacked) else np.empty(0),
        stacked[:, 1] if len(stacked) else np.empty(0),
        xy_x_bins, xy_y_bins, (-x_limit, x_limit), (min_range, max_range),
    )
    features.extend(float(v) for v in xy_hist.ravel())
    names.extend(f"xy_occupancy_x{xi:02d}_y{yi:02d}" for xi in range(xy_x_bins) for yi in range(xy_y_bins))

    if not all(math.isfinite(v) for v in features):
        features = [0.0 if not math.isfinite(v) else v for v in features]
    return features, names


def slice_posture_segment(data: dict[str, np.ndarray], start_index: int, end_index: int) -> dict[str, np.ndarray]:
    """Slice a posture recording segment by frame indices."""
    seg: dict[str, np.ndarray] = {"points_xyz": np.asarray(data["points_xyz"])[start_index:end_index]}
    for key in ("point_count", "points_velocity"):
        if key in data:
            seg[key] = np.asarray(data[key])[start_index:end_index]
    if "time_s" in data:
        t = np.asarray(data["time_s"], dtype=float)[start_index:end_index]
        seg["time_s"] = t - t[0] if t.size else t
    return seg


def pack_point_cloud_frames(xyz_frames: list[np.ndarray], velocity_frames: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack per-frame point arrays into NaN-padded matrices."""
    frame_count = len(xyz_frames)
    point_count = np.array([len(f) for f in xyz_frames], dtype=np.uint16)
    max_points = int(np.max(point_count)) if len(point_count) else 0
    points_xyz = np.full((frame_count, max_points, 3), np.nan, dtype=float)
    points_velocity = np.full((frame_count, max_points), np.nan, dtype=float)
    for i, xyz in enumerate(xyz_frames):
        count = len(xyz)
        if count == 0:
            continue
        points_xyz[i, :count, :] = xyz
        vel = velocity_frames[i]
        points_velocity[i, : min(count, len(vel))] = vel[:count]
    return point_count, points_xyz, points_velocity
