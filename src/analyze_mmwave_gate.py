"""
Analyze mmWave feature distributions per gesture from all labeled sessions.
Computes empirical thresholds for the gating layer at configurable percentiles.

Outputs:
  1. Per-gesture feature statistics (mean, std, percentiles)
  2. Recommended veto thresholds at P5, P10
  3. Pairwise confusion analysis for double-confirmation candidates
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# Reuse the realtime demo's feature extraction
from src.realtime_demo import (
    _compute_features,
    extract_features_from_reading,
    _movement_score,
)
from src.sensors.base_reader import Reading


def load_mmwave_from_csv(csv_path: Path) -> list[Reading]:
    readings = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pts_str = row.get("points", "[]")
            try:
                points = json.loads(pts_str)
            except (json.JSONDecodeError, TypeError):
                points = []
            rp_str = row.get("range_profile", "null")
            try:
                rp = json.loads(rp_str) if rp_str and rp_str != "null" else None
            except (json.JSONDecodeError, TypeError):
                rp = None
            reading = Reading(
                sensor_id="mmwave-0",
                sensor_type="mmwave",
                data={
                    "points": points,
                    "num_points": len(points),
                    "range_profile": rp,
                    "motion_score": float(row.get("motion_score", 0)),
                },
                confidence=float(row.get("confidence", 0.8)),
            )
            readings.append(reading)
    return readings


def load_events(events_csv: Path) -> list[dict]:
    events = []
    with open(events_csv) as f:
        for row in csv.DictReader(f):
            events.append(row)
    return events


def compute_window_features(
    mm_readings: list[Reading],
    events: list[dict],
    window_size: int = 10,
    stride: int = 5,
) -> dict[str, list[np.ndarray]]:
    """Group by gesture, extract window-level mmWave features."""
    per_gesture: dict[str, list[list[float]]] = defaultdict(list)
    raw_feats_by_gesture: dict[str, list[np.ndarray]] = defaultdict(list)

    for i in range(0, len(events) - window_size + 1, stride):
        window_events = events[i : i + window_size]
        gesture = window_events[0].get("gesture", "unknown")
        if gesture == "none" or not gesture:
            continue

        # Verify all frames in window have same gesture
        win_gestures = {e.get("gesture") for e in window_events}
        if len(win_gestures) > 1:
            continue  # skip boundary windows

        window_mm = mm_readings[i : i + window_size]
        if len(window_mm) < window_size:
            continue

        feats = _compute_features(window_mm, "mmwave")
        per_gesture[gesture].append(feats)
        raw_feats_by_gesture[gesture].append(
            np.array([extract_features_from_reading(r, "mmwave") for r in window_mm])
        )

    return per_gesture, raw_feats_by_gesture


MM_FEATURE_LABELS = [
    "num_points",
    "mean_x", "std_x",
    "mean_y", "std_y",
    "distance_from_origin",
]

WINDOW_FEATURE_LABELS = (
    [f"mean_{lbl}" for lbl in MM_FEATURE_LABELS]
    + [f"std_{lbl}" for lbl in MM_FEATURE_LABELS]
    + ["path_length"]
)


def compute_statistics(
    per_gesture: dict[str, list[list[float]]], percentiles: list[float] = [1, 5, 10, 25, 50, 75, 90, 95, 99]
) -> dict:
    stats = {}
    for gesture, feat_list in sorted(per_gesture.items()):
        arr = np.array(feat_list)
        stats[gesture] = {
            "n_windows": len(feat_list),
            "n_features": arr.shape[1] if arr.ndim > 1 else 0,
        }
        for fi in range(arr.shape[1]):
            col = arr[:, fi]
            label = WINDOW_FEATURE_LABELS[fi] if fi < len(WINDOW_FEATURE_LABELS) else f"feat_{fi}"
            pvals = np.percentile(col, percentiles)
            stats[gesture][label] = {
                "mean": float(np.mean(col)),
                "std": float(np.std(col)),
                "min": float(np.min(col)),
                "max": float(np.max(col)),
                "pct": {int(p): float(v) for p, v in zip(percentiles, pvals)},
            }
    return stats


def suggest_veto_thresholds(stats: dict, percentile: int = 5) -> dict:
    """Suggest veto thresholds per gesture based on key physical features."""
    thresholds = {}
    for gesture, gstats in sorted(stats.items()):
        t = {}
        # path_length — needs some minimum movement
        if "path_length" in gstats:
            t["min_path_length"] = gstats["path_length"]["pct"].get(percentile, 0.0)

        # mean_distance_from_origin — how far hand is from body
        if "mean_distance_from_origin" in gstats:
            t["min_mean_distance"] = gstats["mean_distance_from_origin"]["pct"].get(percentile, 0.0)

        # mean_num_points — minimum point cloud presence
        if "mean_num_points" in gstats:
            t["min_mean_num_points"] = gstats["mean_num_points"]["pct"].get(percentile, 0.0)

        # std_distance_from_origin — how much hand moves in range (push/pull need this)
        if "std_distance_from_origin" in gstats:
            t["min_std_distance"] = gstats["std_distance_from_origin"]["pct"].get(percentile, 0.0)

        # mean_y — vertical position (raise-arms, t-arm need high y)
        if "mean_mean_y" in gstats:
            t["min_mean_y"] = gstats["mean_mean_y"]["pct"].get(percentile, 0.0)
            t["max_mean_y"] = gstats["mean_mean_y"]["pct"].get(100 - percentile, float("inf"))

        # mean_x — lateral position (left/right need lateral displacement)
        if "mean_mean_x" in gstats:
            t["min_mean_x_abs"] = abs(gstats["mean_mean_x"]["pct"].get(percentile, 0.0))

        thresholds[gesture] = t
    return thresholds


def main():
    window = 10
    stride = 5

    # Discover all candidate sessions
    session_sources = []

    # multi_raw sessions (all have both mmwave + imu + events)
    multi_dir = Path("archive/multi_raw")
    if multi_dir.exists():
        for d in sorted(multi_dir.glob("session_*")):
            if (d / "mmwave.csv").exists() and (d / "events.csv").exists():
                session_sources.append(("multi_raw", d))

    # raw sessions that have BOTH mmwave + imu + events
    raw_dir = Path("archive/raw")
    if raw_dir.exists():
        for d in sorted(raw_dir.glob("session_*")):
            if (d / "mmwave.csv").exists() and (d / "events.csv").exists():
                session_sources.append(("raw", d))

    print(f"Found {len(session_sources)} sessions with mmWave data\n")

    all_per_gesture: dict[str, list[list[float]]] = defaultdict(list)

    for source_type, session_dir in session_sources:
        name = session_dir.name
        events = load_events(session_dir / "events.csv")
        mm_readings = load_mmwave_from_csv(session_dir / "mmwave.csv")

        # Trim to matching lengths
        n = min(len(events), len(mm_readings))
        events = events[:n]
        mm_readings = mm_readings[:n]

        pg, _ = compute_window_features(mm_readings, events, window, stride)
        n_wins = sum(len(v) for v in pg.values())
        gestures_found = list(pg.keys())
        print(f"  {name} ({source_type}): {n_wins} windows, gestures={gestures_found}")

        for gesture, feat_list in pg.items():
            all_per_gesture[gesture].extend(feat_list)

    print(f"\n=== AGGREGATE STATISTICS (window={window}, stride={stride}) ===\n")

    stats = compute_statistics(all_per_gesture)

    for gesture, gstats in sorted(stats.items()):
        print(f"\n{'='*60}")
        print(f"  GESTURE: {gesture}")
        print(f"  Windows: {gstats['n_windows']}")
        print(f"{'='*60}")
        for feat_name in WINDOW_FEATURE_LABELS:
            if feat_name in gstats:
                fs = gstats[feat_name]
                print(f"    {feat_name:>35s}: mean={fs['mean']:8.4f}  std={fs['std']:8.4f}  "
                      f"p5={fs['pct'][5]:8.4f}  p10={fs['pct'][10]:8.4f}  "
                      f"p50={fs['pct'][50]:8.4f}  p90={fs['pct'][90]:8.4f}  "
                      f"p95={fs['pct'][95]:8.4f}")

    print(f"\n\n=== SUGGESTED VETO THRESHOLDS (P5) ===\n")
    thresholds_p5 = suggest_veto_thresholds(stats, percentile=5)
    thresholds_p10 = suggest_veto_thresholds(stats, percentile=10)

    for gesture in sorted(thresholds_p5.keys()):
        print(f"\n  {gesture}:")
        print(f"    P5 thresholds:  {thresholds_p5[gesture]}")
        print(f"    P10 thresholds: {thresholds_p10[gesture]}")
        print(f"    Windows: {stats[gesture]['n_windows']}")

    # Also compute per-gesture mean feature vectors for reference
    print(f"\n\n=== MEAN FEATURE VECTORS (for reference) ===\n")
    all_gestures_sorted = sorted(all_per_gesture.keys())
    n_feats = len(WINDOW_FEATURE_LABELS)
    header = f"{'gesture':>20s}" + "".join(f"{lbl:>14s}" for lbl in WINDOW_FEATURE_LABELS[:8])
    print(header)
    print("-" * len(header))
    for gesture in all_gestures_sorted:
        arr = np.array(all_per_gesture[gesture])
        if arr.ndim > 1 and arr.shape[1] >= 8:
            means = np.mean(arr, axis=0)
            row = f"{gesture:>20s}"
            for i in range(min(8, len(means))):
                row += f"{means[i]:>14.4f}"
            print(row)

    print(f"\n\nDone. Total windows analyzed: {sum(len(v) for v in all_per_gesture.values())}")


if __name__ == "__main__":
    main()
