"""UWB (Ultra-Wideband) ranging utilities extracted from
``mmwave/labs/lab09-uwb-lab/UWB_lab/``.

Provides ranging-log parsing, distance filtering (MAD-based outlier
rejection), statistical feature extraction for ML, and summary statistics.
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

SEQUENCE_PATTERN = re.compile(r"sequence n:\s*(\d+)")
INTERVAL_PATTERN = re.compile(r"ranging interval:\s*([0-9.]+)\s*ms")
STATUS_PATTERN = re.compile(r"status:\s*([A-Za-z0-9_]+)\s*\((0x[0-9a-fA-F]+)\)")
DISTANCE_PATTERN = re.compile(r"distance:\s*([-+]?[0-9]*\.?[0-9]+)\s*cm")


class RangeLogParser:
    """Streaming parser for UWB ranging terminal logs.

    Call ``feed()`` for each line; it returns a sample dict (or *None*)
    when a complete ``(sequence, status, distance)`` triplet has been
    assembled.
    """

    def __init__(self) -> None:
        self.sequence: Optional[int] = None
        self.interval_ms: Optional[float] = None
        self.status: Optional[str] = None
        self.status_code: Optional[str] = None

    def feed(self, line: str) -> Optional[dict[str, Any]]:
        m = SEQUENCE_PATTERN.search(line)
        if m:
            self.sequence = int(m.group(1))
            self.status = None
            self.status_code = None
            return None
        m = INTERVAL_PATTERN.search(line)
        if m:
            self.interval_ms = float(m.group(1))
            return None
        m = STATUS_PATTERN.search(line)
        if m:
            self.status = m.group(1)
            self.status_code = m.group(2)
            return None
        m = DISTANCE_PATTERN.search(line)
        if m:
            sample: dict[str, Any] = {
                "time_s": None,
                "sequence": self.sequence,
                "interval_ms": self.interval_ms,
                "status": self.status or "unknown",
                "status_code": self.status_code or "",
                "distance_cm": float(m.group(1)),
            }
            self.status = None
            self.status_code = None
            return sample
        return None


def parse_ranging_log(path: Path) -> list[dict[str, Any]]:
    """Parse an entire UWB ranging terminal log into a list of sample dicts."""
    parser = RangeLogParser()
    samples: list[dict[str, Any]] = []
    if not path.exists():
        return samples
    with open(path, errors="replace") as f:
        for line in f:
            sample = parser.feed(line)
            if sample:
                samples.append(sample)
    return samples


def read_ranging_csv(path: Path) -> list[dict[str, Any]]:
    """Read a ranging CSV file (written by ``write_ranging_csv``)."""
    samples: list[dict[str, Any]] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            samples.append(
                {
                    "time_s": float(row["time_s"]) if row.get("time_s") else None,
                    "sequence": int(row["sequence"]) if row.get("sequence") else None,
                    "interval_ms": float(row["interval_ms"]) if row.get("interval_ms") else None,
                    "status": row.get("status", ""),
                    "status_code": row.get("status_code", ""),
                    "distance_cm": float(row["distance_cm"]),
                }
            )
    return samples


def write_ranging_csv(samples: list[dict], path: Path) -> None:
    """Write ranging samples to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["time_s", "sequence", "interval_ms", "status", "status_code", "distance_cm"],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(samples)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


def summarize_ranging(samples: list[dict]) -> dict[str, Any]:
    """Compute aggregate statistics over a list of ranging samples."""
    ok = [
        s["distance_cm"]
        for s in samples
        if s.get("status") == "Ok" and 0.0 < float(s.get("distance_cm", 0.0)) < 60000.0
    ]
    if not ok:
        return {"total_samples": len(samples), "ok_samples": 0, "mean_cm": None, "median_cm": None}
    ordered = sorted(ok)
    n = len(ordered)
    median = ordered[n // 2] if n % 2 else 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])
    return {
        "total_samples": len(samples),
        "ok_samples": len(ok),
        "mean_cm": sum(ok) / len(ok),
        "median_cm": median,
        "min_cm": min(ok),
        "max_cm": max(ok),
    }


# ---------------------------------------------------------------------------
# Distance outlier filtering
# ---------------------------------------------------------------------------


def filter_distances_cm(
    values: list[float],
    max_distance_cm: float = 10000.0,
    mad_z: float = 4.0,
) -> list[float]:
    """Remove implausible and outlying distance measurements.

    Applies:
    1. Finite / positive / max-distance gate.
    2. Median absolute deviation (MAD) outlier rejection (optional).
    """
    clean = [v for v in values if math.isfinite(v) and 0.0 < v < max_distance_cm]
    if len(clean) < 4 or mad_z <= 0:
        return clean

    ordered = sorted(clean)
    n = len(ordered)
    median = ordered[n // 2] if n % 2 else 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])
    deviations = sorted(abs(v - median) for v in clean)
    mad = deviations[n // 2] if n % 2 else 0.5 * (deviations[n // 2 - 1] + deviations[n // 2])
    if mad > 0:
        limit = mad_z * 1.4826 * mad
        return [v for v in clean if abs(v - median) <= limit]
    return clean


# ---------------------------------------------------------------------------
# Feature extraction (for ML)
# ---------------------------------------------------------------------------

RANGE_FEATURE_NAMES = [
    "count",
    "mean_cm",
    "std_cm",
    "min_cm",
    "max_cm",
    "median_cm",
    "range_cm",
    "q25_cm",
    "q75_cm",
    "iqr_cm",
    "first_cm",
    "last_cm",
    "delta_cm",
    "abs_delta_cm",
    "mean_abs_step_cm",
    "max_abs_step_cm",
    "slope_cm_per_sample",
]


def extract_range_features(samples: list[dict], resample_points: int = 20) -> Optional[list[float]]:
    """Extract a feature vector from a list of range samples.

    Returns *None* if there are fewer than 2 valid samples.
    """
    values = [
        float(s["distance_cm"])
        for s in samples
        if s.get("status") == "Ok" and 0.0 < float(s.get("distance_cm", 0.0)) < 60000.0
    ]
    if len(values) < 2:
        return None

    arr = np.asarray(values, dtype=float)
    diffs = np.diff(arr)
    q25, q75 = np.percentile(arr, [25, 75])
    x = np.arange(len(arr), dtype=float)
    slope = float(np.polyfit(x, arr, deg=1)[0])

    features = [
        float(len(arr)),
        float(arr.mean()),
        float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        float(arr.min()),
        float(arr.max()),
        float(np.median(arr)),
        float(arr.max() - arr.min()),
        float(q25),
        float(q75),
        float(q75 - q25),
        float(arr[0]),
        float(arr[-1]),
        float(arr[-1] - arr[0]),
        float(abs(arr[-1] - arr[0])),
        float(np.mean(np.abs(diffs))) if len(diffs) > 0 else 0.0,
        float(np.max(np.abs(diffs))) if len(diffs) > 0 else 0.0,
        slope,
    ]

    if resample_points > 0:
        src_x = np.arange(len(arr), dtype=float)
        dst_x = np.linspace(0.0, float(len(arr) - 1), resample_points)
        shape = np.interp(dst_x, src_x, arr)
        shape = shape - shape[0]
        features.extend(float(v) for v in shape)

    return features


__all__ = [
    "RangeLogParser",
    "parse_ranging_log",
    "read_ranging_csv",
    "write_ranging_csv",
    "summarize_ranging",
    "filter_distances_cm",
    "RANGE_FEATURE_NAMES",
    "extract_range_features",
]
