"""Spatial localization: trilateration + confidence-weighted belief grid.

Combines all sensor modalities to estimate the person's (x, y) position
in the room using a 2D belief grid with temporal decay.

Usage
-----
    localizer = Localizer("config/localization.example.yaml")
    localizer.update(observations)            # feed latest sensor readings
    result = localizer.estimate_position()    # {x, y, confidence, grid}
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from sensors.base import SensorObservation

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class PositionEstimate:
    x: float
    y: float
    confidence: float  # 0-1
    belief_grid: list[list[float]]  # 2D list, row-major
    grid_cells: int
    room_origin: tuple[float, float]
    room_size: tuple[float, float]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class RoomConfig:
    origin_x: float = 0.0
    origin_y: float = 0.0
    size_x: float = 8.0
    size_y: float = 6.0
    grid_cells: int = 20


@dataclass
class AnchorConfig:
    id: str
    x: float
    y: float
    z: float


# ---------------------------------------------------------------------------
# Trilateration
# ---------------------------------------------------------------------------


def trilaterate(
    anchors: list[AnchorConfig],
    distances: list[float],
) -> tuple[float, float, float]:
    """Least-squares 2D trilateration from N >= 2 anchors.

    Returns ``(x, y, residual)``.  Residual is the RMS fit error (metres);
    a high residual indicates inconsistent readings (e.g. NLOS).
    """
    if len(anchors) < 2:
        return (0.0, 0.0, 999.0)
    n = len(anchors)
    # Use anchor 0 as reference, build (n-1) × 2 system
    x0, y0 = anchors[0].x, anchors[0].y
    a_rows: list[list[float]] = []
    b_vals: list[float] = []
    for i in range(1, n):
        xi, yi = anchors[i].x, anchors[i].y
        di = distances[i]
        d0 = distances[0]
        a_rows.append([2.0 * (xi - x0), 2.0 * (yi - y0)])
        b_vals.append(d0 * d0 - di * di + xi * xi + yi * yi - x0 * x0 - y0 * y0)

    a_mat = np.array(a_rows, dtype=np.float64)
    b = np.array(b_vals, dtype=np.float64)

    try:
        # Solve via least-squares
        ata = a_mat.T @ a_mat
        atb = a_mat.T @ b
        # Add small ridge for numerical stability
        ata += np.eye(2) * 1e-8
        p = np.linalg.solve(ata, atb)
        x_est, y_est = float(p[0]), float(p[1])
    except np.linalg.LinAlgError:
        return (0.0, 0.0, 999.0)

    # Compute residual
    residuals = []
    for i in range(n):
        dx = x_est - anchors[i].x
        dy = y_est - anchors[i].y
        residuals.append(abs(math.sqrt(dx * dx + dy * dy) - distances[i]))
    residual = float(np.mean(residuals))

    return (x_est, y_est, residual)


# ---------------------------------------------------------------------------
# Belief grid helpers
# ---------------------------------------------------------------------------


def _gaussian_kernel(sigma: float, size: int = 7) -> np.ndarray:
    """2D Gaussian kernel (size × size)."""
    center = size // 2
    kernel = np.zeros((size, size), dtype=np.float64)
    for i in range(size):
        for j in range(size):
            dx = i - center
            dy = j - center
            kernel[i, j] = math.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    return kernel


def _world_to_grid(wx: float, wy: float, cfg: RoomConfig) -> tuple[int, int]:
    """Convert world coordinates to grid cell indices (clamped)."""
    gx = int((wx - cfg.origin_x) / cfg.size_x * cfg.grid_cells)
    gy = int((wy - cfg.origin_y) / cfg.size_y * cfg.grid_cells)
    gx = max(0, min(cfg.grid_cells - 1, gx))
    gy = max(0, min(cfg.grid_cells - 1, gy))
    return (gx, gy)


def _grid_to_world(gx: int, gy: int, cfg: RoomConfig) -> tuple[float, float]:
    """Grid cell center → world coordinates."""
    wx = cfg.origin_x + (gx + 0.5) / cfg.grid_cells * cfg.size_x
    wy = cfg.origin_y + (gy + 0.5) / cfg.grid_cells * cfg.size_y
    return (wx, wy)


# ---------------------------------------------------------------------------
# Localizer
# ---------------------------------------------------------------------------


class Localizer:
    """Main localisation engine.

    Maintains a 2D belief grid updated from all sensor observations.
    Call ``update(observations)`` whenever new readings arrive, then
    ``estimate_position()`` for the current best estimate + grid.
    """

    def __init__(self, config_path: str = "config/localization.example.yaml") -> None:
        raw = self._load_config(config_path)
        room_raw = raw.get("room", {})
        loc_raw = raw.get("localization", {})

        self._cfg = RoomConfig(
            origin_x=room_raw.get("origin", {}).get("x", 0.0),
            origin_y=room_raw.get("origin", {}).get("y", 0.0),
            size_x=room_raw.get("size", {}).get("x", 8.0),
            size_y=room_raw.get("size", {}).get("y", 6.0),
            grid_cells=room_raw.get("grid_cells", 20),
        )

        # UWB anchors
        self._anchors: dict[str, AnchorConfig] = {}
        for a in raw.get("uwb_anchors", []):
            p = a["position"]
            self._anchors[a["id"]] = AnchorConfig(id=a["id"], x=p["x"], y=p["y"], z=p["z"])

        # RFID reader positions
        self._rfid_reader_pos: dict[str, dict[str, float]] = {}
        for r in raw.get("rfid_readers", []):
            self._rfid_reader_pos[r["id"]] = r["position"]

        # Known RFID tag positions
        self._rfid_tag_pos: dict[str, dict[str, float]] = {}
        for t in raw.get("rfid_tag_positions", []):
            self._rfid_tag_pos[t["tag_id"]] = t["position"]

        # WiFi AP positions
        self._wifi_ap_pos: dict[str, dict[str, float]] = {}
        for w in raw.get("wifi_ap_positions", []):
            self._wifi_ap_pos[w["bssid"]] = w["position"]

        # Localization parameters
        p = loc_raw
        self._temporal_decay = p.get("temporal_decay", 0.92)
        self._uwb_sigma = p.get("uwb_gaussian_sigma", 0.5)
        self._rfid_sigma = p.get("rfid_gaussian_sigma", 0.8)
        self._mmwave_sigma = p.get("mmwave_gaussian_sigma", 1.0)
        self._wifi_sigma = p.get("wifi_gaussian_sigma", 1.5)
        self._imu_motion_threshold = p.get("imu_motion_threshold", 0.5)
        self._position_smoothing = p.get("position_smoothing", 0.7)

        # Precompute Gaussian kernels
        self._uwb_kernel = _gaussian_kernel(self._uwb_sigma)
        self._rfid_kernel = _gaussian_kernel(self._rfid_sigma)
        self._mmwave_kernel = _gaussian_kernel(self._mmwave_sigma)
        self._wifi_kernel = _gaussian_kernel(self._wifi_sigma)

        # Belief grid (row-major: cells[y][x])
        nc = self._cfg.grid_cells
        self._grid: np.ndarray = np.ones((nc, nc), dtype=np.float64) / (nc * nc)
        self._last_imu_accel: Optional[float] = None
        self._smoothed_x: Optional[float] = None
        self._smoothed_y: Optional[float] = None

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_config(path: str) -> dict[str, Any]:
        import yaml

        p = Path(path)
        if not p.exists():
            # Try example path
            p = Path("config/localization.example.yaml")
        if not p.exists():
            return {}
        with open(p) as f:
            return yaml.safe_load(f)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, observations: list[SensorObservation]) -> None:
        """Process a batch of sensor observations and update the belief grid.

        Call this whenever new readings arrive (e.g. from ReaderPool callback).
        Thread-safe if called from only one thread (or externally synchronised).
        """
        if not observations:
            return

        # 1. Temporal decay
        self._grid *= self._temporal_decay

        # 2. UWB trilateration
        uwb_obs = _filter_obs(observations, "uwb")
        if uwb_obs:
            self._update_from_uwb(uwb_obs)

        # 3. RFID
        rfid_obs = _filter_obs(observations, "rfid")
        if rfid_obs:
            self._update_from_rfid(rfid_obs)

        # 4. mmWave
        mmwave_obs = _filter_obs(observations, "mmwave")
        if mmwave_obs:
            self._update_from_mmwave(mmwave_obs)

        # 5. WiFi
        wifi_obs = _filter_obs(observations, "wifi")
        if wifi_obs:
            self._update_from_wifi(wifi_obs)

        # 6. IMU motion detection
        imu_obs = _filter_obs(observations, "imu")
        if imu_obs:
            self._update_from_imu(imu_obs)

        # 7. Normalise
        total = self._grid.sum()
        if total > 0:
            self._grid /= total

    def estimate_position(self) -> PositionEstimate:
        """Return the current best position estimate + full belief grid.

        Returns
        -------
        PositionEstimate
            x, y in world coordinates, confidence (0-1), and the raw grid.
        """
        nc = self._cfg.grid_cells
        # Weighted centroid of the belief grid
        xs = np.arange(nc, dtype=np.float64)
        ys = np.arange(nc, dtype=np.float64)
        gx, gy = np.meshgrid(xs, ys)
        total = self._grid.sum()
        if total <= 0:
            cx, cy = nc // 2, nc // 2
        else:
            cx = float(np.sum(gx * self._grid)) / total
            cy = float(np.sum(gy * self._grid)) / total

        # World coordinates
        wx, wy = _grid_to_world(cx, cy, self._cfg)

        # Confidence = max cell value (or peak probability)
        confidence = float(np.max(self._grid))

        # Apply smoothing — reduce alpha (smooth more) when IMU shows no motion
        alpha = self._position_smoothing
        if self._last_imu_accel is not None and self._last_imu_accel < self._imu_motion_threshold:
            alpha *= 0.3  # heavy smoothing when stationary
        if self._smoothed_x is not None and self._smoothed_y is not None:
            wx = alpha * wx + (1.0 - alpha) * self._smoothed_x
            wy = alpha * wy + (1.0 - alpha) * self._smoothed_y
        self._smoothed_x = wx
        self._smoothed_y = wy

        return PositionEstimate(
            x=wx,
            y=wy,
            confidence=confidence,
            belief_grid=self._grid.tolist(),
            grid_cells=nc,
            room_origin=(self._cfg.origin_x, self._cfg.origin_y),
            room_size=(self._cfg.size_x, self._cfg.size_y),
        )

    def _add_bump(self, wx: float, wy: float, weight: float, kernel: np.ndarray) -> None:
        """Add a Gaussian bump centred on (wx, wy) to the grid."""
        gx, gy = _world_to_grid(wx, wy, self._cfg)
        ks = kernel.shape[0] // 2
        nc = self._cfg.grid_cells
        for ki in range(kernel.shape[0]):
            for kj in range(kernel.shape[1]):
                gi = gy + ki - ks
                gj = gx + kj - ks
                if 0 <= gi < nc and 0 <= gj < nc:
                    self._grid[gi, gj] += weight * kernel[ki, kj]

    # ------------------------------------------------------------------
    # Per-sensor update helpers
    # ------------------------------------------------------------------

    def _update_from_uwb(self, observations: list[SensorObservation]) -> None:
        """Trilaterate from UWB anchor readings."""
        # Group UWB observations by tag_id (we're tracking one person/tag)
        # For each unique tag, collect anchor distances
        tag_readings: dict[str, list[tuple[str, float, float]]] = {}
        for obs in observations:
            tag = obs.tag_id or "person"
            obs_data = obs.observation
            if isinstance(obs_data, dict):
                anchor_id = obs_data.get("anchor_id", obs.sensor_id)
                range_m = obs_data.get("range_m", 0.0)
                conf = obs.confidence
                tag_readings.setdefault(tag, []).append((anchor_id, range_m, conf))

        for tag, readings in tag_readings.items():
            anchors_ordered: list[AnchorConfig] = []
            distances: list[float] = []
            weights: list[float] = []
            for anchor_id, dist, conf in readings:
                anc = self._anchors.get(anchor_id)
                if anc is not None:
                    anchors_ordered.append(anc)
                    distances.append(dist)
                    weights.append(conf)

            if len(anchors_ordered) >= 2:
                x, y, residual = trilaterate(anchors_ordered, distances)
                # Weight bump by inverse of residual (better fit = higher weight)
                bump_weight = math.exp(-residual / 2.0) * np.mean(weights)
                self._add_bump(x, y, bump_weight, self._uwb_kernel)

    def _update_from_rfid(self, observations: list[SensorObservation]) -> None:
        """Bias belief toward known RFID reader or tag positions."""
        for obs in observations:
            tag_id = obs.tag_id
            if tag_id and tag_id in self._rfid_tag_pos:
                pos = self._rfid_tag_pos[tag_id]
                self._add_bump(pos["x"], pos["y"], obs.confidence, self._rfid_kernel)
            elif obs.sensor_id in self._rfid_reader_pos:
                pos = self._rfid_reader_pos[obs.sensor_id]
                self._add_bump(pos["x"], pos["y"], obs.confidence * 0.5, self._rfid_kernel)

    def _update_from_mmwave(self, observations: list[SensorObservation]) -> None:
        """Use radar point cloud as soft zone vote.

        Each detected object contributes a small Gaussian bump at its
        sensor-relative (x, y) position.  The radar has a known position
        (assumed at room center for now; can be configured).
        """
        radar_pos = {"x": self._cfg.size_x / 2, "y": self._cfg.size_y / 2}
        for obs in observations:
            obs_data = obs.observation
            if not isinstance(obs_data, dict):
                continue
            objects = obs_data.get("objects", [])
            for obj in objects:
                if isinstance(obj, dict):
                    wx = radar_pos["x"] + obj.get("x", 0.0)
                    wy = radar_pos["y"] + obj.get("y", 0.0)
                    weight = obs.confidence / max(len(objects), 1)
                    self._add_bump(wx, wy, weight, self._mmwave_kernel)

    def _update_from_wifi(self, observations: list[SensorObservation]) -> None:
        """RSSI-weighted proximity to known AP positions.

        Higher RSSI = closer to that AP → bump grid around AP position.
        """
        for obs in observations:
            obs_data = obs.observation
            if not isinstance(obs_data, dict):
                continue
            bssid = obs.tag_id
            if bssid and bssid in self._wifi_ap_pos:
                ap_pos = self._wifi_ap_pos[bssid]
                rssi = obs_data.get("rssi", -90.0)
                # Convert RSSI to a weight: -30 dBm ≈ 1.0, -90 dBm ≈ 0.1
                weight = max(0.1, min(1.0, (-rssi - 30.0) / 60.0))
                self._add_bump(
                    ap_pos["x"], ap_pos["y"], weight * obs.confidence, self._wifi_kernel,
                )

    def _update_from_imu(self, observations: list[SensorObservation]) -> None:
        """Detect motion vs stationary state from IMU acceleration magnitude."""
        accel_mags = []
        for obs in observations:
            obs_data = obs.observation
            if not isinstance(obs_data, dict):
                continue
            ax = obs_data.get("accel_x", 0.0)
            ay = obs_data.get("accel_y", 0.0)
            az = obs_data.get("accel_z", 0.0)
            mag = math.sqrt(ax * ax + ay * ay + az * az)
            accel_mags.append(mag)

        if accel_mags:
            avg_accel = np.mean(accel_mags)
            self._last_imu_accel = avg_accel
            # If motion is below threshold, smoothing will be more aggressive
            # (handled in estimate_position via _position_smoothing)


def _filter_obs(
    observations: list[SensorObservation], sensor_type: str,
) -> list[SensorObservation]:
    return [o for o in observations if o.sensor_type == sensor_type]


__all__ = ["Localizer", "PositionEstimate", "trilaterate", "RoomConfig", "AnchorConfig"]
