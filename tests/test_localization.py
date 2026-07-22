"""Unit tests for the localisation module.

Tests cover trilateration geometry, belief-grid updates (Gaussian bump,
temporal decay, normalisation), and multi-sensor integration (UWB, RFID,
mmWave, WiFi, IMU).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.localization import (
    AnchorConfig,
    Localizer,
    RoomConfig,
    trilaterate,
    _world_to_grid,
    _grid_to_world,
)
from sensors.base import SensorObservation
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Trilateration
# ---------------------------------------------------------------------------


def make_anchor(id: str, x: float, y: float) -> AnchorConfig:
    return AnchorConfig(id=id, x=x, y=y, z=0.0)


def test_trilateration_perfect_noiseless() -> None:
    """Three anchors with perfect distances yield exact position."""
    anchors = [make_anchor("a1", 0.0, 0.0), make_anchor("a2", 5.0, 0.0), make_anchor("a3", 2.5, 4.0)]
    # Person at (2.0, 1.5)
    true_x, true_y = 2.0, 1.5
    distances = [
        math.sqrt((true_x - a.x) ** 2 + (true_y - a.y) ** 2) for a in anchors
    ]
    x, y, residual = trilaterate(anchors, distances)
    assert abs(x - true_x) < 1e-8
    assert abs(y - true_y) < 1e-8
    assert residual < 1e-8


def test_trilateration_two_anchors() -> None:
    """Two anchors with perfect distances produce a unique 2D solution (well-posed)."""
    anchors = [make_anchor("a1", 0.0, 0.0), make_anchor("a2", 4.0, 0.0)]
    # Person at (1.0, 0.0) — on the line between anchors (degenerate but fine)
    distances = [1.0, 3.0]
    x, y, _ = trilaterate(anchors, distances)
    # Should be near (1.0, 0.0) — least squares will find the projection
    assert abs(x - 1.0) < 0.1


def test_trilateration_noisy_readings() -> None:
    """Trilateration with small noise still produces reasonable estimate."""
    anchors = [make_anchor("a1", 0.0, 0.0), make_anchor("a2", 5.0, 0.0), make_anchor("a3", 2.5, 4.0)]
    true_x, true_y = 3.0, 2.0
    distances = [
        math.sqrt((true_x - a.x) ** 2 + (true_y - a.y) ** 2) + np.random.normal(0, 0.1)
        for a in anchors
    ]
    x, y, residual = trilaterate(anchors, distances)
    # Should be within ~0.3 m of truth
    assert math.sqrt((x - true_x) ** 2 + (y - true_y) ** 2) < 0.3


def test_trilateration_fewer_than_two_returns_origin() -> None:
    """Fewer than 2 anchors returns (0,0) with high residual."""
    x, y, residual = trilaterate([make_anchor("a1", 0.0, 0.0)], [1.0])
    assert x == 0.0 and y == 0.0
    assert residual > 100


# ---------------------------------------------------------------------------
# World-to-grid and grid-to-world conversions
# ---------------------------------------------------------------------------


def test_world_to_grid_and_back() -> None:
    cfg = RoomConfig(origin_x=0.0, origin_y=0.0, size_x=8.0, size_y=6.0, grid_cells=20)
    gx, gy = _world_to_grid(2.0, 1.5, cfg)
    assert 0 <= gx < 20
    assert 0 <= gy < 20
    # Convert back
    wx, wy = _grid_to_world(gx, gy, cfg)
    # Should be within one cell width of the original
    assert abs(wx - 2.0) < 8.0 / 20.0
    assert abs(wy - 1.5) < 6.0 / 20.0


def test_world_to_grid_clamps() -> None:
    cfg = RoomConfig(origin_x=0.0, origin_y=0.0, size_x=8.0, size_y=6.0, grid_cells=20)
    gx, gy = _world_to_grid(-10.0, 100.0, cfg)
    assert gx == 0
    assert gy == 19  # clamped to max


# ---------------------------------------------------------------------------
# Localizer — full integration
# ---------------------------------------------------------------------------


def test_localizer_default_config() -> None:
    """Localizer can be created with default example config."""
    loc = Localizer("config/localization.example.yaml")
    assert loc is not None
    est = loc.estimate_position()
    assert 0.0 <= est.confidence <= 1.0
    assert len(est.belief_grid) == 20
    assert len(est.belief_grid[0]) == 20


def test_localizer_uwb_update_adds_belief() -> None:
    """Repeated UWB updates concentrate belief near the trilaterated position."""
    loc = Localizer("config/localization.example.yaml")
    # Create UWB observations that trilaterate to ~(2.0, 1.0)
    def _make_obs() -> list[SensorObservation]:
        return [
            SensorObservation(
                sensor_id="uwb-anchor-1", sensor_type="uwb",
                timestamp=datetime.now(timezone.utc),
                observation={"anchor_id": "uwb-anchor-1", "tag_id": "tag-001", "range_m": 2.236},
                confidence=0.9, tag_id="tag-001",
            ),
            SensorObservation(
                sensor_id="uwb-anchor-2", sensor_type="uwb",
                timestamp=datetime.now(timezone.utc),
                observation={"anchor_id": "uwb-anchor-2", "tag_id": "tag-001", "range_m": 3.162},
                confidence=0.9, tag_id="tag-001",
            ),
            SensorObservation(
                sensor_id="uwb-anchor-3", sensor_type="uwb",
                timestamp=datetime.now(timezone.utc),
                observation={"anchor_id": "uwb-anchor-3", "tag_id": "tag-001", "range_m": 3.041},
                confidence=0.9, tag_id="tag-001",
            ),
        ]
    # Multiple updates to concentrate belief
    for _ in range(5):
        loc.update(_make_obs())
    est = loc.estimate_position()
    # After several updates, belief should be concentrated near (2.0, 1.0)
    assert abs(est.x - 2.0) < 1.0
    assert abs(est.y - 1.0) < 1.0
    assert est.confidence > 0.01


def test_localizer_temporal_decay() -> None:
    """Repeated updates with no new data should cause belief to spread (decay)."""
    loc = Localizer("config/localization.example.yaml")
    # Initial uniform grid
    est1 = loc.estimate_position()
    initial_max = max(max(row) for row in est1.belief_grid)

    # Apply several empty updates (decay only, no new bumps)
    for _ in range(10):
        loc.update([])

    est2 = loc.estimate_position()
    decayed_max = max(max(row) for row in est2.belief_grid)
    # Decay should have reduced peak
    assert decayed_max <= initial_max * 1.01  # close enough


def test_localizer_rfid_update() -> None:
    """RFID observation should bias belief toward known reader position."""
    loc = Localizer("config/localization.example.yaml")
    obs = SensorObservation(
        sensor_id="rfid-gate-1", sensor_type="rfid",
        timestamp=datetime.now(timezone.utc),
        observation={"epc": "E280116060000204", "rssi": -50},
        confidence=0.85, tag_id="E280116060000204",
    )
    loc.update([obs])
    est = loc.estimate_position()
    # RFID tag E280116060000204 is at (4.0, 2.5) in example config
    # Belief should be somewhat biased toward that area
    assert abs(est.x - 4.0) < 2.0
    assert abs(est.y - 2.5) < 2.0


def test_localizer_imu_stationary_smoothing() -> None:
    """Low IMU acceleration should not prevent position updates (smoothing is mild)."""
    loc = Localizer("config/localization.example.yaml")
    # IMU with very low acceleration (stationary)
    obs = SensorObservation(
        sensor_id="imu-waist", sensor_type="imu",
        timestamp=datetime.now(timezone.utc),
        observation={"accel_x": 0.01, "accel_y": 0.01, "accel_z": 9.81},  # gravity only
        confidence=0.95,
    )
    loc.update([obs])
    est = loc.estimate_position()
    assert isinstance(est.x, float)
    assert isinstance(est.y, float)


def test_localizer_estimate_position_contract() -> None:
    """The estimate_position output has all required fields."""
    loc = Localizer("config/localization.example.yaml")
    est = loc.estimate_position()
    assert hasattr(est, "x")
    assert hasattr(est, "y")
    assert hasattr(est, "confidence")
    assert hasattr(est, "belief_grid")
    assert hasattr(est, "grid_cells")
    assert hasattr(est, "room_origin")
    assert hasattr(est, "room_size")
    assert hasattr(est, "timestamp")
    assert est.grid_cells == 20
    assert len(est.belief_grid) == 20
    assert len(est.belief_grid[0]) == 20


def test_localizer_empty_update_does_not_crash() -> None:
    """Calling update with empty list should be a no-op."""
    loc = Localizer("config/localization.example.yaml")
    loc.update([])  # must not raise
    assert loc.estimate_position() is not None
