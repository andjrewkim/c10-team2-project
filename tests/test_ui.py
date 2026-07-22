"""Unit tests for the UI state model (PanelState) — no FastAPI, no WebSocket, no MQTT."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from ui.server import LIVE_WINDOW_SECONDS, PanelState


def _configure(state: PanelState) -> None:
    state.set_sensors([("imu-1", "imu"), ("radar-1", "mmwave"), ("uwb-1", "uwb")])


def test_initial_snapshot_marks_everything_stale() -> None:
    state = PanelState()
    _configure(state)
    snap = state.snapshot()
    assert len(snap) == 3
    assert all(s["status"] == "stale" for s in snap)
    assert all(s["last_seen"] is None for s in snap)


def test_update_from_message_marks_live_for_configured() -> None:
    state = PanelState()
    _configure(state)
    state.update_sensor("imu-1", "imu", 0.85, datetime.now(timezone.utc).isoformat())
    snap = state.snapshot()
    assert snap[0]["status"] == "live"
    assert snap[0]["last_confidence"] == 0.85


def test_unconfigured_sensor_is_ignored() -> None:
    state = PanelState()
    _configure(state)
    state.update_sensor("rogue-sensor", "unknown", 0.99, datetime.now(timezone.utc).isoformat())
    snap = state.snapshot()
    assert [s["sensor_id"] for s in snap] == ["imu-1", "radar-1", "uwb-1"]


def test_status_transitions_live_to_stale_with_age() -> None:
    state = PanelState()
    _configure(state)
    old_ts = (datetime.now(timezone.utc) - timedelta(seconds=LIVE_WINDOW_SECONDS + 1)).isoformat()
    state.update_sensor("imu-1", "imu", 0.5, old_ts)
    assert state.snapshot()[0]["status"] == "stale"


def test_status_live_at_window_boundary() -> None:
    state = PanelState()
    _configure(state)
    within = (datetime.now(timezone.utc) - timedelta(seconds=LIVE_WINDOW_SECONDS - 1)).isoformat()
    state.update_sensor("imu-1", "imu", 0.5, within)
    assert state.snapshot()[0]["status"] == "live"


def test_naive_timestamp_is_treated_as_utc() -> None:
    state = PanelState()
    _configure(state)
    naive = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
    state.update_sensor("imu-1", "imu", 0.5, naive)
    assert state.snapshot()[0]["status"] == "live"


def test_recording_state_defaults() -> None:
    state = PanelState()
    info = state.recording_info()
    assert info["state"] == "idle"
    assert info["label"] == ""
    assert info["elapsed_seconds"] == 0.0
