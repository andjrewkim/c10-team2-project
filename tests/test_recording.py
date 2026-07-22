"""Tests for the RecordingSession using in-memory ingestion (no MQTT broker)."""

from __future__ import annotations

import csv
import json
import uuid

from recording.session import RecordingSession


def _make_session() -> RecordingSession:
    # No MQTT config → in-memory mode; tests use ingest() directly.
    return RecordingSession(output_dir="/tmp/test_recording")


def test_recording_session_start_and_stop() -> None:
    session = _make_session()
    session.start(label="walking", participant_id="alice")
    assert session._started is True
    assert session._label == "walking"
    assert session._participant_id == "alice"

    # Simulate receiving a few observations
    session.ingest(
        sensor_id="imu-1",
        sensor_type="imu",
        timestamp="2026-07-22T10:00:00Z",
        observation='{"accel_x": 1.0}',
        confidence=0.95,
    )
    session.ingest(
        sensor_id="radar-1",
        sensor_type="mmwave",
        timestamp="2026-07-22T10:00:01Z",
        observation='{"num_objects": 3}',
        confidence=0.80,
    )
    session.ingest(
        sensor_id="uwb-1",
        sensor_type="uwb",
        timestamp="2026-07-22T10:00:02Z",
        observation='{"range_m": 1.5}',
        confidence=0.90,
        tag_id="tag-001",
        position={"x": 0.0, "y": 0.0, "z": 2.5},
    )

    path = session.stop()
    assert path.exists()

    # Read back and verify
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 3

    assert rows[0]["sensor_id"] == "imu-1"
    assert rows[0]["sensor_type"] == "imu"
    assert rows[0]["label"] == "walking"
    assert rows[0]["participant_id"] == "alice"
    assert rows[0]["tag_id"] == ""

    assert rows[1]["sensor_type"] == "mmwave"

    assert rows[2]["sensor_id"] == "uwb-1"
    assert rows[2]["tag_id"] == "tag-001"
    assert rows[2]["position_x"] == "0.0"
    assert rows[2]["position_y"] == "0.0"
    assert rows[2]["position_z"] == "2.5"

    path.unlink()  # cleanup


def test_recording_session_subset_of_sensors() -> None:
    """Only a subset of sensor types are active."""
    session = _make_session()
    session.start(label="sitting", participant_id="bob")

    # Only IMU and WiFi are connected
    session.ingest(sensor_id="imu-hip", sensor_type="imu", timestamp="now", observation="{}", confidence=0.9)
    session.ingest(sensor_id="wifi-1", sensor_type="wifi", timestamp="now", observation="{}", confidence=0.7, tag_id="aabbcc")

    path = session.stop()
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    sensor_types = {r["sensor_type"] for r in rows}
    assert sensor_types == {"imu", "wifi"}

    path.unlink()


def test_recording_session_session_id_is_unique() -> None:
    s1 = _make_session()
    s1.start(label="a", participant_id="p1")
    s1.ingest(sensor_id="s1", sensor_type="mock", timestamp="t1", observation="{}", confidence=1.0)
    p1 = s1.stop()

    s2 = _make_session()
    s2.start(label="b", participant_id="p2")
    s2.ingest(sensor_id="s2", sensor_type="mock", timestamp="t2", observation="{}", confidence=1.0)
    p2 = s2.stop()

    with open(p1) as f:
        r1 = list(csv.DictReader(f))
    with open(p2) as f:
        r2 = list(csv.DictReader(f))

    assert r1[0]["session_id"] != r2[0]["session_id"]
    p1.unlink()
    p2.unlink()
