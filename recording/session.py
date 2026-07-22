"""Recording session: subscribe to sensor topics and persist raw observations.

Usage
-----
    session = RecordingSession(
        output_dir="data/raw",
        mqtt_config=...,   # from config/mqtt.example.yaml
    )
    session.start(label="walking", participant_id="alice")
    # ... wait for duration ...
    session.stop()
"""

from __future__ import annotations

import csv
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from transport.mqtt_client import MqttClient, MqttConfig

# -----------------------------------------------------------------------
# TODO: Migrate from CSV to Parquet once the team agrees on a schema.
#   Parquet is preferred for ML training pipelines (columnar, efficient,
#   typed).  Requires pyarrow.  Uncomment the writer below and replace
#   the CSV writer when ready.
#
#   import pyarrow as pa
#   import pyarrow.parquet as pq
# -----------------------------------------------------------------------

_OBSERVATION_COLUMNS = [
    "session_id",
    "label",
    "participant_id",
    "sensor_id",
    "sensor_type",
    "tag_id",
    "timestamp",
    "observation",
    "confidence",
    "position_x",
    "position_y",
    "position_z",
    "raw_metadata",
]


class RecordingSession:
    """Manages one data-collection session.

    Connects to the MQTT broker, subscribes to all sensor observation
    topics, and writes every received observation to a single file.

    **Important**: not all sensors need to be connected for every session.
    Only the observations that reach the broker will be recorded.
    """

    def __init__(
        self,
        mqtt_config: MqttConfig | None = None,
        output_dir: str | Path = "data/raw",
    ) -> None:
        self._mqtt: MqttClient | None = MqttClient(mqtt_config) if mqtt_config else None
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._session_id: str = ""
        self._label: str = ""
        self._participant_id: str = ""
        self._rows: list[dict[str, str | float | None]] = []
        self._started: bool = False
        self._topic_subscribed: str = ""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, label: str, participant_id: str) -> None:
        """Begin a recording session.

        Parameters
        ----------
        label : str
            Activity label for the session (e.g. "walking", "sitting").
        participant_id : str
            Identifier for the person performing the activity.
        """
        self._session_id = uuid.uuid4().hex[:12]
        self._label = label
        self._participant_id = participant_id
        self._rows = []

        if self._mqtt is not None:
            self._mqtt.connect()
            self._topic_subscribed = "+/+/+/observation"
            self._mqtt.subscribe(self._topic_subscribed, self._on_observation)

        self._started = True

    def stop(self) -> Path:
        """End the session and flush buffered observations to disk.

        Returns
        -------
        Path
            Path to the written file.
        """
        if not self._started:
            raise RuntimeError("Session was never started.")

        if self._mqtt is not None:
            self._mqtt.disconnect()

        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"session_{self._session_id}_{self._label}_{timestamp_str}.csv"
        filepath = self._output_dir / filename

        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_OBSERVATION_COLUMNS)
            writer.writeheader()
            writer.writerows(self._rows)

        print(f"[RecordingSession] Wrote {len(self._rows)} observations to {filepath}")
        self._started = False
        return filepath

    # ------------------------------------------------------------------
    # MQTT callback
    # ------------------------------------------------------------------

    def _on_observation(self, topic: str, payload: bytes) -> None:
        """Deserialise a JSON observation and buffer it."""
        try:
            data = json.loads(payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return  # skip malformed messages

        pos = data.get("position")
        row: dict[str, str | float | None] = {
            "session_id": self._session_id,
            "label": self._label,
            "participant_id": self._participant_id,
            "sensor_id": data.get("sensor_id"),
            "sensor_type": data.get("sensor_type"),
            "tag_id": data.get("tag_id"),
            "timestamp": data.get("timestamp"),
            "observation": json.dumps(data.get("observation"), default=str),
            "confidence": data.get("confidence"),
            "position_x": pos.get("x") if isinstance(pos, dict) else None,
            "position_y": pos.get("y") if isinstance(pos, dict) else None,
            "position_z": pos.get("z") if isinstance(pos, dict) else None,
            "raw_metadata": json.dumps(data.get("metadata", {}), default=str),
        }
        self._rows.append(row)

    # ------------------------------------------------------------------
    # Direct ingestion (for testing without real MQTT broker)
    # ------------------------------------------------------------------

    def ingest(
        self,
        sensor_id: str,
        sensor_type: str,
        timestamp: str,
        observation: str,
        confidence: float,
        tag_id: str | None = None,
        position: dict[str, float] | None = None,
        metadata: str = "{}",
    ) -> None:
        """Directly insert a row into the current session buffer.

        Used for unit testing; skips MQTT entirely.
        """
        row: dict[str, str | float | None] = {
            "session_id": self._session_id,
            "label": self._label,
            "participant_id": self._participant_id,
            "sensor_id": sensor_id,
            "sensor_type": sensor_type,
            "tag_id": tag_id,
            "timestamp": timestamp,
            "observation": observation,
            "confidence": confidence,
            "position_x": position.get("x") if position else None,
            "position_y": position.get("y") if position else None,
            "position_z": position.get("z") if position else None,
            "raw_metadata": metadata,
        }
        self._rows.append(row)
