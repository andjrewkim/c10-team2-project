#!/usr/bin/env python3
"""Single-file FastAPI backend for the live sensor UI.

Subscribes to all sensor topics via the existing MqttClient wrapper,
maintains in-memory state, and pushes live updates to browsers over a
WebSocket. Recording control delegates to the existing RecordingSession.

Run with:
    pip install -e ".[dev,ui]"
    uvicorn ui.server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from recording.session import RecordingSession
from transport.mqtt_client import MqttClient, MqttConfig

HERE = Path(__file__).parent
LIVE_WINDOW_SECONDS = 5
OBSERVATION_TOPIC = "+/+/+/observation"

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_active_session: RecordingSession | None = None


class _SensorEntry:
    __slots__ = ("sensor_id", "sensor_type", "last_seen", "last_confidence")

    def __init__(self, sensor_id: str, sensor_type: str) -> None:
        self.sensor_id = sensor_id
        self.sensor_type = sensor_type
        self.last_seen: Optional[str] = None
        self.last_confidence: Optional[float] = None


class PanelState:
    """Thread-safe in-memory store of live sensor status + recording info."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sensors: dict[str, _SensorEntry] = {}
        self.recording_state: str = "idle"
        self.recording_label: str = ""
        self.recording_started_at: Optional[datetime] = None
        self.broker_connected: bool = False

    def set_sensors(self, sensor_ids: list[tuple[str, str]]) -> None:
        with self._lock:
            self._sensors = {}
            for sid, stype in sensor_ids:
                self._sensors[sid] = _SensorEntry(sid, stype)

    def update_sensor(self, sensor_id: str, sensor_type: str, confidence: Optional[float], timestamp: str) -> None:
        with self._lock:
            entry = self._sensors.get(sensor_id)
            if entry is not None:
                entry.sensor_type = sensor_type
                entry.last_seen = timestamp
                entry.last_confidence = confidence

    def snapshot(self) -> list[dict[str, Any]]:
        now = datetime.now(timezone.utc)
        results: list[dict[str, Any]] = []
        with self._lock:
            for sid, entry in self._sensors.items():
                status = "stale"
                if entry.last_seen is not None:
                    try:
                        dt = datetime.fromisoformat(entry.last_seen)
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        age = (now - dt).total_seconds()
                        status = "live" if age <= LIVE_WINDOW_SECONDS else "stale"
                    except ValueError:
                        pass
                results.append({
                    "sensor_id": sid,
                    "sensor_type": entry.sensor_type,
                    "status": status,
                    "last_seen": entry.last_seen,
                    "last_confidence": entry.last_confidence,
                })
        return results

    def recording_info(self) -> dict[str, Any]:
        with self._lock:
            elapsed = 0.0
            if self.recording_state == "recording" and self.recording_started_at is not None:
                elapsed = (datetime.now(timezone.utc) - self.recording_started_at).total_seconds()
            return {
                "state": self.recording_state,
                "label": self.recording_label,
                "elapsed_seconds": elapsed,
                "started_at": self.recording_started_at.isoformat() if self.recording_started_at else None,
            }


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

state = PanelState()
mqtt_client: Optional[MqttClient] = None
clients: set[asyncio.Queue[dict[str, Any]]] = set()


def _get_mqtt_config() -> MqttConfig:
    path = os.getenv("MQTT_CONFIG", "config/mqtt.example.yaml")
    try:
        import yaml
        with open(path) as f:
            raw = yaml.safe_load(f)
        return MqttConfig(
            host=raw.get("host", "localhost"),
            port=raw.get("port", 1883),
            keepalive=raw.get("keepalive", 60),
            username=raw.get("username"),
            password=raw.get("password"),
            tls_enabled=raw.get("tls_enabled", False),
            client_id=raw.get("client_id", "iot-ui"),
            topic_prefix=raw.get("topic_prefix", ""),
            qos=raw.get("qos", 1),
        )
    except Exception:
        return MqttConfig()


def _load_sensor_ids() -> list[tuple[str, str]]:
    path = os.getenv("SENSORS_CONFIG", "config/sensors.example.yaml")
    try:
        import yaml
        with open(path) as f:
            raw = yaml.safe_load(f)
        return [(s["id"], s["type"]) for s in raw.get("sensors", [])]
    except Exception:
        return [("mock-001", "mock")]


def _on_mqtt_message(topic: str, payload: bytes) -> None:
    try:
        data = json.loads(payload.decode("utf-8"))
    except Exception:
        return
    sid = data.get("sensor_id", "")
    stype = data.get("sensor_type", "")
    conf = data.get("confidence")
    ts = data.get("timestamp", datetime.now(timezone.utc).isoformat())
    state.update_sensor(sid, stype, conf, ts)


def _bootstrap_mqtt() -> None:
    global mqtt_client
    try:
        config = _get_mqtt_config()
        client = MqttClient(config)
        client.connect()
        client.subscribe(OBSERVATION_TOPIC, _on_mqtt_message)
        mqtt_client = client
        state.broker_connected = True
        print(f"[ui.server] MQTT connected to {config.host}:{config.port}")
    except Exception as exc:
        print(f"[ui.server] MQTT not available: {exc}")
        state.broker_connected = False


async def _broadcast(data: dict[str, Any]) -> None:
    dead: list[asyncio.Queue] = []
    for q in clients:
        try:
            q.put_nowait(data)
        except asyncio.QueueFull:
            dead.append(q)
    for q in dead:
        clients.discard(q)


async def _heartbeat() -> None:
    while True:
        await asyncio.sleep(1)
        msg = {
            "type": "sensor_snapshot",
            "sensors": state.snapshot(),
            "recording": state.recording_info(),
            "broker_connected": state.broker_connected,
        }
        await _broadcast(msg)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> Any:
    state.set_sensors(_load_sensor_ids())
    threading.Thread(target=_bootstrap_mqtt, daemon=True).start()
    bg = asyncio.create_task(_heartbeat())
    yield
    bg.cancel()
    if mqtt_client is not None:
        mqtt_client.disconnect()


app = FastAPI(lifespan=_lifespan)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(HERE / "index.html")


@app.get("/api/sensors")
async def api_sensors() -> list[dict[str, Any]]:
    return state.snapshot()


@app.get("/api/recording")
async def api_recording() -> dict[str, Any]:
    return state.recording_info()


class StartRequest(BaseModel):
    label: str = Field(min_length=1)
    participant: str = Field(min_length=1)
    duration: int = Field(default=0, ge=0)


@app.post("/api/recording/start")
async def api_start_recording(req: StartRequest) -> dict[str, Any]:  # async — MqttClient.connect() runs in a thread via to_thread
    global _active_session
    if state.recording_state == "recording":
        raise HTTPException(400, "Recording already in progress")
    try:
        mqtt_cfg = _get_mqtt_config()
        session = RecordingSession(mqtt_config=mqtt_cfg, output_dir=os.getenv("RECORDING_OUTPUT_DIR", "data/raw"))
        await asyncio.to_thread(session.start, label=req.label, participant_id=req.participant)
        _active_session = session
        state.recording_state = "recording"
        state.recording_label = req.label
        state.recording_started_at = datetime.now(timezone.utc)
        if req.duration > 0:
            asyncio.create_task(_auto_stop(req.duration))
        return {"status": "started", "label": req.label}
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.post("/api/recording/stop")
def api_stop_recording() -> dict[str, Any]:  # sync — session.stop() is synchronous
    global _active_session
    session = _active_session
    if session is None:
        raise HTTPException(400, "No active recording session")
    try:
        path = session.stop()
        _active_session = None
        state.recording_state = "stopped"
        return {"status": "stopped", "output_path": str(path)}
    except Exception as exc:
        raise HTTPException(500, str(exc))


async def _auto_stop(duration: int) -> None:
    global _active_session
    await asyncio.sleep(duration)
    session = _active_session
    if session is not None:
        try:
            path = session.stop()
            _active_session = None
            state.recording_state = "stopped"
            await _broadcast({"type": "recording_stopped", "output_path": str(path)})
        except Exception:
            pass


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    clients.add(q)
    try:
        await websocket.send_json({
            "type": "sensor_snapshot",
            "sensors": state.snapshot(),
            "recording": state.recording_info(),
            "broker_connected": state.broker_connected,
        })
        while True:
            data = await q.get()
            await websocket.send_json(data)
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(q)
