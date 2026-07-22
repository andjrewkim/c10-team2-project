#!/usr/bin/env python3
"""FastAPI dashboard: live sensor status + spatial localisation heatmap.

Polls all sensors in background threads via :class:`ReaderPool`, runs
the :class:`Localizer` on each new batch, and pushes updates to every
connected browser over a single WebSocket.

Run with:
    pip install -e ".[dev,ui]"
    uvicorn src.dashboard.server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import os
import threading
import time as time_module
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from sensors.base import SensorObservation
from src.localization import Localizer
from src.reader_pool import ReaderPool, load_sensors_from_config

HERE = Path(__file__).parent
STATIC_DIR = HERE / "static"
SENSORS_CONFIG = os.getenv("SENSORS_CONFIG", "config/sensors.example.yaml")
LOCALIZATION_CONFIG = os.getenv("LOCALIZATION_CONFIG", "config/localization.example.yaml")
LIVE_WINDOW_SECONDS = 5

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

localizer = Localizer(LOCALIZATION_CONFIG)
pool: Optional[ReaderPool] = None
clients: set[asyncio.Queue[dict[str, Any]]] = set()
_latest_observations: list[SensorObservation] = []
_latest_obs_lock = threading.Lock()
_position_cache: dict[str, Any] = {}
_position_lock = threading.Lock()


def _on_observation(obs: SensorObservation) -> None:
    """Callback invoked by ReaderPool threads for every new reading."""
    with _latest_obs_lock:
        _latest_observations.append(obs)
        # Keep a bounded buffer
        if len(_latest_observations) > 500:
            _latest_observations[:] = _latest_observations[-200:]


def _polling_thread() -> None:
    """Background thread: batched sensor processing + localisation update."""
    while True:
        time_module.sleep(0.1)  # 10 Hz update rate
        with _latest_obs_lock:
            batch = list(_latest_observations)
            _latest_observations.clear()
        if not batch:
            continue
        localizer.update(batch)
        est = localizer.estimate_position()
        with _position_lock:
            _position_cache.clear()
            _position_cache["position"] = {"x": est.x, "y": est.y, "confidence": est.confidence}
            _position_cache["belief_grid"] = est.belief_grid
            _position_cache["grid_cells"] = est.grid_cells
            _position_cache["room_origin"] = list(est.room_origin)
            _position_cache["room_size"] = list(est.room_size)
            _position_cache["timestamp"] = est.timestamp


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(app: FastAPI) -> Any:
    global pool
    sensors = load_sensors_from_config(SENSORS_CONFIG)
    pool = ReaderPool(sensors, min_interval=0.05)
    pool.set_callback(_on_observation)
    pool.start()
    t = threading.Thread(target=_polling_thread, daemon=True)
    t.start()
    bg = asyncio.create_task(_heartbeat())
    yield
    bg.cancel()
    if pool is not None:
        pool.stop()


app = FastAPI(lifespan=_lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
async def api_status() -> dict[str, Any]:
    """Return latest sensor snapshot."""
    sensors_info = []
    if pool is not None:
        for s in pool.sensors:
            sensors_info.append({
                "sensor_id": s.sensor_id,
                "sensor_type": s.sensor_type,
                "status": "connected",
            })
    return {"sensors": sensors_info}


@app.get("/api/localization")
async def api_localization() -> dict[str, Any]:
    """Return the latest position estimate + belief grid."""
    with _position_lock:
        return dict(_position_cache)


async def _heartbeat() -> None:
    """Push sensor + localisation snapshot to all clients at ~8 Hz."""
    while True:
        await asyncio.sleep(0.125)
        # Build sensor snapshot from pool
        sensors_info: list[dict[str, Any]] = []
        if pool is not None:
            for s in pool.sensors:
                sensors_info.append({
                    "sensor_id": s.sensor_id,
                    "sensor_type": s.sensor_type,
                    "status": "connected",
                })
        with _position_lock:
            pos_data = dict(_position_cache)
        msg: dict[str, Any] = {
            "type": "dashboard_update",
            "sensors": sensors_info,
        }
        if pos_data:
            msg["localization"] = pos_data
        await _broadcast(msg)


async def _broadcast(data: dict[str, Any]) -> None:
    dead: list[asyncio.Queue] = []
    for q in clients:
        try:
            q.put_nowait(data)
        except asyncio.QueueFull:
            dead.append(q)
    for q in dead:
        clients.discard(q)


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    clients.add(q)
    try:
        # Push initial state
        with _position_lock:
            initial_pos = dict(_position_cache)
        sensors_info = []
        if pool is not None:
            for s in pool.sensors:
                sensors_info.append({
                    "sensor_id": s.sensor_id,
                    "sensor_type": s.sensor_type,
                    "status": "connected",
                })
        initial: dict[str, Any] = {"type": "dashboard_update", "sensors": sensors_info}
        if initial_pos:
            initial["localization"] = initial_pos
        await websocket.send_json(initial)

        while True:
            data = await q.get()
            await websocket.send_json(data)
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(q)
