"""src — All runtime application code for the multi-sensor IoT system.

Package layout
--------------
src.actions       — Action handlers triggered by fusion decisions
src.dashboard     — Localization heatmap dashboard (FastAPI)
src.fusion        — Sensor fusion strategies + localization engine
src.sensors       — Sensor drivers, mock sensor, reader pool, lab integration
src.transport     — MQTT client for sensor data transport
src.ui            — Debug web UI (FastAPI + WebSocket)
src.session       — Recording session manager
src.cli           — Recording CLI tool
"""

from __future__ import annotations
