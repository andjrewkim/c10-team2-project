"""Signal-processing and data-parsing utilities for sensor drivers.

Each submodule wraps one sensor's signal-processing algorithms so the
production drivers in ``sensors/drivers/`` can use real parsing and
algorithms instead of stub mock values.

Module layout
-------------
wifi     — CSI line parsing, IQ → amplitude, sliding-window motion detection
imu      — IMU sample parsing, quaternion math, dead-reckoned trajectory
rfid     — TCP socket reader, log-file record parsing, touch detection
uwb      — UWB ranging‑log parsing, outlier filtering, feature extraction
mmwave   — TI mmWave radar point‑cloud / range‑profile TLV decoding
"""

from __future__ import annotations
