"""Signal-processing and data-parsing utilities extracted from the COSMOS lab
repository (``mmwave/`` folder of this project).

Each submodule wraps one sensor's lab implementations so the production
drivers in ``sensors/drivers/`` can use real parsing and algorithms instead
of stub mock values.  The original lab scripts live in ``mmwave/labs/`` for
reference; this package provides the clean, reusable, project-integrated
versions of their key functions.

Module layout
-------------
wifi     — CSI line parsing, IQ → amplitude, sliding-window motion detection
imu      — IMU sample parsing, quaternion math, dead-reckoned trajectory
rfid     — TCP socket reader, log-file record parsing, touch detection
uwb      — UWB ranging‑log parsing, outlier filtering, feature extraction
mmwave   — TI mmWave radar point‑cloud / range‑profile TLV decoding
"""

from __future__ import annotations
