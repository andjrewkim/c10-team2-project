"""WiFi CSI (Channel State Information) utilities extracted from
``mmwave/labs/lab03-wifi-lab/wifi_lab/tools/``.

Provides CSI line parsing, IQ → amplitude conversion, and a
sliding-window variance motion detector — the same algorithms used
in the student lab but packaged for the project's sensor drivers.
"""

from __future__ import annotations

import collections
import re
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPECTED_SUBCARRIERS = 64

CSI_LINE_PATTERN = re.compile(
    r"<timestamp>(\d+)</timestamp><rssi>(-?\d+)</rssi>"
    r"<address>([0-9A-Fa-f:]+)</address>(.+)"
)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

CSI_SAMPLE_FIELDS = ("timestamp_cycles", "rssi", "address", "iq_values")


def parse_csi_line(line: str) -> Optional[dict]:
    """Parse one ESP32 sniffer serial line into a CSI sample dict, or
    return *None* if the line doesn't match the expected format.

    Result keys
    -----------
    timestamp_cycles : int
    rssi : int               (dBm)
    address : str            (MAC of the transmitter)
    iq_values : list[int]    (flat interleaved imaginary, real)
    """
    match = CSI_LINE_PATTERN.search(line)
    if not match:
        return None
    timestamp_str, rssi_str, address, values_str = match.groups()
    tokens = values_str.split()
    if not tokens:
        return None
    try:
        iq_values = [int(t) for t in tokens]
    except ValueError:
        return None
    return {
        "timestamp_cycles": int(timestamp_str),
        "rssi": int(rssi_str),
        "address": address,
        "iq_values": iq_values,
    }


def iq_to_amplitude(iq_values: list[int]) -> np.ndarray:
    """Convert a flat (imag, real) × *N* int list into an *N*-length
    amplitude (magnitude) array.

    If the list has an odd number of entries the last value is silently
    dropped so the result is always well-defined.
    """
    values = np.asarray(iq_values, dtype=np.float32)
    if values.size % 2 != 0:
        values = values[:-1]
    imag = values[0::2]
    real = values[1::2]
    return np.sqrt(imag**2 + real**2)


# ---------------------------------------------------------------------------
# Motion detection
# ---------------------------------------------------------------------------


class SlidingVarianceMotionDetector:
    """Sliding-window variance-based motion detector.

    Appends each new amplitude vector to a fixed-size history deque and
    computes the per-subcarrier variance across that window.  The mean of
    those variances is the motion *score*; when it exceeds the threshold
    the detector reports *motion=True*.
    """

    def __init__(self, window_size: int = 20, threshold: float = 2.0) -> None:
        self.window_size = window_size
        self.threshold = threshold
        self._history: collections.deque[np.ndarray] = collections.deque(
            maxlen=window_size
        )

    def update(self, amplitude_vector: np.ndarray) -> dict:
        """Feed one amplitude vector and return the detection result.

        Returns
        -------
        dict with keys ``score`` (float) and ``motion`` (bool).
        """
        self._history.append(amplitude_vector)
        if len(self._history) < 2:
            return {"score": 0.0, "motion": False}

        per_subcarrier_var = np.var(np.stack(self._history), axis=0)
        score = float(np.mean(per_subcarrier_var))
        return {"score": score, "motion": score > self.threshold}

    def reset(self) -> None:
        self._history.clear()


__all__ = [
    "EXPECTED_SUBCARRIERS",
    "CSI_LINE_PATTERN",
    "CSI_SAMPLE_FIELDS",
    "parse_csi_line",
    "iq_to_amplitude",
    "SlidingVarianceMotionDetector",
]
