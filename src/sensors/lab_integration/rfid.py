"""UHF RFID reader utilities extracted from
``mmwave/labs/lab06-rfid-lab/RFID_Lab/``.

Provides a TCP socket reader, log-file record parsing, and a baseline-based
touch detection algorithm — the same components used in the student lab's
``touch_detector_gui.py``.
"""

from __future__ import annotations

import logging
import socket
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RfidRecord:
    epc: str
    timestamp: str
    rssi: int
    read_count: int
    line_number: int
    source_file: Optional[Path] = None


# ---------------------------------------------------------------------------
# Log-file parsing
# ---------------------------------------------------------------------------


def parse_rfid_line(line: str, line_number: int = 0, source_file: Optional[Path] = None) -> Optional[RfidRecord]:
    """Parse one line from a Zebra / Impinj TCP log into an ``RfidRecord``.

    Expected format (space-separated)::

        EPC_TIMESTAMP  RSSI  READ_COUNT

    Returns *None* if the line doesn't match.
    """
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        rssi = int(parts[-2])
        read_count = int(parts[-1])
    except ValueError:
        return None
    return RfidRecord(
        epc=parts[0],
        timestamp=" ".join(parts[1:-2]),
        rssi=rssi,
        read_count=read_count,
        line_number=line_number,
        source_file=source_file,
    )


def read_log_file(path: Path, encoding: str = "utf-8") -> list[RfidRecord]:
    """Read an entire RFID log file into a list of ``RfidRecord``."""
    records: list[RfidRecord] = []
    with path.open("r", encoding=encoding, errors="replace") as f:
        for ln, line in enumerate(f, start=1):
            record = parse_rfid_line(line, ln, path)
            if record is not None:
                records.append(record)
    return records


# ---------------------------------------------------------------------------
# TCP stream reader
# ---------------------------------------------------------------------------


def read_tcp_stream(
    host: str,
    port: int,
    duration: float,
    *,
    buffer_size: int = 4096,
    encoding: str = "utf-8",
    connect_timeout: float = 10.0,
) -> list[RfidRecord]:
    """Connect to an RFID reader over TCP and read records for *duration*
    seconds.

    Returns a list of ``RfidRecord`` instances parsed from the stream.
    """
    records: list[RfidRecord] = []
    end_time = time.monotonic() + duration
    line_number = 0

    with socket.create_connection((host, port), timeout=connect_timeout) as sock:
        buffer = ""
        while time.monotonic() < end_time:
            remaining = end_time - time.monotonic()
            sock.settimeout(max(0.1, remaining))
            try:
                data = sock.recv(buffer_size)
            except socket.timeout:
                continue
            if not data:
                break

            buffer += data.decode(encoding, errors="replace")
            lines = buffer.splitlines(keepends=True)
            if lines and not lines[-1].endswith(("\n", "\r")):
                buffer = lines.pop()
            else:
                buffer = ""
            for raw_line in lines:
                line_number += 1
                rec = parse_rfid_line(raw_line, line_number)
                if rec is not None:
                    records.append(rec)

    return records


# ---------------------------------------------------------------------------
# Touch detection (baseline-based)
# ---------------------------------------------------------------------------


@dataclass
class RfidTouchDetector:
    """Baseline-based RFID touch/release detector for one or more tags.

    Calibrates by collecting baseline RSSI and read-rate statistics over a
    calibration window, then monitors for drops that indicate a touch event.
    """

    selected_epcs: list[str] = field(default_factory=list)
    calibration_seconds: float = 10.0
    window_seconds: float = 2.0
    no_read_seconds: float = 1.0
    rssi_drop_touch_db: float = 8.0
    rssi_drop_release_db: float = 3.0
    rate_drop_touch_fraction: float = 0.50
    rate_drop_release_fraction: float = 0.30

    # --- internal state ---
    _stage: str = "idle"
    _calibration_end: float = 0.0
    _detection_start: float = 0.0
    _window_samples: dict[str, deque] = field(default_factory=dict, repr=False)
    _calibration_samples: dict[str, list] = field(default_factory=dict, repr=False)
    _baseline_rssi: dict[str, Optional[float]] = field(default_factory=dict, repr=False)
    _baseline_rate: dict[str, Optional[float]] = field(default_factory=dict, repr=False)
    _touch_state: dict[str, bool] = field(default_factory=dict, repr=False)
    _last_seen: dict[str, Optional[float]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        for epc in self.selected_epcs:
            self._window_samples[epc] = deque()
            self._calibration_samples[epc] = []
            self._baseline_rssi[epc] = None
            self._baseline_rate[epc] = None
            self._touch_state[epc] = False
            self._last_seen[epc] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_calibration(self, now: float) -> None:
        self._stage = "calibrating"
        self._calibration_end = now + self.calibration_seconds
        for epc in self.selected_epcs:
            self._window_samples[epc].clear()
            self._calibration_samples[epc].clear()
            self._baseline_rssi[epc] = None
            self._baseline_rate[epc] = None
            self._touch_state[epc] = False
            self._last_seen[epc] = None

    def feed_record(self, record: RfidRecord, now: float) -> None:
        epc = record.epc.upper()
        epc_lookup = {e.upper(): e for e in self.selected_epcs}
        canonical = epc_lookup.get(epc)
        if canonical is None:
            return

        self._window_samples[canonical].append((now, record.rssi, record.read_count))
        self._last_seen[canonical] = now
        if self._stage == "calibrating":
            self._calibration_samples[canonical].append(record.rssi)

        self._prune(now)

    def update_stage(self, now: float) -> None:
        self._prune(now)
        if self._stage == "calibrating" and now >= self._calibration_end:
            self._finish_calibration(now)

    def touch_status(self, epc: str, now: float) -> dict:
        samples = list(self._window_samples.get(epc, []))
        current_rssi: Optional[float] = (
            statistics.median([s[1] for s in samples]) if samples else None
        )
        current_rate = (
            sum(s[2] for s in samples) / self.window_seconds if samples else 0.0
        )

        baseline_rssi = self._baseline_rssi.get(epc)
        baseline_rate = self._baseline_rate.get(epc)

        rssi_drop: Optional[float] = None
        if baseline_rssi is not None and current_rssi is not None:
            rssi_drop = baseline_rssi - current_rssi

        rate_drop: Optional[float] = None
        if baseline_rate is not None and baseline_rate > 0:
            rate_drop = max(0.0, 1.0 - current_rate / baseline_rate)

        seconds_since_seen: Optional[float] = None
        last = self._last_seen.get(epc)
        if last is not None:
            seconds_since_seen = now - last

        no_read = seconds_since_seen is not None and seconds_since_seen >= self.no_read_seconds

        touched, reason = self._evaluate(epc, rssi_drop, rate_drop, no_read)
        return {
            "epc": epc,
            "touched": touched,
            "reason": reason,
            "baseline_rssi": baseline_rssi,
            "current_rssi": current_rssi,
            "rssi_drop": rssi_drop,
            "baseline_rate": baseline_rate,
            "current_rate": current_rate,
            "rate_drop": rate_drop,
            "seconds_since_seen": seconds_since_seen,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        for epc in self.selected_epcs:
            dq = self._window_samples[epc]
            while dq and dq[0][0] < cutoff:
                dq.popleft()

    def _finish_calibration(self, now: float) -> None:
        for epc in self.selected_epcs:
            rssi_vals = self._calibration_samples[epc]
            if rssi_vals:
                self._baseline_rssi[epc] = statistics.median(rssi_vals)
        self._detection_start = now
        self._stage = "detecting"

    def _evaluate(
        self,
        epc: str,
        rssi_drop: Optional[float],
        rate_drop: Optional[float],
        no_read: bool,
    ) -> tuple[bool, str]:
        if self._stage != "detecting":
            return False, "not detecting"

        # If the tag disappears entirely, that's a strong touch signal.
        if no_read:
            self._touch_state[epc] = True
            return True, "no recent reads"

        was_touched = self._touch_state.get(epc, False)

        # Transition to TOUCHED.
        if not was_touched:
            if rssi_drop is not None and rssi_drop >= self.rssi_drop_touch_db:
                self._touch_state[epc] = True
                return True, f"RSSI drop {rssi_drop:.1f} dB"
            if rate_drop is not None and rate_drop >= self.rate_drop_touch_fraction:
                self._touch_state[epc] = True
                return True, f"rate drop {rate_drop:.0%}"

        # Transition to CLEAR (release).
        if was_touched:
            rssi_ok = rssi_drop is None or rssi_drop < self.rssi_drop_release_db
            rate_ok = rate_drop is None or rate_drop < self.rate_drop_release_fraction
            if rssi_ok and rate_ok:
                self._touch_state[epc] = False
                return False, "signal recovered"

            return True, "still touching"

        return False, "baseline stable"

    @property
    def stage(self) -> str:
        return self._stage

    @property
    def touched_epcs(self) -> list[str]:
        return [epc for epc in self.selected_epcs if self._touch_state.get(epc, False)]


__all__ = [
    "RfidRecord",
    "parse_rfid_line",
    "read_log_file",
    "read_tcp_stream",
    "RfidTouchDetector",
]
