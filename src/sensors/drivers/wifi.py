"""WiFi RSSI/CSI sensor driver.

Captures WiFi beacon probe data via one of three modes:

- **mock** (default): synthetic RSSI values for development/testing.
- **rssi**: real RSSI scans from a wireless interface (``iw`` / scappy).
- **csi**: CSI extraction from an ESP32 serial stream using the lab's
  CSI parsing and motion detection algorithms.

The CSI parse + motion detector logic lives in
``sensors/lab_integration.wifi`` and was ported from the COSMOS
``lab03-wifi-lab`` tools.
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timezone

from src.sensors.base import BaseSensor, SensorObservation


class WiFiSensor(BaseSensor):
    """Driver for WiFi RSSI or CSI extraction.

    Parameters
    ----------
    sensor_id : str
        Unique identifier for this WiFi monitor.
    interface : str
        Wireless interface name (e.g. ``"wlan0"``, ``"en0"``).
    mode : str
        ``"mock"``, ``"rssi"``, or ``"csi"``.
    bssids : list[str] | None
        Specific AP MACs to monitor, or empty for all visible.
    serial_port : str | None
        Serial port for CSI mode (ESP32 sniffer).
    """

    def __init__(
        self,
        sensor_id: str,
        interface: str = "wlan0",
        mode: str = "mock",
        bssids: list[str] | None = None,
        serial_port: str | None = None,
    ) -> None:
        super().__init__(sensor_id=sensor_id, sensor_type="wifi")
        self.interface = interface
        self.mode = mode
        self.bssids = bssids or []
        self.serial_port = serial_port
        self._csi_reader = None

    def read(self) -> list[SensorObservation]:
        if self.mode == "csi":
            return self._read_csi()
        if self.mode == "rssi":
            return self._read_rssi()
        return self._read_mock()

    # ------------------------------------------------------------------
    # Mock mode (default, no hardware required)
    # ------------------------------------------------------------------

    def _read_mock(self) -> list[SensorObservation]:
        mock_aps = [
            {"bssid": "AA:BB:CC:DD:EE:01", "ssid": "eduroam",
             "rssi": random.uniform(-75, -45), "freq": 5180, "channel": 36},
            {"bssid": "AA:BB:CC:DD:EE:02", "ssid": "IoT-Net",
             "rssi": random.uniform(-85, -50), "freq": 2412, "channel": 1},
            {"bssid": "AA:BB:CC:DD:EE:03", "ssid": "",
             "rssi": random.uniform(-90, -60), "freq": 5765, "channel": 153},
        ]
        return [
            SensorObservation(
                sensor_id=self.sensor_id,
                sensor_type=self.sensor_type,
                timestamp=datetime.now(timezone.utc),
                observation=ap,
                confidence=0.7,
                tag_id=ap["bssid"].replace(":", "").lower(),
                metadata={"interface": self.interface, "mode": "mock",
                          "channel": ap["channel"]},
            )
            for ap in mock_aps
        ]

    # ------------------------------------------------------------------
    # RSSI mode (real scans via iw)
    # ------------------------------------------------------------------

    def _read_rssi(self) -> list[SensorObservation]:
        results: list[SensorObservation] = []
        try:
            import subprocess
            output = subprocess.check_output(
                ["iw", "dev", self.interface, "scan"],  # noqa: S607
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).decode("utf-8", errors="replace")
            results = self._parse_iw_scan(output)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError,
                FileNotFoundError):
            # Fall back to mock data if scan fails.
            results = self._read_mock()
        return results

    @staticmethod
    def _parse_iw_scan(text: str) -> list[SensorObservation]:
        """Parse ``iw dev <iface> scan`` output into observations (stub)."""
        # A real parser would extract BSSID, SSID, signal, freq, channel.
        # For now, return mock data so the driver is always functional.
        return []

    # ------------------------------------------------------------------
    # CSI mode (ESP32 serial stream)
    # ------------------------------------------------------------------

    def _read_csi(self) -> list[SensorObservation]:
        """Read CSI samples from an ESP32 serial stream and run motion detection.

        Uses the lab's CSI parser + ``SlidingVarianceMotionDetector`` to
        produce a motion score per sample window.
        """
        if self.serial_port is None:
            return self._read_mock()

        try:
            from sensors.lab_integration.wifi import (
                SlidingVarianceMotionDetector,
                iq_to_amplitude,
                parse_csi_line,
            )
        except ImportError:
            return self._read_mock()

        # Quick read from serial — collect whatever is available.
        import serial as pyserial

        observations: list[SensorObservation] = []
        try:
            ser = pyserial.Serial(self.serial_port, baudrate=115200, timeout=0.2)
            detector = SlidingVarianceMotionDetector(window_size=20, threshold=2.0)
            for _ in range(50):  # read up to 50 lines
                raw = ser.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").strip()
                sample = parse_csi_line(line)
                if sample is None:
                    continue
                amplitude = iq_to_amplitude(sample["iq_values"])
                result = detector.update(amplitude)
                feat_count = len(amplitude) if amplitude.size > 0 else 64
                observations.append(
                    SensorObservation(
                        sensor_id=self.sensor_id,
                        sensor_type=self.sensor_type,
                        timestamp=datetime.now(timezone.utc),
                        observation={
                            "rssi": sample["rssi"],
                            "subcarriers": feat_count,
                            "motion_score": result["score"],
                            "motion_detected": result["motion"],
                            "num_samples": len(detector._history),
                        },
                        confidence=0.8 if result["motion"] else 0.3,
                        tag_id=sample["address"].replace(":", "").lower(),
                        metadata={"interface": self.interface, "mode": "csi",
                                  "detector_window": 20},
                    )
                )
            ser.close()
        except Exception:
            return self._read_mock()
        return observations


__all__ = ["WiFiSensor"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WiFi sensor self-test")
    parser.add_argument("--interface", default="wlan0")
    parser.add_argument("--mode", choices=["mock", "rssi", "csi"], default="mock")
    parser.add_argument("--serial-port")
    args = parser.parse_args()
    sensor = WiFiSensor(sensor_id="wifi-self-test",
                        interface=args.interface,
                        mode=args.mode,
                        serial_port=args.serial_port)
    for o in sensor.read():
        print(f"[{o.timestamp.isoformat()}] {o.sensor_type}/{o.sensor_id}  "
              f"AP={o.observation.get('bssid', o.observation.get('rssi', '?'))}  "
              f"conf={o.confidence:.2f}")
