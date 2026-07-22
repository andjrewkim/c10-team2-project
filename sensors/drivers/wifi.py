"""WiFi RSSI/CSI sensor driver.

Expected capture method: iw scan (RSSI mode) or nexmon CSI extractor / ESP32.
Each visible access point produces one observation with a tag_id derived
from its BSSID.

TODO: Replace stub body with real interface capture (scapy, iw, nexmon).
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timezone

from sensors.base import BaseSensor, SensorObservation


class WiFiSensor(BaseSensor):
    """Driver for WiFi RSSI or CSI extraction.

    Parameters
    ----------
    sensor_id : str
        Unique identifier for this WiFi monitor.
    interface : str
        Wireless interface name (e.g. "wlan0", "en0").
    mode : str
        "rssi" or "csi".
    bssids : list[str] | None
        Specific AP MACs to monitor, or empty for all visible.
    """

    def __init__(
        self,
        sensor_id: str,
        interface: str = "wlan0",
        mode: str = "rssi",
        bssids: list[str] | None = None,
    ) -> None:
        super().__init__(sensor_id=sensor_id, sensor_type="wifi")
        self.interface = interface
        self.mode = mode
        self.bssids = bssids or []

    def read(self) -> list[SensorObservation]:
        """Blocking read: capture WiFi probe / beacon RSSI or CSI.

        RSSI mode observation dict:
            bssid, ssid, rssi (dBm), freq (MHz), channel
        Each AP gets its own observation with tag_id=bssid hex.
        """
        if self.mode == "rssi":
            mock_aps = [
                {"bssid": "AA:BB:CC:DD:EE:01", "ssid": "eduroam", "rssi": random.uniform(-75, -45), "freq": 5180, "channel": 36},
                {"bssid": "AA:BB:CC:DD:EE:02", "ssid": "IoT-Net", "rssi": random.uniform(-85, -50), "freq": 2412, "channel": 1},
                {"bssid": "AA:BB:CC:DD:EE:03", "ssid": "", "rssi": random.uniform(-90, -60), "freq": 5765, "channel": 153},
            ]
            results = []
            for ap in mock_aps:
                tag = ap["bssid"].replace(":", "").lower()
                results.append(
                    SensorObservation(
                        sensor_id=self.sensor_id,
                        sensor_type=self.sensor_type,
                        timestamp=datetime.now(timezone.utc),
                        observation=ap,
                        confidence=0.7,
                        tag_id=tag,
                        metadata={"interface": self.interface, "mode": self.mode, "channel": ap["channel"]},
                    )
                )
            return results
        return []


__all__ = ["WiFiSensor"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WiFi sensor self-test")
    parser.add_argument("--interface", default="wlan0")
    parser.add_argument("--mode", choices=["rssi", "csi"], default="rssi")
    args = parser.parse_args()
    sensor = WiFiSensor(sensor_id="wifi-self-test", interface=args.interface, mode=args.mode)
    for o in sensor.read():
        print(f"[{o.timestamp.isoformat()}] {o.sensor_type}/{o.sensor_id}  AP={o.observation.get('bssid','?')}  RSSI={o.observation.get('rssi','?'):.1f} dBm")
