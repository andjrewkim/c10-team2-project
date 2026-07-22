"""WiFi RSSI/CSI sensor driver stub.

TODO: Replace the stub body with real interface capture using
      scapy, iw, nexmon CSI extractor, or ESP32 promiscuous mode.
"""

from __future__ import annotations

import argparse
import random
import subprocess
from datetime import datetime, timezone
from typing import Any

from sensors.base import BaseSensor, SensorObservation


class WiFiSensor(BaseSensor):
    """Driver for WiFi RSSI or CSI extraction.

    Mode ``"rssi"`` polls the wireless interface and reports signal
    strength per visible access point.  Mode ``"csi"`` (channel state
    information) requires specialised firmware and a tool like nexmon.

    Parameters
    ----------
    sensor_id : str
        Unique identifier for this WiFi monitor (e.g. "wifi-1").
    interface : str
        Wireless interface name (e.g. "wlan0", "en0").
    mode : str
        ``"rssi"`` or ``"csi"``.
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

        RSSI mode — expected observation format (dict):
            bssid    : str        (AP MAC address)
            ssid     : str        (network name, may be hidden)
            rssi     : float      (dBm, e.g. -55.0)
            freq     : int        (channel frequency, MHz)
            channel  : int        (channel number)

        CSI mode — expected observation format (dict):
            bssid    : str
            csi      : list[complex]  (channel state matrix)
            rssi     : float
            n_subcarriers : int

        Returns
        -------
        list[SensorObservation]
            One observation per visible AP.
        """
        # TODO: implement real capture.
        #   RSSI:  `subprocess.run(["iw", "dev", self.interface, "scan"])`
        #          or `scapy.all.sniff(prn=...)`
        #   CSI:   nexmon csi tool → stdout parse
        #          or ESP32-csi-tool over UDP
        if self.mode == "rssi":
            # Simulate 3 visible APs
            mock_aps = [
                {"bssid": "AA:BB:CC:DD:EE:01", "ssid": "eduroam", "rssi": random.uniform(-75, -45), "freq": 5180, "channel": 36},  # noqa: E501
                {"bssid": "AA:BB:CC:DD:EE:02", "ssid": "IoT-Net", "rssi": random.uniform(-85, -50), "freq": 2412, "channel": 1},  # noqa: E501
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
                        metadata={
                            "interface": self.interface,
                            "mode": self.mode,
                            "channel": ap["channel"],
                        },
                    )
                )
            return results

        # CSI mode placeholder
        return []

    # ------------------------------------------------------------------
    # TODO: add helpers for:
    #   - iw scan parsing (RSSI)
    #   - nexmon csi tool binary invocation and output parsing (CSI)
    #   - ESP32-csi-tool UDP listener (CSI)
    #   - Channel hopping if monitoring multiple frequencies
    # ------------------------------------------------------------------


def _test_main() -> None:
    parser = argparse.ArgumentParser(description="WiFi sensor self-test")
    parser.add_argument("--interface", default="wlan0")
    parser.add_argument("--mode", choices=["rssi", "csi"], default="rssi")
    args = parser.parse_args()

    sensor = WiFiSensor(sensor_id="wifi-self-test", interface=args.interface, mode=args.mode)
    obs_list = sensor.read()
    for obs in obs_list:
        print(f"[{obs.timestamp.isoformat()}] {obs.sensor_type}/{obs.sensor_id}")
        bssid = obs.observation.get("bssid", "?")
        rssi = obs.observation.get("rssi", "?")
        print(f"  AP={bssid}  RSSI={rssi} dBm")


if __name__ == "__main__":
    _test_main()
