# WiFi Sensor Driver

**Hardware expected**: Any 802.11 wireless interface capable of
monitor mode. Two modes:

- **RSSI** — works with any interface and `iw` (Linux) or `airport`
  (macOS). No special firmware needed.
- **CSI** — requires monitor-mode firmware and tools like nexmon
  (Broadcom) or ESP32-csi-tool.

## Standalone test

```bash
python -m sensors.drivers.wifi.driver --interface wlan0 --mode rssi
```

## What still needs implementing

- RSSI: `iw dev <iface> scan` parsing (`subprocess` + regex)
- RSSI: `scapy` probe-request capture (passive)
- CSI: nexmon CSI tool invocation and binary output parser
- CSI: ESP32 UDP packet parser (amplitude/phase per subcarrier)
- Channel hopping logic for multi-channel monitoring
- macOS `airport` / `wakelan` fallback
