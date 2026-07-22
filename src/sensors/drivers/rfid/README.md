# RFID Reader Driver

**Hardware expected**: UHF RFID reader (Impinj Speedway R420 / R700,
ThingMagic M6e, Zebra FX9600) with one or more antennas. The reader
communicates over TCP/IP (LLRP or vendor SDK).

## Wiring / Network

- Reader connects via Ethernet to the same LAN as the recording laptop.
- Assign a static IP to the reader or use hostname resolution.
- Connect antennas to reader ports.

## Standalone test

```bash
python -m sensors.drivers.rfid.driver --host 192.168.1.100 --port 5084
```

## What still needs implementing

- TCP socket management with automatic reconnection
- LLRP message building (RO_ACCESS_SPEC, RO_REPORT, etc.)
- Impinj/ThingMagic vendor-extension parsing
- Tag-report deduplication and timestamps
- Antenna-port cycling if multiple antennas are used
