# UWB Anchor Driver

**Hardware expected**: UWB ranging module (DW1000, DWM1001, Decawave)
operating as an anchor with known fixed position. Communicates over
serial UART.

## Wiring

| UWB Module | Computer      |
|------------|---------------|
| TX         | RX (serial)   |
| RX         | TX (serial)   |
| VCC        | 3.3 V / 5 V   |
| GND        | GND            |

## Standalone test

```bash
python -m sensors.drivers.uwb.driver --port /dev/ttyACM0 --baud 115200
```

## What still needs implementing

- Serial port open/close with `pyserial`
- The specific command/response protocol of the chosen UWB module
- Parsing of range, RSSI, and NLoS flags from raw bytes
- Multi-tag scheduling if the anchor polls multiple tags per cycle
