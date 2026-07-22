# mmWave Radar Driver

**Hardware expected**: TI IWR6843 / IWR1843 mmWave radar sensor
connected over dual UART (data port + CLI port). Tested with
IWR6843ISK EVM.

## Wiring

| Radar Pin | Computer     |
|-----------|--------------|
| DATA TX   | RX (USB0)    |
| DATA RX   | TX (USB0)    |
| CLI TX    | RX (USB1)    |
| CLI RX    | TX (USB1)    |
| VCC       | 5 V (USB)    |
| GND       | GND          |

## Standalone test

```bash
python -m sensors.drivers.mmwave.driver --port /dev/ttyUSB0 --cli-port /dev/ttyUSB1
```

## What still needs implementing

- Dual serial port management (data + CLI)
- TI TLV packet parsing (magic word `0x7080_6070`, header, TLV items)
- CLI configuration sequence (`sensorStop`, `flushCfg`, `sensorStart`)
- Frame header timestamp and CRC validation
- Point cloud vs. range-doppler mode selection
