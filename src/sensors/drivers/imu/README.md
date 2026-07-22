# IMU Driver

**Hardware expected**: I2C inertial measurement unit (accelerometer +
gyroscope, optionally magnetometer). Tested with BNO055; should work
with MPU6050, ICM-20948, or similar.

## Wiring

| IMU Pin | Raspberry Pi GPIO |
|---------|-------------------|
| VCC     | 3.3 V             |
| GND     | GND               |
| SDA     | GPIO 2 (SDA)      |
| SCL     | GPIO 3 (SCL)      |

## Standalone test

```bash
python -m sensors.drivers.imu.driver --bus 1 --address 0x28
```

This performs one read cycle and prints the result. Connect the IMU
before running.

## What still needs implementing

- I2C bus open/close in `__init__` / `__del__`
- Register read helper (e.g. `_read_registers(reg, length)`)
- Real register map and scaling factors for the chosen IMU
- Error handling (bus errors, device not found)
