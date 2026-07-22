#!/usr/bin/env python3
"""End-to-end demo: mock sensor → fusion → action.

Run with:
    python run_demo.py

No real hardware, MQTT broker, or external dependencies beyond
what pip install provides.
"""

import time

from sensors.mock_sensor import MockSensor
from fusion.weighted_average import WeightedAverageFusion
from actions.console_action import ConsoleAction


def main() -> None:
    print("=" * 50)
    print("IoT Activity Detection — Demo Pipeline")
    print("=" * 50)

    sensor = MockSensor(sensor_id="demo-01")
    fuser = WeightedAverageFusion(type_weights={"mock": 1.0})
    action = ConsoleAction(min_confidence=0.3)

    for cycle in range(5):
        print(f"\n--- Cycle {cycle + 1} ---")

        observations = sensor.read()
        print(f"  Sensor produced {len(observations)} observation(s)")

        fused = fuser.fuse(observations)
        print(f"  Fused confidence: {fused.confidence:.4f}")

        result = action.evaluate(fused)
        if result is not None:
            print(f"  Action fired: {result.action_name}")
        else:
            print("  Action suppressed (below threshold)")

        time.sleep(0.5)

    print("\nDone.")


if __name__ == "__main__":
    main()
