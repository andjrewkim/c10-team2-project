#!/usr/bin/env python3
"""End-to-end demo: mock sensor → fusion → action.

Uses MlFusion if ``fusion/model.pkl`` exists; otherwise falls back
to WeightedAverageFusion.  No real hardware or MQTT broker required.

Run with:
    python run_demo.py
"""

import time

from sensors.mock_sensor import MockSensor
from actions.console_action import ConsoleAction


def _get_fuser():
    try:
        from fusion.ml_fusion import MlFusion
        return MlFusion()
    except ImportError:
        from fusion.weighted_average import WeightedAverageFusion
        print("[demo] MlFusion unavailable — using WeightedAverageFusion")
        return WeightedAverageFusion()


def main() -> None:
    print("=" * 50)
    print("IoT Activity Detection — Demo Pipeline")
    print("=" * 50)

    sensor = MockSensor(sensor_id="demo-01")
    fuser = _get_fuser()
    action = ConsoleAction(min_confidence=0.3)

    for cycle in range(5):
        print(f"\n--- Cycle {cycle + 1} ---")

        observations = sensor.read()
        print(f"  Sensor produced {len(observations)} observation(s)")

        fused = fuser.fuse(observations)
        print(f"  Fused: label={fused.activity_label}  confidence={fused.confidence:.4f}")

        result = action.evaluate(fused)
        if result is not None:
            print(f"  Action fired: {result.action_name}")
        else:
            print("  Action suppressed (below threshold)")

        time.sleep(0.5)

    print("\nDone.")


if __name__ == "__main__":
    main()
