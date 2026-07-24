from __future__ import annotations

# Make `from src.*` imports resolve no matter how the script is launched:
# `python src/debug_3d.py` from the project root, `python -m src.debug_3d`,
# or `cd src && python debug_3d.py` should all just work — without requiring
# the user to remember to set PYTHONPATH or `cd` to the right directory.
import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import time
from collections import deque
from pathlib import Path

import numpy as np

from src.sensors.imu_reader import ImuReader
from src.sensors.mmwave_reader import MmWaveReader
from src.sensors.lab_integration.posture_lab_common import filter_posture_points
from src.sensors.lab_integration.imu import (
    quat_multiply,
    quat_from_angular_velocity,
    quat_rotate,
    quat_normalized,
)


SENSOR_REGISTRY = {
    "mmwave": MmWaveReader,
    "imu": ImuReader,
}


class WristTracker:
    """Dead-reckoned wrist position from IMU on right wrist (streaming)."""
    
    def __init__(self, gravity=np.array([0.0, 0.0, -9.81]), dt_nominal=0.1):
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.orientation = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z
        self.gravity = gravity
        self.initialized = False
        self.trajectory = deque(maxlen=200)
        self.trajectory.append(self.position.copy())
        self.dt_nominal = dt_nominal
        
    def update(self, accel, gyro, dt):
        """Update position using IMU integration (streaming version)."""
        accel = np.array(accel, dtype=float)
        gyro = np.array(gyro, dtype=float)
        
        if not self.initialized:
            self.initialized = True
            return self.position.copy()
            
        # Gyro integration: orientation update
        # gyro is in rad/s (mock sensor) or deg/s (real) - assume rad/s for now
        delta_q = quat_from_angular_velocity(gyro, dt)
        self.orientation = quat_normalized(quat_multiply(self.orientation, delta_q))
        
        # Accelerometer: rotate to world frame, subtract gravity
        specific_force_world = quat_rotate(self.orientation, accel)
        linear_accel = specific_force_world - self.gravity
        
        # Deadband to reduce drift
        if np.linalg.norm(linear_accel) < 0.5:
            linear_accel = np.zeros(3)
        
        # Simple velocity/position integration
        self.velocity += linear_accel * dt
        # Velocity damping to prevent drift
        self.velocity *= 0.98
        self.position += self.velocity * dt
        
        self.trajectory.append(self.position.copy())
        return self.position.copy()
    
    def get_trajectory(self):
        return np.array(self.trajectory)


def extract_mmwave_points(reading):
    """Extract point cloud from mmWave reading."""
    data = reading.data
    points = data.get("points", [])
    if not points:
        return np.zeros((0, 3))
    return np.array([[p.get("x", 0), p.get("y", 0), p.get("z", 0)] for p in points])


def extract_human_centroid(points):
    """Extract human body centroid from point cloud using posture filtering."""
    if len(points) == 0:
        return np.array([0.0, 0.0, 0.0])
    filtered = filter_posture_points(points, {
        "x_limit_m": 2.0,
        "min_range_m": 0.3,
        "max_range_m": 2.5,
    })
    if len(filtered) < 5:
        return np.array([0.0, 0.0, 0.0])
    centroid_2d = np.mean(filtered[:, :2], axis=0)
    mean_z = np.mean(filtered[:, 2])
    return np.array([centroid_2d[0], centroid_2d[1], mean_z])


def load_model(model_path):
    import pickle
    with open(model_path, "rb") as f:
        return pickle.load(f)


def extract_features_for_inference(window, sensor_types):
    """Same feature extraction as realtime_demo.py"""
    from src.realtime_demo import extract_features_from_reading
    import numpy as np
    
    features = []
    for name in sensor_types:
        readings = [f[name] for f in window if name in f]
        sensor_feats = []
        for r in readings:
            feats = extract_features_from_reading(r, sensor_types[name])
            sensor_feats.append(feats)
        if sensor_feats:
            sensor_feats = np.array(sensor_feats)
            features.extend(np.mean(sensor_feats, axis=0).tolist())
        else:
            n_feats = 10 if sensor_types[name] == "mmwave" else 6
            features.extend([0.0] * n_feats)
    return np.array(features).reshape(1, -1)


def main():
    parser = argparse.ArgumentParser(description="3D Debug Visualizer: mmWave + IMU Wrist + Predictions")
    parser.add_argument("--model", default="models/best_model.pkl", help="Trained model path")
    parser.add_argument("--features", default="data/processed/features.npz", help="Features NPZ for gesture labels")
    parser.add_argument("--mode", default="mock", choices=["mock", "serial"], help="Sensor mode")
    parser.add_argument("--window", type=int, default=10, help="Window size for inference")
    parser.add_argument("--stride", type=int, default=5, help="Stride between predictions")
    parser.add_argument("--no-predict", action="store_true", help="Disable prediction, just visualize")
    args = parser.parse_args()

    # Load model and gesture labels
    pipeline = None
    gestures = []
    if not args.no_predict:
        model_path = Path(args.model)
        if model_path.exists():
            pipeline = load_model(model_path)
            features_path = Path(args.features)
            if features_path.exists():
                data = np.load(features_path, allow_pickle=True)
                gestures = data["gestures"].tolist() if "gestures" in data else []
            print(f"Loaded model: {model_path}")
            if gestures:
                print(f"Gestures: {', '.join(gestures)}")

    # Initialize sensors
    mmwave = MmWaveReader(mode=args.mode)
    imu = ImuReader(mode=args.mode)
    mmwave.start()
    imu.start()
    print(f"Started mmWave ({args.mode}) and IMU ({args.mode})")

    # Trackers
    wrist_tracker = WristTracker()
    frame_buffer = deque(maxlen=args.window)
    frame_count = 0
    predict_count = 0
    
    # For visualization
    # Pick the best available backend for this platform. MacOSX is the native
    # macOS backend (smooth 3D rendering); TkAgg is the cross-platform fallback.
    # Setting TkAgg on macOS commonly results in a blank window because the
    # event loop never has enough time to commit the first frame.
    import matplotlib
    backend_errors = (ImportError, RuntimeError)
    try:
        matplotlib.use("MacOSX")
    except backend_errors:
        try:
            matplotlib.use("TkAgg")
        except backend_errors:
            print(
                f"[debug_3d] warning: MacOSX/TkAgg backends unavailable, "
                f"using matplotlib's default '{matplotlib.get_backend()}'"
            )
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    plt.ion()
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # ---- Static axes setup (run ONCE; not in the per-frame loop) ----
    # These would otherwise be reset to defaults every iteration, which is
    # what prevents the user from rotating / zooming the view.
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Debug: mmWave Point Cloud + Right Wrist IMU + Gesture Prediction')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-1, 2)
    # Radar perspective: camera at origin looking along +Z boresight.
    ax.view_init(elev=0, azim=180)

    # Static origin coordinate frame (radar boresight).
    _axis_len = 0.5
    ax.plot([0, _axis_len], [0, 0], [0, 0], 'r-', alpha=0.5)
    ax.plot([0, 0], [0, _axis_len], [0, 0], 'g-', alpha=0.5)
    ax.plot([0, 0], [0, 0], [0, _axis_len], 'b-', alpha=0.5)

    # Persistent dynamic artists — their data is updated in place each frame
    # instead of being recreated. This eliminates ax.clear() per frame so the
    # axis view / limits / labels stay sticky while the user interacts.
    mmwave_scatter = ax.scatter([], [], [], c='blue', alpha=0.4, s=15,
                                label='mmWave Points')
    centroid_scatter = ax.scatter([], [], [], c='red', s=200, marker='*',
                                  label='Human Centroid (Radar)',
                                  edgecolors='black', linewidth=1)
    trajectory_line, = ax.plot([], [], [], c='green', alpha=0.7, linewidth=2,
                               label='Wrist Trajectory (IMU)')
    wrist_marker = ax.scatter([], [], [], c='green', s=120, marker='o',
                              label='Wrist Position',
                              edgecolors='black', linewidth=1)
    ax.legend(loc='upper right', fontsize=9)

    # Figure-level text overlays (these would be wiped by ax.clear() if they
    # were attached to the axes; fig.text keeps them across the whole window).
    pred_text = fig.text(0.02, 0.95, "", fontsize=12, transform=fig.transFigure,
                         bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    info_text = fig.text(0.02, 0.02, "", fontsize=10, transform=fig.transFigure,
                         bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8))
    
    last_time = time.monotonic()
    
    try:
        while True:
            current_time = time.monotonic()
            dt = current_time - last_time
            last_time = current_time
            
            # Read sensors
            mmwave_reading = mmwave.read()
            imu_reading = imu.read()
            
            frame_data = {"mmwave": mmwave_reading, "imu": imu_reading}
            frame_buffer.append(frame_data)
            frame_count += 1
            
            # --- Update wrist tracker ---
            imu_data = imu_reading.data
            accel = imu_data.get("accel", [0, 0, 0])
            gyro = imu_data.get("gyro", [0, 0, 0])
            wrist_pos = wrist_tracker.update(accel, gyro, dt)
            
            # --- Run prediction periodically ---
            pred_label = ""
            pred_conf = 0.0
            if pipeline and len(frame_buffer) >= args.window and frame_count % args.stride == 0:
                sensor_types = {"mmwave": "mmwave", "imu": "imu"}
                X = extract_features_for_inference(list(frame_buffer), sensor_types)
                pred = pipeline.predict(X)[0]
                proba = pipeline.predict_proba(X)[0]
                pred_conf = float(max(proba))
                pred_label = gestures[pred] if pred < len(gestures) else str(pred)
                predict_count += 1
            
            # --- Get mmWave points and human centroid ---
            points = extract_mmwave_points(mmwave_reading)
            human_centroid = extract_human_centroid(points)
            
            # --- Visualization update (in-place; no ax.clear()) ---
            # Updating artists in place instead of clearing + redrawing keeps
            # the user's interactive view (rotation/zoom/pan) intact and is
            # noticeably cheaper per frame.

            # 1. mmWave point cloud
            if len(points) > 0:
                mmwave_scatter._offsets3d = (points[:, 0], points[:, 1], points[:, 2])
            else:
                mmwave_scatter._offsets3d = (np.array([]), np.array([]), np.array([]))

            # 2. Human centroid from radar
            if np.any(human_centroid != 0):
                centroid_scatter._offsets3d = (
                    np.array([human_centroid[0]]),
                    np.array([human_centroid[1]]),
                    np.array([human_centroid[2]]),
                )
            else:
                centroid_scatter._offsets3d = (np.array([]), np.array([]), np.array([]))

            # 3. Wrist trajectory (IMU dead-reckoning) + current wrist marker
            traj = wrist_tracker.get_trajectory()
            if len(traj) > 1:
                trajectory_line.set_data_3d(traj[:, 0], traj[:, 1], traj[:, 2])
                wrist_marker._offsets3d = (
                    np.array([wrist_pos[0]]),
                    np.array([wrist_pos[1]]),
                    np.array([wrist_pos[2]]),
                )
            else:
                trajectory_line.set_data_3d([], [], [])
                wrist_marker._offsets3d = (np.array([]), np.array([]), np.array([]))

            # Prediction text
            if pred_label:
                pred_text.set_text(f"Prediction: {pred_label}  (confidence: {pred_conf:.2f})")
            else:
                pred_text.set_text("Waiting for prediction...")

            # Info text
            info_text.set_text(
                f"Frames: {frame_count} | Predictions: {predict_count} | "
                f"mmWave pts: {len(points)} | Wrist: [{wrist_pos[0]:.2f}, {wrist_pos[1]:.2f}, {wrist_pos[2]:.2f}]"
            )

            # Force a backend draw and flush pending GUI events so the canvas
            # actually updates. plt.pause(0.01) alone is too short on macOS
            # TkAgg and leaves the window visually blank.
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)
            
    except KeyboardInterrupt:
        print("\nVisualization stopped.")
    finally:
        mmwave.stop()
        imu.stop()
        plt.ioff()
        plt.close()


if __name__ == "__main__":
    main()