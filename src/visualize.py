from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _safe_save(fig, gesture: str, sensor: str, mode: str, session_ts: str | None) -> None:
    ts = session_ts or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path("results") / f"{sensor}_figures"
    name = f"{gesture}_{sensor}_{mode}_{ts}.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_dir / name), dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_dir / name}")


def load_jsonl(path: Path) -> list[dict]:
    frames = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                frames.append(json.loads(line))
    return frames


def load_session_csvs(session_dir: Path) -> dict[str, list[dict]]:
    events_path = session_dir / "events.csv"
    if not events_path.exists():
        raise FileNotFoundError(f"No events.csv in {session_dir}")

    sensor_csvs = sorted(
        p for p in session_dir.glob("*.csv")
        if p.name not in ("events.csv", "trials.csv")
    )

    events_rows: list[dict] = []
    with open(events_path, newline="") as f:
        for row in csv.DictReader(f):
            events_rows.append(row)

    sensor_rows: dict[str, list[dict]] = {}
    for csv_path in sensor_csvs:
        name = csv_path.stem
        rows = []
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                parsed: dict = {}
                for k, v in row.items():
                    if k == "frame_index":
                        continue
                    if v == "" or v == "null":
                        parsed[k] = None
                    elif v.startswith(("[", "{")):
                        try:
                            parsed[k] = json.loads(v)
                        except json.JSONDecodeError:
                            parsed[k] = v
                    else:
                        try:
                            parsed[k] = int(v)
                        except ValueError:
                            try:
                                parsed[k] = float(v)
                            except ValueError:
                                parsed[k] = v
                rows.append(parsed)
        sensor_rows[name] = rows

    frames_by_key: dict[str, list[dict]] = {}
    for idx, ev in enumerate(events_rows):
        frame: dict = {
            "timestamp": ev["timestamp"],
            "gesture": ev["gesture"],
            "trial": int(ev["trial"]),
            "elapsed": float(ev["elapsed"]),
        }
        if "collector" in ev and ev["collector"]:
            frame["collector"] = ev["collector"]
        for sname, sdict_list in sensor_rows.items():
            if idx < len(sdict_list):
                sdict = dict(sdict_list[idx])
                confidence = sdict.pop("confidence", 0.0)
                frame[sname] = {
                    "data": sdict,
                    "confidence": confidence,
                    "sensor_type": sname,
                }
        key = f"{ev['gesture']}_t{ev['trial']}"
        frames_by_key.setdefault(key, []).append(frame)

    return frames_by_key


def print_stats(frames: list[dict]) -> None:
    gestures = {}
    for f in frames:
        g = f.get("gesture", "?")
        gestures[g] = gestures.get(g, 0) + 1

    print(f"\nTotal frames: {len(frames)}")
    print(f"Gestures: {gestures}")
    sensors_in_frame = [k for k in frames[0].keys() if k not in ("timestamp", "gesture", "trial", "elapsed")]
    print(f"Sensors: {sensors_in_frame}")
    print()


def show_mmwave(frames: list[dict], max_frames: int = 200, session_ts: str | None = None) -> None:
    import matplotlib.pyplot as plt

    n = min(len(frames), max_frames)
    ts = np.arange(n)
    xs, ys, npts = [], [], []
    x_spans, y_spans = [], []
    all_pts_x, all_pts_y = [], []
    active_count = 0
    for f in frames[:n]:
        mm = f.get("mmwave", {})
        d = mm.get("data", {})
        pts = d.get("points", [])
        npts.append(len(pts))
        if pts:
            active_count += 1
            px = np.array([p["x"] for p in pts])
            py = np.array([p["y"] for p in pts])
            xs.append(np.mean(px))
            ys.append(np.mean(py))
            x_spans.append(float(np.max(px) - np.min(px)))
            y_spans.append(float(np.max(py) - np.min(py)))
            all_pts_x.extend(px.tolist())
            all_pts_y.extend(py.tolist())
        else:
            xs.append(0)
            ys.append(0)
            x_spans.append(0)
            y_spans.append(0)

    active_frac = active_count / n if n > 0 else 0.0

    fig, axs = plt.subplots(2, 3, figsize=(16, 8))

    axs[0, 0].plot(ts, xs, label="x", alpha=0.7)
    axs[0, 0].plot(ts, ys, label="y", alpha=0.7)
    axs[0, 0].set_ylabel("Centroid (m)")
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].set_title("Centroid x / y")

    axs[0, 1].plot(ts, x_spans, label="x span (width)", color="purple", alpha=0.7)
    axs[0, 1].plot(ts, y_spans, label="y span (depth)", color="brown", alpha=0.7)
    axs[0, 1].set_ylabel("Span (m)")
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].set_title("Body width (x span) / Body depth (y span)")

    axs[0, 2].plot(ts, npts, color="green")
    axs[0, 2].set_ylabel("Point count")
    axs[0, 2].grid(True, alpha=0.3)
    axs[0, 2].set_title("Point count")

    axs[1, 0].bar(["Active"], [active_frac], color="steelblue", width=0.4)
    axs[1, 0].set_ylim(0, 1)
    axs[1, 0].set_ylabel("Fraction")
    axs[1, 0].grid(True, axis="y", alpha=0.3)
    axs[1, 0].set_title(f"Active frame fraction = {active_frac:.3f}")

    if all_pts_x and all_pts_y:
        h = axs[1, 1].hist2d(all_pts_x, all_pts_y, bins=(40, 40), cmap="hot")
        axs[1, 1].set_xlabel("x (m)")
        axs[1, 1].set_ylabel("y (m)")
        fig.colorbar(h[3], ax=axs[1, 1], label="Point density")
    else:
        axs[1, 1].text(0.5, 0.5, "No points", ha="center", va="center", transform=axs[1, 1].transAxes)
    axs[1, 1].set_title("2D occupancy heatmap")

    axs[1, 2].set_visible(False)

    gesture = frames[0].get("gesture", "?")
    fig.suptitle(f"mmWave — {gesture} (trial {frames[0].get('trial', 0)})")
    _safe_save(fig, gesture, "mmwave", "mmwave", session_ts)
    plt.tight_layout()
    plt.show()


def show_imu(frames: list[dict], max_frames: int = 200, session_ts: str | None = None) -> None:
    import matplotlib.pyplot as plt

    n = min(len(frames), max_frames)
    ts = np.arange(n)
    axs, ays, azs = [], [], []
    gxs, gys, gzs = [], [], []
    for f in frames[:n]:
        im = f.get("imu", {})
        d = im.get("data", {})
        acc = d.get("accel", None)
        gyr = d.get("gyro", None)
        if isinstance(acc, list):
            axs.append(acc[0] if len(acc) > 0 else 0.0)
            ays.append(acc[1] if len(acc) > 1 else 0.0)
            azs.append(acc[2] if len(acc) > 2 else 0.0)
        else:
            axs.append(d.get("accel_x", 0.0))
            ays.append(d.get("accel_y", 0.0))
            azs.append(d.get("accel_z", 0.0))
        if isinstance(gyr, list):
            gxs.append(gyr[0] if len(gyr) > 0 else 0.0)
            gys.append(gyr[1] if len(gyr) > 1 else 0.0)
            gzs.append(gyr[2] if len(gyr) > 2 else 0.0)
        else:
            gxs.append(d.get("gyro_x", 0.0))
            gys.append(d.get("gyro_y", 0.0))
            gzs.append(d.get("gyro_z", 0.0))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax1.plot(ts, axs, label="accel x")
    ax1.plot(ts, ays, label="accel y")
    ax1.plot(ts, azs, label="accel z")
    ax1.set_ylabel("Accel (m/s²)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(ts, gxs, label="gyro x")
    ax2.plot(ts, gys, label="gyro y")
    ax2.plot(ts, gzs, label="gyro z")
    ax2.set_ylabel("Gyro (rad/s)")
    ax2.set_xlabel("Frame")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    gesture = frames[0].get("gesture", "?")
    fig.suptitle(f"IMU — {gesture} (trial {frames[0].get('trial', 0)})")
    _safe_save(fig, gesture, "imu", "imu", session_ts)
    plt.tight_layout()
    plt.show()


def show_uwb(frames: list[dict], max_frames: int = 200, session_ts: str | None = None) -> None:
    import matplotlib.pyplot as plt

    n = min(len(frames), max_frames)
    ts = np.arange(n)
    tag1_dists, tag2_dists = np.zeros(n), np.zeros(n)
    for i, f in enumerate(frames[:n]):
        uwb = f.get("uwb", {})
        d = uwb.get("data", {})
        ranges = d.get("ranges_cm", [])
        if len(ranges) > 0 and ranges[0] is not None:
            tag1_dists[i] = float(ranges[0]) / 100.0
        if len(ranges) > 1 and ranges[1] is not None:
            tag2_dists[i] = float(ranges[1]) / 100.0

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ts, tag1_dists, label="Tag 1 distance", alpha=0.8)
    ax.plot(ts, tag2_dists, label="Tag 2 distance", alpha=0.8)
    ax.set_ylabel("Distance (m)")
    ax.set_xlabel("Frame")
    ax.legend()
    ax.grid(True, alpha=0.3)

    gesture = frames[0].get("gesture", "?")
    fig.suptitle(f"UWB — {gesture} (trial {frames[0].get('trial', 0)})")
    _safe_save(fig, gesture, "uwb", "uwb", session_ts)
    plt.tight_layout()
    plt.show()


def extract_mmwave_timeseries(frames: list[dict], max_frames: int = 200) -> dict[str, np.ndarray]:
    n = min(len(frames), max_frames)
    xs, ys, zs, vs, npts = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    x_spans, y_spans = np.zeros(n), np.zeros(n)
    for i, f in enumerate(frames[:n]):
        mm = f.get("mmwave", {})
        d = mm.get("data", {})
        pts = d.get("points", [])
        npts[i] = len(pts)
        if pts:
            px = np.array([p["x"] for p in pts])
            py = np.array([p["y"] for p in pts])
            pz = np.array([p.get("z", 0) for p in pts])
            pv = np.array([abs(p.get("velocity", 0)) for p in pts])
            xs[i] = np.mean(px)
            ys[i] = np.mean(py)
            zs[i] = np.mean(pz)
            vs[i] = np.mean(pv)
            x_spans[i] = float(np.max(px) - np.min(px))
            y_spans[i] = float(np.max(py) - np.min(py))
    return {
        "centroid_x": xs, "centroid_y": ys, "centroid_z": zs,
        "mean_vel": vs, "num_points": npts,
        "body_width": x_spans, "body_depth": y_spans,
    }


def extract_imu_timeseries(frames: list[dict], max_frames: int = 200) -> dict[str, np.ndarray]:
    n = min(len(frames), max_frames)
    ax, ay, az = np.zeros(n), np.zeros(n), np.zeros(n)
    gx, gy, gz = np.zeros(n), np.zeros(n), np.zeros(n)
    for i, f in enumerate(frames[:n]):
        im = f.get("imu", {})
        d = im.get("data", {})
        acc = d.get("accel", None)
        gyr = d.get("gyro", None)
        if isinstance(acc, list):
            ax[i] = float(acc[0]) if len(acc) > 0 else 0.0
            ay[i] = float(acc[1]) if len(acc) > 1 else 0.0
            az[i] = float(acc[2]) if len(acc) > 2 else 0.0
        else:
            ax[i] = float(d.get("accel_x", 0.0))
            ay[i] = float(d.get("accel_y", 0.0))
            az[i] = float(d.get("accel_z", 0.0))
        if isinstance(gyr, list):
            gx[i] = float(gyr[0]) if len(gyr) > 0 else 0.0
            gy[i] = float(gyr[1]) if len(gyr) > 1 else 0.0
            gz[i] = float(gyr[2]) if len(gyr) > 2 else 0.0
        else:
            gx[i] = float(d.get("gyro_x", 0.0))
            gy[i] = float(d.get("gyro_y", 0.0))
            gz[i] = float(d.get("gyro_z", 0.0))
    return {
        "accel_x": ax, "accel_y": ay, "accel_z": az,
        "gyro_x": gx, "gyro_y": gy, "gyro_z": gz,
    }


def extract_uwb_timeseries(frames: list[dict], max_frames: int = 200) -> dict[str, np.ndarray]:
    n = min(len(frames), max_frames)
    tag1, tag2 = np.zeros(n), np.zeros(n)
    for i, f in enumerate(frames[:n]):
        uwb = f.get("uwb", {})
        d = uwb.get("data", {})
        ranges = d.get("ranges_cm", [])
        if len(ranges) > 0 and ranges[0] is not None:
            tag1[i] = float(ranges[0]) / 100.0
        if len(ranges) > 1 and ranges[1] is not None:
            tag2[i] = float(ranges[1]) / 100.0
    return {"tag1_distance_m": tag1, "tag2_distance_m": tag2}


def _extract_timeseries(frames: list[dict], sensor: str, max_frames: int = 200) -> dict[str, np.ndarray]:
    if sensor == "mmwave":
        return extract_mmwave_timeseries(frames, max_frames)
    elif sensor == "imu":
        return extract_imu_timeseries(frames, max_frames)
    elif sensor == "uwb":
        return extract_uwb_timeseries(frames, max_frames)
    return {}


SENSOR_FEATURES = {
    "mmwave": ["centroid_x", "centroid_y", "num_points", "body_width", "body_depth"],
    "imu": ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"],
    "uwb": ["tag1_distance_m", "tag2_distance_m"],
}


def _resolve_features(features_arg: str, sensor: str) -> list[str]:
    all_feats = SENSOR_FEATURES.get(sensor, [])
    if features_arg == "all":
        return all_feats
    return [features_arg]


def _resample_to_target(values: np.ndarray, target: int) -> np.ndarray:
    if len(values) == 0:
        return np.zeros(target)
    if len(values) == 1:
        return np.full(target, values[0])
    return np.interp(np.linspace(0, len(values) - 1, target), np.arange(len(values)), values)


def _filter_by_gesture(frames_by_file: dict[str, list[dict]], gestures: list[str] | None):
    if not gestures:
        return frames_by_file
    gesture_set = set(gestures)
    filtered = {}
    for key, frames in frames_by_file.items():
        if frames and frames[0].get("gesture") in gesture_set:
            filtered[key] = frames
    return filtered


def compare_gestures(frames_by_trial: dict[str, list[dict]], sensor: str, features: list[str], max_frames: int = 100, session_ts: str | None = None) -> None:
    import matplotlib.pyplot as plt

    n_features = len(features)
    cols = 3
    rows = (n_features + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows), constrained_layout=True)
    axs = axs.flatten() if isinstance(axs, np.ndarray) else [axs]

    for idx, feat in enumerate(features):
        ax = axs[idx]
        by_gesture: dict[str, list[np.ndarray]] = {}
        for key, frames in frames_by_trial.items():
            g = frames[0].get("gesture", "?")
            ts = _extract_timeseries(frames, sensor, max_frames)
            vals = ts.get(feat, np.zeros(max_frames))
            by_gesture.setdefault(g, []).append(vals)

        for gesture, trials in sorted(by_gesture.items()):
            max_len = min(max(len(t) for t in trials), max_frames)
            aligned = np.array([_resample_to_target(t, max_len) for t in trials])
            mean_curve = np.mean(aligned, axis=0)
            std_curve = np.std(aligned, axis=0)
            t_axis = np.arange(max_len)
            ax.plot(t_axis, mean_curve, label=gesture, linewidth=2)
            ax.fill_between(t_axis, mean_curve - std_curve, mean_curve + std_curve, alpha=0.12)

        ax.set_title(feat)
        ax.set_xlabel("Frame (resampled)")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

    for idx in range(n_features, len(axs)):
        axs[idx].set_visible(False)

    feat_label = "_".join(features) if n_features > 1 else features[0]
    fig.suptitle(f"{sensor} feature comparison ({feat_label}) — mean ± std across trials")
    _safe_save(fig, "all_gestures", sensor, f"compare_{feat_label}", session_ts)
    if rows > 3:
        print(f"  Figure too large for screen — open saved PNG to scroll")
        plt.close(fig)
    else:
        plt.show()


def consistency_gesture(frames_by_trial: dict[str, list[dict]], sensor: str, gesture: str, features: list[str], max_frames: int = 200, session_ts: str | None = None) -> None:
    import matplotlib.pyplot as plt

    trials = [f for key, f in frames_by_trial.items() if f[0].get("gesture", "?") == gesture]
    if not trials:
        print(f"No trials found for gesture '{gesture}'")
        return

    n_features = len(features)
    cols = 3
    rows = (n_features + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(16, 4 * rows), constrained_layout=True)
    axs = axs.flatten() if isinstance(axs, np.ndarray) else [axs]

    for idx, feat in enumerate(features):
        ax = axs[idx]
        all_aligned = []
        for t_idx, frames in enumerate(trials):
            ts = _extract_timeseries(frames, sensor, max_frames)
            vals = ts.get(feat, np.zeros(max_frames))
            aligned = _resample_to_target(vals, max_frames)
            all_aligned.append(aligned)
            ax.plot(aligned, alpha=0.35, linewidth=1, label=f"trial {t_idx + 1}" if n_features == 1 and len(trials) <= 10 else None)

        if all_aligned:
            mean_curve = np.mean(all_aligned, axis=0)
            std_curve = np.std(all_aligned, axis=0)
            t_axis = np.arange(max_frames)
            ax.plot(t_axis, mean_curve, color="black", linewidth=2.5, label="mean")
            ax.fill_between(t_axis, mean_curve - std_curve, mean_curve + std_curve, color="black", alpha=0.1)
        ax.set_title(feat)
        ax.set_xlabel("Frame")
        ax.grid(True, alpha=0.3)

    for idx in range(n_features, len(axs)):
        axs[idx].set_visible(False)

    if n_features == 1 and len(trials) <= 10:
        axs[0].legend(fontsize=7)

    fig.suptitle(f"{sensor} consistency — {gesture} ({len(trials)} trials)")
    _safe_save(fig, gesture, sensor, f"consistency_{'_'.join(features)}", session_ts)
    if rows > 3:
        print(f"  Figure too large for screen — open saved PNG to scroll")
        plt.close(fig)
    else:
        plt.show()


def consistency_multi_gestures(frames_by_trial: dict[str, list[dict]], sensor: str, features: list[str], max_frames: int = 200, session_ts: str | None = None) -> None:
    import matplotlib.pyplot as plt

    by_gesture: dict[str, list[list[dict]]] = {}
    for key, frames in frames_by_trial.items():
        g = frames[0].get("gesture", "?")
        by_gesture.setdefault(g, []).append(frames)

    if not by_gesture:
        print(f"No gestures with {sensor} data found")
        return

    gestures_sorted = sorted(by_gesture.keys())
    n_gestures = len(gestures_sorted)
    n_features = len(features)
    fig, axs = plt.subplots(n_gestures, n_features, figsize=(8 * n_features + 2, 3 * n_gestures), constrained_layout=True)
    if n_gestures == 1 and n_features == 1:
        axs = np.array([[axs]])
    elif n_gestures == 1:
        axs = axs.reshape(1, -1)
    elif n_features == 1:
        axs = axs.reshape(-1, 1)

    for i, gesture in enumerate(gestures_sorted):
        trials = by_gesture[gesture]
        for j, feat in enumerate(features):
            ax = axs[i, j]
            all_aligned = []
            for frames in trials:
                ts = _extract_timeseries(frames, sensor, max_frames)
                vals = ts.get(feat, np.zeros(max_frames))
                aligned = _resample_to_target(vals, max_frames)
                all_aligned.append(aligned)
                ax.plot(aligned, alpha=0.3, linewidth=0.8)
            if all_aligned:
                mean_curve = np.mean(all_aligned, axis=0)
                std_curve = np.std(all_aligned, axis=0)
                t_axis = np.arange(max_frames)
                ax.plot(t_axis, mean_curve, color="black", linewidth=2)
                ax.fill_between(t_axis, mean_curve - std_curve, mean_curve + std_curve, color="black", alpha=0.1)
            if i == 0:
                ax.set_title(feat, fontsize=20, fontweight="bold")
            if j == 0:
                ax.set_ylabel(gesture, fontsize=20, fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=11)

    fig.suptitle(f"{sensor} consistency — all gestures ({', '.join(features)})", fontsize=24)
    _safe_save(fig, "all_gestures", sensor, f"consistency_all", session_ts)
    print(f"  Figure too large for screen — open saved PNG to scroll")
    plt.close(fig)


def overlay_comparison(frames_by_gesture: dict[str, list[dict]], sensor: str, features: list[str], max_frames: int = 100, session_ts: str | None = None) -> None:
    import matplotlib.pyplot as plt

    n_features = len(features)
    cols = 3
    rows = (n_features + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), constrained_layout=True)
    axs = axs.flatten() if isinstance(axs, np.ndarray) else [axs]

    for idx, feat in enumerate(features):
        ax = axs[idx]
        for gesture, flist in frames_by_gesture.items():
            n = min(len(flist), max_frames)
            vals = []
            for f in flist[:n]:
                sd = f.get(sensor, {})
                d = sd.get("data", {})
                if sensor == "mmwave":
                    pts = d.get("points", [])
                    if feat == "num_points":
                        vals.append(len(pts))
                    elif feat == "mean_x" and pts:
                        vals.append(np.mean([p["x"] for p in pts]))
                    elif feat == "mean_y" and pts:
                        vals.append(np.mean([p["y"] for p in pts]))
                    elif feat == "mean_vel" and pts:
                        vals.append(np.mean([abs(p.get("velocity", 0)) for p in pts]))
                    else:
                        vals.append(0)
                elif sensor == "imu":
                    acc = d.get("accel", None)
                    if not isinstance(acc, list):
                        acc = [d.get("accel_x", 0), d.get("accel_y", 0), d.get("accel_z", 0)]
                    gyr = d.get("gyro", None)
                    if not isinstance(gyr, list):
                        gyr = [d.get("gyro_x", 0), d.get("gyro_y", 0), d.get("gyro_z", 0)]
                    if feat == "accel_x":
                        vals.append(acc[0])
                    elif feat == "accel_y":
                        vals.append(acc[1])
                    elif feat == "accel_z":
                        vals.append(acc[2])
                    elif feat == "gyro_x":
                        vals.append(gyr[0])
                    elif feat == "gyro_y":
                        vals.append(gyr[1])
                    elif feat == "gyro_z":
                        vals.append(gyr[2])
                    else:
                        vals.append(0)
                elif sensor == "uwb":
                    ranges = d.get("ranges_cm", [])
                    if feat == "tag1_distance_m" and len(ranges) > 0 and ranges[0] is not None:
                        vals.append(float(ranges[0]) / 100.0)
                    elif feat == "tag2_distance_m" and len(ranges) > 1 and ranges[1] is not None:
                        vals.append(float(ranges[1]) / 100.0)
                    else:
                        vals.append(0)
                else:
                    vals.append(0)
            ax.plot(vals, label=gesture, alpha=0.8)
        ax.set_ylabel(feat)
        ax.set_xlabel("Frame")
        ax.legend()
        ax.grid(True, alpha=0.3)

    for idx in range(n_features, len(axs)):
        axs[idx].set_visible(False)

    feat_label = "_".join(features) if n_features > 1 else features[0]
    fig.suptitle(f"Overlay: {sensor}.{feat_label} by gesture")
    _safe_save(fig, "overlay", sensor, f"overlay_{feat_label}", session_ts)
    if rows > 3:
        print(f"  Figure too large for screen — open saved PNG to scroll")
        plt.close(fig)
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize recorded gesture data")
    parser.add_argument("--input", default="data/raw",
                        help="Session folder (with events.csv) or JSONL file/directory")
    parser.add_argument("--mode", choices=["stats", "sensor", "view", "overlay", "compare", "consistency"],
                        default="stats",
                        help="What to show (stats: dataset stats, sensor: trial table, view: per-trial sensor plots, overlay: overlay lines, compare: feature comparison, consistency: trial consistency)")
    parser.add_argument("--sensor", required=True, choices=["mmwave", "imu", "uwb"],
                        help="Which sensor data to visualize")
    parser.add_argument("--features", default="all",
                        help="Feature(s) to display (default: all features for the selected sensor)")
    parser.add_argument("--gesture", nargs="+", default=None,
                        help="Filter to one or more gestures (default: all)")
    parser.add_argument("--no-details", action="store_true",
                        help="With --mode stats, skip per-trial detail output (gesture summary only)")
    args = parser.parse_args()

    path = Path(args.input)
    frames_by_file: dict[str, list[dict]] = {}
    session_ts: str | None = None

    if path.is_dir() and (path / "events.csv").exists():
        frames_by_file = load_session_csvs(path)
        folder_ts = path.name.removeprefix("session_")
        session_ts = folder_ts if "_" in folder_ts else None
        print(f"Loaded session folder: {path}")
    elif path.is_dir():
        session_dirs = sorted(path.glob("*/events.csv"))
        if session_dirs:
            for ev_csv in session_dirs:
                session_name = ev_csv.parent.name
                session_data = load_session_csvs(ev_csv.parent)
                for key, trial_frames in session_data.items():
                    unique_key = f"{session_name}/{key}"
                    frames_by_file[unique_key] = trial_frames
            print(f"Loaded {len(session_dirs)} session folders from {path}")
        else:
            files = sorted(path.glob("*.jsonl"))
            if not files:
                print(f"No session folders or .jsonl files found in {path}")
                return
            for f in files:
                frames = load_jsonl(f)
                gesture = frames[0].get("gesture", "?") if frames else "?"
                trial = frames[0].get("trial", 0) if frames else 0
                key = f"{gesture}_t{trial}"
                frames_by_file[key] = frames
            print(f"Loaded {len(files)} .jsonl files")
    elif path.suffix == ".jsonl":
        frames = load_jsonl(path)
        gesture = frames[0].get("gesture", "?") if frames else "?"
        trial = frames[0].get("trial", 0) if frames else 0
        key = f"{gesture}_t{trial}"
        frames_by_file[key] = frames
        print(f"Loaded {path.name}")
    elif path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        frames_list = data.get("frames", data if isinstance(data, list) else [])
        if not frames_list:
            print(f"No frames found in {path}")
            return
        for frame in frames_list:
            g = frame.get("gesture", "?")
            t = frame.get("trial", 0)
            key = f"{g}_t{t}"
            frames_by_file.setdefault(key, []).append(frame)
        print(f"Loaded combined dataset: {path.name} ({len(frames_list)} frames, {len(frames_by_file)} trials)")
    elif path.suffix == ".csv":
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame = {
                    "timestamp": row["timestamp"],
                    "gesture": row["gesture"],
                    "trial": int(row["trial"]),
                    "elapsed": float(row["elapsed"]),
                }
                points_raw = row.get("points", "[]")
                try:
                    points = json.loads(points_raw) if points_raw and points_raw != "null" else []
                except json.JSONDecodeError:
                    points = []
                range_raw = row.get("range_profile", "null")
                try:
                    range_profile = json.loads(range_raw) if range_raw and range_raw != "null" else None
                except json.JSONDecodeError:
                    range_profile = None
                mm_data = {
                    "num_points": int(row.get("num_points", len(points))),
                    "points": points,
                    "range_profile": range_profile,
                    "motion_score": float(row.get("motion_score", 0.0)),
                }
                frame["mmwave"] = {
                    "data": mm_data,
                    "confidence": float(row.get("confidence", 0.0)),
                    "sensor_type": "mmwave",
                }
                key = f"{row['gesture']}_t{row['trial']}"
                frames_by_file.setdefault(key, []).append(frame)
        print(f"Loaded combined CSV: {path.name} ({sum(len(v) for v in frames_by_file.values())} frames, {len(frames_by_file)} trials)")
    else:
        print(f"Unrecognized input: {path}")
        return

    filtered_frames = _filter_by_gesture(frames_by_file, args.gesture)
    for name, frames in filtered_frames.items():
        print(f"  {name}: {len(frames)} frames")

    features_list = _resolve_features(args.features, args.sensor)

    if args.mode == "overlay":
        field = args.field if args.field != "all" else "centroid_y"
        by_gesture: dict[str, list[dict]] = {}
        for name, frames in filtered_frames.items():
            g = frames[0].get("gesture", "?")
            by_gesture.setdefault(g, []).extend(frames)
        overlay_comparison(by_gesture, args.sensor, features_list, session_ts=session_ts)
        return

    if args.mode == "compare":
        compare_gestures(filtered_frames, args.sensor, features_list, session_ts=session_ts)
        return

    if args.mode == "consistency":
        if args.gesture and len(args.gesture) == 1:
            consistency_gesture(filtered_frames, args.sensor, args.gesture[0], features_list, session_ts=session_ts)
        else:
            consistency_multi_gestures(filtered_frames, args.sensor, features_list, session_ts=session_ts)
        return

    if args.mode == "sensor":
        meta_keys = {"timestamp", "gesture", "trial", "elapsed"}
        by_gesture: dict[str, dict[str, int]] = {}
        for name, frames in filtered_frames.items():
            g = frames[0].get("gesture", "?")
            trial_sensors = set(k for k in frames[0] if k not in meta_keys)
            for f in frames[1:]:
                trial_sensors |= set(k for k in f if k not in meta_keys)
            d = by_gesture.setdefault(g, {"trial_count": 0})
            d["trial_count"] += 1
            for s in trial_sensors:
                d[s] = d.get(s, 0) + 1
        all_sensors = sorted({s for v in by_gesture.values() for s in v if s != "trial_count"})
        header = f"{'Gesture':<25} {'Trials':>6}"
        for s in all_sensors:
            header += f"  {s:>8}"
        print(header)
        print("-" * len(header))
        for g in sorted(by_gesture):
            info = by_gesture[g]
            line = f"{g:<25} {info['trial_count']:>6}"
            for s in all_sensors:
                line += f"  {info.get(s, 0):>8}"
            print(line)
        return

    if args.mode == "stats":
        by_gesture: dict[str, list[tuple[str, list[dict]]]] = {}
        for name, frames in filtered_frames.items():
            g = frames[0].get("gesture", "?")
            by_gesture.setdefault(g, []).append((name, frames))

        for gesture_name, trials in sorted(by_gesture.items()):
            trial_n = []
            trial_dur = []
            trial_fps = []
            all_dt = []

            for trial_name, frames in trials:
                elapsed = np.array([f.get("elapsed", 0.0) for f in frames])
                n = len(frames)
                if n < 2:
                    continue
                dur = elapsed[-1] - elapsed[0]
                fps = n / dur if dur > 0 else 0.0
                dt = np.diff(elapsed) * 1000  # ms

                trial_n.append(n)
                trial_dur.append(dur)
                trial_fps.append(fps)
                all_dt.extend(dt)

            if not trial_n:
                continue

            print(f"\n{'='*60}")
            print(f"Gesture: {gesture_name}")
            print(f"  Trials: {len(trials)}")
            print(f"  Frames per trial: mean={np.mean(trial_n):.1f}  median={np.median(trial_n):.0f}  min={min(trial_n)}  max={max(trial_n)}")
            print(f"  Duration per trial (s): mean={np.mean(trial_dur):.2f}  median={np.median(trial_dur):.2f}  min={min(trial_dur):.2f}  max={max(trial_dur):.2f}")

            if not args.no_details:
                print(f"\n  Trial details:")
                for trial_name, frames in trials:
                    elapsed = np.array([f.get("elapsed", 0.0) for f in frames])
                    n = len(frames)
                    if n < 2:
                        print(f"    {trial_name}: {n} frames (too few for timing)")
                        continue
                    dur = elapsed[-1] - elapsed[0]
                    fps = n / dur if dur > 0 else 0.0
                    dt = np.diff(elapsed) * 1000

                    print(f"    {trial_name}:")
                    print(f"      Frames: {n}")
                    print(f"      Duration: {dur:.2f}s")
                    print(f"      Avg frame rate: {fps:.1f} fps")
                    print(f"      Δt (ms): min={dt.min():.1f}  max={dt.max():.1f}  mean={dt.mean():.1f}  std={dt.std():.1f}")

        return

    if args.mode == "view":
        for name, frames in filtered_frames.items():
            gesture = frames[0].get("gesture", "?")
            print(f"\n=== {name} ({gesture}) ===")
            if args.sensor == "mmwave":
                if any("mmwave" in f for f in frames):
                    show_mmwave(frames, session_ts=session_ts)
                else:
                    print("No mmWave data in this file")
            elif args.sensor == "imu":
                if any("imu" in f for f in frames):
                    show_imu(frames, session_ts=session_ts)
                else:
                    print("No IMU data in this file")
            elif args.sensor == "uwb":
                if any("uwb" in f for f in frames):
                    show_uwb(frames, session_ts=session_ts)
                else:
                    print("No UWB data in this file")
        return


if __name__ == "__main__":
    main()
