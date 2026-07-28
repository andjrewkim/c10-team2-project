from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


FIG_DIR = Path("results") / "mmwave_figures"


def _safe_save(fig, gesture: str, sensor: str, mode: str, session_ts: str | None) -> None:
    ts = session_ts or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    name = f"{gesture}_{sensor}_{mode}_{ts}.png"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(FIG_DIR / name), dpi=150, bbox_inches="tight")
    print(f"  Saved: {FIG_DIR / name}")


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
        acc = d.get("accel", [0, 0, 0])
        gyr = d.get("gyro", [0, 0, 0])
        axs.append(acc[0])
        ays.append(acc[1])
        azs.append(acc[2])
        gxs.append(gyr[0])
        gys.append(gyr[1])
        gzs.append(gyr[2])

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


def _resample_to_target(values: np.ndarray, target: int) -> np.ndarray:
    if len(values) == 0:
        return np.zeros(target)
    if len(values) == 1:
        return np.full(target, values[0])
    return np.interp(np.linspace(0, len(values) - 1, target), np.arange(len(values)), values)


def _filter_by_gesture(frames_by_file: dict[str, list[dict]], gesture: str | None):
    if not gesture:
        return frames_by_file
    filtered = {}
    for key, frames in frames_by_file.items():
        if frames and frames[0].get("gesture") == gesture:
            filtered[key] = frames
    return filtered


def compare_gestures(frames_by_trial: dict[str, list[dict]], feature: str, max_frames: int = 100, session_ts: str | None = None) -> None:
    import matplotlib.pyplot as plt

    by_gesture: dict[str, list[np.ndarray]] = {}
    for key, frames in frames_by_trial.items():
        g = frames[0].get("gesture", "?")
        ts = extract_mmwave_timeseries(frames, max_frames)
        vals = ts.get(feature, ts.get("centroid_y", np.zeros(max_frames)))
        by_gesture.setdefault(g, []).append(vals)

    fig, ax = plt.subplots(figsize=(14, 6))
    for gesture, trials in sorted(by_gesture.items()):
        max_len = min(max(len(t) for t in trials), max_frames)
        aligned = np.array([_resample_to_target(t, max_len) for t in trials])
        mean_curve = np.mean(aligned, axis=0)
        std_curve = np.std(aligned, axis=0)
        t_axis = np.arange(max_len)
        ax.plot(t_axis, mean_curve, label=gesture, linewidth=2)
        ax.fill_between(t_axis, mean_curve - std_curve, mean_curve + std_curve, alpha=0.12)

    ax.set_ylabel(feature)
    ax.set_xlabel("Frame (resampled)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.suptitle(f"mmWave feature comparison — {feature} (mean ± std across trials)")
    _safe_save(fig, "all_gestures", "mmwave", f"compare_{feature}", session_ts)
    plt.tight_layout()
    plt.show()


def consistency_gesture(frames_by_trial: dict[str, list[dict]], gesture: str, feature: str, max_frames: int = 200, session_ts: str | None = None) -> None:
    import matplotlib.pyplot as plt

    trials = [f for key, f in frames_by_trial.items() if f[0].get("gesture", "?") == gesture]
    if not trials:
        print(f"No trials found for gesture '{gesture}'")
        return

    all_features = ["centroid_x", "centroid_y", "num_points", "body_width", "body_depth"]
    features_to_plot = all_features if feature == "all" else [feature]

    n_features = len(features_to_plot)
    cols = 3
    rows = (n_features + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axs = axs.flatten() if isinstance(axs, np.ndarray) else [axs]

    for idx, feat in enumerate(features_to_plot):
        ax = axs[idx]
        all_aligned = []
        for t_idx, frames in enumerate(trials):
            ts = extract_mmwave_timeseries(frames, max_frames)
            if np.all(ts["num_points"] == 0):
                continue
            vals = ts.get(feat, np.zeros(max_frames))
            aligned = _resample_to_target(vals, max_frames)
            all_aligned.append(aligned)
            ax.plot(aligned, alpha=0.35, linewidth=1, label=f"trial {t_idx + 1}" if n_features == 1 and len(trials) <= 10 else None)

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

    fig.suptitle(f"mmWave consistency — {gesture} ({len(trials)} trials)")
    _safe_save(fig, gesture, "mmwave", f"consistency_{feature}", session_ts)
    plt.tight_layout()
    plt.show()


def consistency_multi_gestures(frames_by_trial: dict[str, list[dict]], field: str, max_frames: int = 200, session_ts: str | None = None) -> None:
    import matplotlib.pyplot as plt

    by_gesture: dict[str, list[list[dict]]] = {}
    for key, frames in frames_by_trial.items():
        g = frames[0].get("gesture", "?")
        ts = extract_mmwave_timeseries(frames, max_frames)
        if np.all(ts["num_points"] == 0):
            continue
        by_gesture.setdefault(g, []).append(frames)

    if not by_gesture:
        print("No gestures with mmWave data found")
        return

    gestures_sorted = sorted(by_gesture.keys())
    all_features = ["centroid_x", "centroid_y", "num_points", "body_width", "body_depth"]
    features_to_plot = all_features if field == "all" else [field]

    n_gestures = len(gestures_sorted)
    n_features = len(features_to_plot)
    fig, axs = plt.subplots(n_gestures, n_features, figsize=(8 * n_features + 2, 3 * n_gestures), constrained_layout=True)
    if n_gestures == 1 and n_features == 1:
        axs = np.array([[axs]])
    elif n_gestures == 1:
        axs = axs.reshape(1, -1)
    elif n_features == 1:
        axs = axs.reshape(-1, 1)

    for i, gesture in enumerate(gestures_sorted):
        trials = by_gesture[gesture]
        for j, feat in enumerate(features_to_plot):
            ax = axs[i, j]
            all_aligned = []
            for frames in trials:
                ts = extract_mmwave_timeseries(frames, max_frames)
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

    fig.suptitle(f"mmWave consistency — all gestures ({field})", fontsize=24)
    _safe_save(fig, "all_gestures", "mmwave", f"consistency_{field}", session_ts)
    print(f"  Figure too large for screen — open saved PNG to scroll")
    plt.close(fig)


def overlay_comparison(frames_by_gesture: dict[str, list[dict]], sensor: str, field: str, max_frames: int = 100, session_ts: str | None = None) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    for gesture, flist in frames_by_gesture.items():
        n = min(len(flist), max_frames)
        vals = []
        for f in flist[:n]:
            s = f.get(sensor, {})
            d = s.get("data", {})
            pts = d.get("points", [])
            if field == "num_points":
                vals.append(len(pts))
            elif field == "mean_x" and pts:
                vals.append(np.mean([p["x"] for p in pts]))
            elif field == "mean_y" and pts:
                vals.append(np.mean([p["y"] for p in pts]))
            elif field == "mean_vel" and pts:
                vals.append(np.mean([abs(p.get("velocity", 0)) for p in pts]))
            else:
                vals.append(0)
        ax.plot(vals, label=gesture, alpha=0.8)
    ax.set_ylabel(field)
    ax.set_xlabel("Frame")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.suptitle(f"Overlay: {sensor}.{field} by gesture")
    _safe_save(fig, "overlay", sensor, f"overlay_{field}", session_ts)
    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize recorded gesture data")
    parser.add_argument("--input", default="data/raw",
                        help="Session folder (with events.csv) or JSONL file/directory")
    parser.add_argument("--mode", choices=["stats", "mmwave", "imu", "overlay", "compare", "consistency"],
                        default="stats",
                        help="What to show")
    """mode argument options"""
        # stats: prints frame count per gesture and which sensors were used
        # mmwave: 
            # centroid x/y over time
            # body width (x span) and body depth (y span)
            # point count
            # active frame fraction (single bar)
            # 2D occupancy heatmap (all accumulated points)
        # imu: IMU accelerometer and gyro data
        # overlay: Plots all gestures concatenated (all trials merged) as separate lines on one axis
        # compare: Plots one feature with one line per gesture (mean ± std across trials, resampled to same length)
        # consistency: Plots all trials of one gesture overlaid (each trial = faint line, black = mean ± std)
    parser.add_argument("--field", default="all",
                        help="Feature field (centroid_x, centroid_y, num_points, body_width, body_depth, or all)")
    """field argument"""
        # specify a field if using --mode overlay, compare, or consistency
        # "all" shows all features (consistency mode only)
    parser.add_argument("--gesture", default=None,
                        help="Filter to one gesture (default: all)")
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
    if args.mode == "overlay":
        field = args.field if args.field != "all" else "centroid_y"
        by_gesture: dict[str, list[dict]] = {}
        for name, frames in filtered_frames.items():
            g = frames[0].get("gesture", "?")
            by_gesture.setdefault(g, []).extend(frames)
        sensor = "mmwave"
        for f in list(by_gesture.values())[0]:
            if "imu" in f:
                sensor = "imu"
                break
        overlay_comparison(by_gesture, sensor, field, session_ts=session_ts)
        return
    if args.mode == "compare":
        field = args.field if args.field != "all" else "centroid_y"
        compare_gestures(filtered_frames, field, session_ts=session_ts)
        return
    if args.mode == "consistency":
        if args.gesture:
            consistency_gesture(filtered_frames, args.gesture, args.field, session_ts=session_ts)
        else:
            consistency_multi_gestures(filtered_frames, args.field, session_ts=session_ts)
        return

    if args.mode == "stats":
        by_gesture: dict[str, list[tuple[str, list[dict]]]] = {}
        for name, frames in filtered_frames.items():
            g = frames[0].get("gesture", "?")
            by_gesture.setdefault(g, []).append((name, frames))

        for gesture_name, trials in sorted(by_gesture.items()):
            trial_n = []
            trial_dur = []
            all_dt = []

            for trial_name, frames in trials:
                elapsed = np.array([f.get("elapsed", 0.0) for f in frames])
                n = len(frames)
                if n < 2:
                    continue
                dur = elapsed[-1] - elapsed[0]
                dt = np.diff(elapsed) * 1000

                trial_n.append(n)
                trial_dur.append(dur)
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

    for name, frames in filtered_frames.items():
        gesture = frames[0].get("gesture", "?")
        print(f"\n=== {name} ({gesture}) ===")

        if args.mode == "mmwave":
            has_mm = any("mmwave" in f for f in frames)
            if has_mm:
                show_mmwave(frames, session_ts=session_ts)
            else:
                print("No mmWave data in this file")

        elif args.mode == "imu":
            has_imu = any("imu" in f for f in frames)
            if has_imu:
                show_imu(frames, session_ts=session_ts)
            else:
                print("No IMU data in this file")


if __name__ == "__main__":
    main()
