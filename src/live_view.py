from __future__ import annotations

import argparse
import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np

from src.sensors.mmwave_reader import MmWaveReader


def main() -> None:
    parser = argparse.ArgumentParser(description="Live mmWave radar viewer")
    parser.add_argument("--mode", default="mock", choices=["mock", "serial"])
    parser.add_argument("--port", default="/dev/cu.usbserial-BH00LUQT")
    parser.add_argument("--history", type=int, default=100)
    args = parser.parse_args()

    reader = MmWaveReader(mode=args.mode, serial_port=args.port)
    reader.start()

    plt.ion()
    fig = plt.figure(figsize=(14, 6))

    # Radar apartment-style layout: 2x2 gridspec
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1])

    # Top-left: 3D scatter
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax3d.set_xlabel("X (m)"); ax3d.set_ylabel("Y (m)"); ax3d.set_zlabel("Z (m)")
    ax3d.set_title("Point Cloud")
    ax3d.set_xlim(-2, 2); ax3d.set_ylim(0, 5); ax3d.set_zlim(-0.5, 2, auto=False)
    sc = ax3d.scatter([], [], [], c=[], cmap="plasma", s=8, alpha=0.8)

    # Top-middle: 2D top-down
    ax2d = fig.add_subplot(gs[0, 1])
    ax2d.set_xlim(-2, 2); ax2d.set_ylim(0, 5)
    ax2d.set_xlabel("X (m)"); ax2d.set_ylabel("Y (m)")
    ax2d.set_title("Top-Down")
    ax2d.grid(True, alpha=0.2)
    sc2d = ax2d.scatter([], [], c=[], cmap="plasma", s=10, alpha=0.8)
    target_marker, = ax2d.plot([], [], "ro", markersize=8, alpha=0.6)
    cx_line, = ax2d.plot([], [], "r-", alpha=0.3, lw=1)

    # Top-middle text info
    info_text = ax2d.text(0.02, 0.95, "", transform=ax2d.transAxes,
                          fontsize=7, color="#ccc", verticalalignment="top",
                          fontfamily="monospace")

    # Top-right: centroid time series
    ax_ts = fig.add_subplot(gs[0, 2])
    ax_ts.set_title("Centroid"); ax_ts.set_xlabel("Frame"); ax_ts.set_ylabel("Meters")
    ax_ts.grid(True, alpha=0.2)
    ts_len = args.history
    x_ts = np.arange(ts_len)
    cx_hist = np.full(ts_len, np.nan)
    cy_hist = np.full(ts_len, np.nan)
    cz_hist = np.full(ts_len, np.nan)
    lx, = ax_ts.plot(x_ts, cx_hist, label="x", color="#d06060", lw=1)
    ly, = ax_ts.plot(x_ts, cy_hist, label="y", color="#40c880", lw=1)
    lz, = ax_ts.plot(x_ts, cz_hist, label="z", color="#60b0e0", lw=1)
    ax_ts.legend(fontsize=6)

    # Bottom-right: velocity
    ax_vel = fig.add_subplot(gs[1, 2])
    ax_vel.set_title("Velocity"); ax_vel.set_xlabel("Frame"); ax_vel.set_ylabel("m/s")
    ax_vel.grid(True, alpha=0.2)
    vel_hist = np.full(ts_len, np.nan)
    lv, = ax_vel.plot(x_ts, vel_hist, color="orange", lw=1)

    # Bottom-middle: point count
    ax_n = fig.add_subplot(gs[1, 1])
    ax_n.set_title("Point Count"); ax_n.set_xlabel("Frame")
    ax_n.grid(True, alpha=0.2)
    n_hist = np.full(ts_len, np.nan)
    ln, = ax_n.plot(x_ts, n_hist, color="green", lw=1)

    plt.tight_layout()

    cx_trail = deque(maxlen=30)

    print("Streaming — close window to stop")
    try:
        frame = 0
        while plt.fignum_exists(fig.number):
            reading = reader.read()
            pts = reading.data.get("points", [])
            num = reading.data.get("num_points", len(pts))
            frame += 1

            if pts:
                xs = np.array([p["x"] for p in pts])
                ys = np.array([p["y"] for p in pts])
                zs = np.array([p["z"] for p in pts])
                vs = np.array([p.get("velocity", 0) for p in pts])

                cx, cy, cz = np.mean(xs), np.mean(ys), np.mean(zs)
                mv = np.mean(np.abs(vs))

                np.roll(cx_hist, -1); cx_hist[-1] = cx
                np.roll(cy_hist, -1); cy_hist[-1] = cy
                np.roll(cz_hist, -1); cz_hist[-1] = cz
                np.roll(vel_hist, -1); vel_hist[-1] = mv
                np.roll(n_hist, -1); n_hist[-1] = num

                cx_trail.append((cx, cy))

                c = vs
                sc._offsets3d = (xs, ys, zs)
                sc.set_array(c)

                sc2d.set_offsets(np.c_[xs, ys])
                sc2d.set_array(c)

                if cx_trail:
                    trail = np.array(cx_trail)
                    target_marker.set_data([cx], [cy])
                    cx_line.set_data(trail[:, 0], trail[:, 1])

                info_text.set_text(
                    f"Points: {num}  Frame: {frame}\n"
                    f"Centroid: ({cx:.2f}, {cy:.2f}, {cz:.2f})\n"
                    f"Velocity: {mv:.3f} m/s"
                )

                lx.set_ydata(cx_hist)
                ly.set_ydata(cy_hist)
                lz.set_ydata(cz_hist)
                lv.set_ydata(vel_hist)
                ln.set_ydata(n_hist)

                for ax in [ax_ts, ax_vel, ax_n]:
                    ax.relim()
                    ax.autoscale_view(scalex=False)

                fig.canvas.draw_idle()
                fig.canvas.start_event_loop(0.001)
            else:
                info_text.set_text(f"No points — frame {frame}")

    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()
        plt.close(fig)


if __name__ == "__main__":
    main()
