import json

import numpy as np


def _load_json(source):
    """Return a dict from a file path (str) or pass through an already-loaded dict."""
    if isinstance(source, str):
        with open(source) as f:
            return json.load(f)
    return source


def generate_lock_deltas(route_json, extra_characteristics_json, output_path=None):
    if isinstance(route_json, str):
        route_path = route_json
        with open(route_json) as f:
            route_data = json.load(f)
    else:
        route_path = None
        route_data = route_json

    if isinstance(extra_characteristics_json, str):
        with open(extra_characteristics_json) as f:
            extra = json.load(f)
    else:
        extra = extra_characteristics_json

    std_x = extra.get("loc_delta_X_std", 0.0)
    std_y = extra.get("loc_delta_Y_std", 0.0)
    std_z = extra.get("loc_delta_Z_std", 0.0)

    lock_points = []
    for pt in route_data["trajectory"]:
        lock_points.append({
            "id": pt["id"],
            "delta_X": float(np.random.normal(0, std_x)),
            "delta_Y": float(np.random.normal(0, std_y)),
            "delta_Z": float(np.random.normal(0, std_z)),
        })

    result = {"trajectory": lock_points}

    if output_path is None:
        if route_path is None:
            raise ValueError("output_path must be provided when route_json is a dict")
        output_path = route_path.replace("_trajectory.json", "_lock_deltas.json")

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


def _battery_altitude(battery):
    """Extract the battery/camera altitude from a battery or extra-characteristics dict."""
    if battery is None:
        return 0.0
    if "BatteryPosition" in battery:
        return float((battery["BatteryPosition"] or {}).get("Altitude", 0.0))
    if "BatteryAltitude" in battery:
        return float(battery["BatteryAltitude"])
    return float(battery.get("Altitude", 0.0))


def _camera_positions(route_json, battery_json=None, lock_deltas_json=None):
    """Resolve aircraft East/North/Altitude (+ noise) and the camera altitude.

    Returns (east_all, north_all, alt_all, cam_altitude, ids). Trajectory X=East,
    Y=North, Z=absolute altitude; the camera sits at the horizontal origin.
    """
    route_data = _load_json(route_json)
    battery = _load_json(battery_json) if battery_json is not None else None
    cam_altitude = _battery_altitude(battery)

    deltas_by_id = {}
    if lock_deltas_json is not None:
        lock_data = _load_json(lock_deltas_json)
        for d in lock_data["trajectory"]:
            deltas_by_id[d["id"]] = (d.get("delta_X", 0.0), d.get("delta_Y", 0.0), d.get("delta_Z", 0.0))

    east_all, north_all, alt_all, ids = [], [], [], []
    for pt in route_data["trajectory"]:
        dx, dy, dz = deltas_by_id.get(pt["id"], (0.0, 0.0, 0.0))
        east_all.append(pt["X"] + dx)
        north_all.append(pt["Y"] + dy)
        alt_all.append(pt["Z"] + dz)
        ids.append(pt["id"])

    return east_all, north_all, alt_all, cam_altitude, ids


def _gimbal_angles(east, north, alt, cam_alt):
    """Return (azimute_deg, altitude_deg) arrays: the gimbal pose for each point."""
    east, north, alt = np.asarray(east), np.asarray(north), np.asarray(alt)
    up = alt - cam_alt
    azimute = np.degrees(np.arctan2(east, north)) % 360.0
    altitude = np.degrees(np.arctan2(up, np.hypot(east, north)))
    return azimute, altitude


def _los_endpoints(azimute_deg, altitude_deg, cam_alt, length):
    """Reconstruct the camera->target endpoint purely from the gimbal angles.

    The azimuth (clockwise from North) and elevation are turned back into a 3D
    unit direction, then scaled by `length` (the true camera-target range) and
    added to the camera position (East=0, North=0, Altitude=cam_alt). Correct
    angles reproduce the aircraft position; wrong angles would miss it.
    """
    az = np.radians(azimute_deg)
    el = np.radians(altitude_deg)
    length = np.asarray(length)
    end_east = length * np.cos(el) * np.sin(az)
    end_north = length * np.cos(el) * np.cos(az)
    end_alt = cam_alt + length * np.sin(el)
    return end_east, end_north, end_alt


def _plot_camera_poses(east, north, alt, cam_alt, plot_path, n_arrows=20):
    """Save a 3-panel figure (N-E, N-A, E-A) of the route, camera, and LOS vectors.

    Aircraft positions are drawn as dots, the camera as a single dot, and a sample
    of camera line-of-sight directions as vectors. Each vector is rebuilt from the
    gimbal azimute/altitude angles (not by connecting to the aircraft dot), so a
    correct pose lands exactly on the aircraft. The camera is at (East=0, North=0).
    """
    from matplotlib.figure import Figure

    east = np.asarray(east)
    north = np.asarray(north)
    alt = np.asarray(alt)

    azimute, altitude = _gimbal_angles(east, north, alt, cam_alt)
    rng = np.sqrt(east ** 2 + north ** 2 + (alt - cam_alt) ** 2)
    end_e, end_n, end_a = _los_endpoints(azimute, altitude, cam_alt, rng)

    step = max(1, len(east) // n_arrows)
    s = slice(None, None, step)

    fig = Figure(figsize=(16, 5.2))
    (ax_ne, ax_na, ax_ea) = fig.subplots(1, 3)

    panels = [
        # ax,    x_data, y_data, cam_xy,       labels,                  vector U,          V
        (ax_ne, east, north, (0.0, 0.0), ("East [m]", "North [m]"), end_e[s], end_n[s]),
        (ax_na, north, alt, (0.0, cam_alt), ("North [m]", "Altitude [m]"), end_n[s], end_a[s] - cam_alt),
        (ax_ea, east, alt, (0.0, cam_alt), ("East [m]", "Altitude [m]"), end_e[s], end_a[s] - cam_alt),
    ]

    for ax, xd, yd, (cx, cy), (xlabel, ylabel), u, v in panels:
        ax.scatter(xd, yd, s=6, c="#1f77b4", label="Aircraft route")
        ax.quiver(np.full(u.shape, cx), np.full(v.shape, cy), u, v,
                  angles="xy", scale_units="xy", scale=1.0,
                  color="#2ca02c", width=0.003, alpha=0.6, label="Camera LOS (from az/el)")
        ax.plot(cx, cy, marker="X", color="#d62728", markersize=11,
                linestyle="None", label="Camera")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.4)
        ax.set_aspect("equal", adjustable="datalim")

    ax_ne.set_title("Top-down (North-East)")
    ax_na.set_title("Side (North-Altitude)")
    ax_ea.set_title("Side (East-Altitude)")
    ax_ne.legend(loc="best", fontsize=8)

    fig.suptitle("Camera gimbal tracking — LOS vectors reconstructed from azimute/altitude")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)


def generate_camera_pose(route_json, battery_json=None, lock_deltas_json=None,
                         output_json=None, plot_path=None):
    """Generate gimbal poses for a camera at the battery tracking the aircraft route.

    The trajectory world frame has the battery at the horizontal origin with
    X=East, Y=North, Z=Up (absolute altitude). The camera line of sight points
    at each aircraft position (optionally perturbed by lock_deltas noise):

      azimute  = clockwise angle from North, atan2(East, North)   [degrees]
      altitude = elevation above horizontal, atan2(Up, horizontal) [degrees]
      roll     = 0
    """
    route_path = route_json if isinstance(route_json, str) else None
    east_all, north_all, alt_all, cam_altitude, ids = _camera_positions(
        route_json, battery_json, lock_deltas_json)

    azimute, altitude = _gimbal_angles(east_all, north_all, alt_all, cam_altitude)
    gimbal_poses = [
        {"id": pt_id, "azimute": float(az), "altitude": float(el), "roll": 0}
        for pt_id, az, el in zip(ids, azimute, altitude)
    ]

    result = {"gimbal_poses": gimbal_poses}

    if output_json is None:
        if route_path is None:
            raise ValueError("output_json must be provided when route_json is a dict")
        output_json = route_path.replace("_trajectory.json", "_gimbal_pose.json")

    with open(output_json, "w") as f:
        json.dump(result, f, indent=2)

    if plot_path is None and route_path is not None:
        plot_path = route_path.replace("_trajectory.json", "_gimbal_pose.png")
    if plot_path is not None:
        _plot_camera_poses(east_all, north_all, alt_all, cam_altitude, plot_path)

    return result


def _axis_limits(values, cam_value, margin=0.08):
    """Return padded (lo, hi) limits spanning the data and the camera position."""
    lo = min(min(values), cam_value)
    hi = max(max(values), cam_value)
    span = hi - lo or 1.0
    return lo - span * margin, hi + span * margin


def generate_camera_pose_video(route_json, battery_json=None, lock_deltas_json=None,
                               output_path=None, fps=20, n_frames=200):
    """Render an MP4 of the camera tracking the aircraft along the route.

    Three panels (North-East, North-Altitude, East-Altitude) show the full route
    as faint dots, the growing flown trail, the current aircraft position, the
    camera, and the current line-of-sight vector. Defaults to [name]_gimbal_pose.mp4.
    """
    import imageio_ffmpeg
    import matplotlib
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.animation import FFMpegWriter

    route_path = route_json if isinstance(route_json, str) else None
    route_data = _load_json(route_json)
    east, north, alt, cam_alt, _ = _camera_positions(route_data, battery_json, lock_deltas_json)
    east, north, alt = np.asarray(east), np.asarray(north), np.asarray(alt)

    # Raw trajectory X/Y/Z (and time) for the on-screen text readout.
    pts = route_data["trajectory"]
    traj_x = np.array([p["X"] for p in pts])
    traj_y = np.array([p["Y"] for p in pts])
    traj_z = np.array([p["Z"] for p in pts])
    traj_t = np.array([p.get("time_stamp", i) for i, p in enumerate(pts)])

    # LOS arrows are reconstructed from the gimbal azimute/altitude angles (not
    # drawn to the aircraft dot), so they only land on the aircraft if the pose
    # computed for the JSON output is correct.
    azimute, altitude = _gimbal_angles(east, north, alt, cam_alt)
    rng = np.sqrt(east ** 2 + north ** 2 + (alt - cam_alt) ** 2)
    end_e, end_n, end_a = _los_endpoints(azimute, altitude, cam_alt, rng)

    if output_path is None:
        if route_path is None:
            raise ValueError("output_path must be provided when route_json is a dict")
        output_path = route_path.replace("_trajectory.json", "_gimbal_pose.mp4")

    n = len(east)
    frame_idx = np.linspace(0, n - 1, min(n_frames, n)).astype(int)
    bg = max(1, n // 300)  # subsample for the faint full-route backdrop

    fig = Figure(figsize=(16, 5.4))
    FigureCanvasAgg(fig)
    ax_ne, ax_na, ax_ea = fig.subplots(1, 3)

    panels = [
        # ax, x_data, y_data, los_end_x, los_end_y, cam_xy, labels, title
        (ax_ne, east, north, end_e, end_n, (0.0, 0.0),
         ("East [m]", "North [m]"), "Top-down (North-East)"),
        (ax_na, north, alt, end_n, end_a, (0.0, cam_alt),
         ("North [m]", "Altitude [m]"), "Side (North-Altitude)"),
        (ax_ea, east, alt, end_e, end_a, (0.0, cam_alt),
         ("East [m]", "Altitude [m]"), "Side (East-Altitude)"),
    ]
    for ax, xd, yd, ex, ey, (cx, cy), (xlabel, ylabel), title in panels:
        ax.set_xlim(*_axis_limits(xd, cx))
        ax.set_ylim(*_axis_limits(yd, cy))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    writer = FFMpegWriter(fps=fps, metadata={"title": "Camera gimbal tracking"})
    with writer.saving(fig, output_path, dpi=100):
        for i in frame_idx:
            for ax, xd, yd, ex, ey, (cx, cy), (xlabel, ylabel), title in panels:
                xlim, ylim = ax.get_xlim(), ax.get_ylim()
                ax.clear()
                ax.scatter(xd[::bg], yd[::bg], s=5, c="#cfd8e3", zorder=1)
                ax.plot(xd[:i + 1], yd[:i + 1], color="#1f77b4", lw=1.5, zorder=2)
                ax.annotate("", xy=(ex[i], ey[i]), xytext=(cx, cy), zorder=3,
                            arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.6))
                ax.plot(xd[i], yd[i], "o", color="#1f77b4", ms=7, zorder=4)
                ax.plot(cx, cy, marker="X", color="#d62728", ms=11, linestyle="None", zorder=5)
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.set_title(title)
                ax.grid(True, alpha=0.4)
            fig.suptitle(
                "Camera LOS reconstructed from azimute/altitude — aircraft (blue), camera (red X)\n"
                f"t = {traj_t[i]:.2f} s    "
                f"Aircraft  X={traj_x[i]:+.1f}  Y={traj_y[i]:+.1f}  Z={traj_z[i]:+.1f} m    "
                f"Camera  Azimuth={azimute[i]:.2f}°  Elevation={altitude[i]:.2f}°",
                fontsize=11)
            fig.tight_layout()
            writer.grab_frame()

    return output_path


if __name__ == "__main__":
    traj = "/home/borisef/projects/ABT_RoutesGui/ABT_FlightInstructions/TEST3_trajectory.json"
    extra = "/home/borisef/projects/ABT_RoutesGui/ABT_FlightCharacteristics/ABT0_extra.json"
    battery = "/home/borisef/projects/ABT_RoutesGui/ABT_FlightInstructions/TEST3_battery.json"

    deltas = generate_lock_deltas(traj, extra)
    print(f"Generated {len(deltas['trajectory'])} lock delta entries.")
    print("First delta entry:", deltas["trajectory"][0])

    lock_deltas_path = traj.replace("_trajectory.json", "_lock_deltas.json")
    poses = generate_camera_pose(traj, battery_json=battery, lock_deltas_json=lock_deltas_path)
    print(f"Generated {len(poses['gimbal_poses'])} gimbal poses.")
    print("First pose:", poses["gimbal_poses"][0])

    video = generate_camera_pose_video(traj, battery_json=battery, lock_deltas_json=lock_deltas_path)
    print("Saved video:", video)
