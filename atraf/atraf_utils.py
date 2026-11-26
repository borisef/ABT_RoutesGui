import copy
import json
import os
import warnings

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


def _camera_intrinsics(camera_json):
    """Return pinhole intrinsics (fx, fy, cx, cy, W, H) from a camera config.

    Config fields: W, H (pixels) and FOV_H, FOV_V (full field of view, degrees).
    Focal length in pixels is (size/2) / tan(fov/2).
    """
    cam = _load_json(camera_json)
    W = float(cam["W"])
    H = float(cam["H"])
    fov_h = np.radians(float(cam["FOV_H"]))
    fov_v = np.radians(float(cam["FOV_V"]))
    fx = (W / 2.0) / np.tan(fov_h / 2.0)
    fy = (H / 2.0) / np.tan(fov_v / 2.0)
    return {"fx": fx, "fy": fy, "cx": W / 2.0, "cy": H / 2.0, "W": W, "H": H}


def _aircraft_schema(aircraft_json):
    """Return (wing_span, body_length, wings_to_body_ratio, vertical_tail_height)."""
    data = _load_json(aircraft_json)
    schema = data["Aircraft_Schema"][0] if "Aircraft_Schema" in data else data
    return (float(schema["wingSpan"]), float(schema["bodyLength"]),
            float(schema.get("wingsToBodyRatio", 0.5)),
            float(schema.get("verticalTailHeight", 0.0)))


def _aircraft_axes(yaw, pitch, roll):
    """Return (forward, starboard, dorsal) each (n,3) ENU unit vectors from aircraft pose."""
    psi = np.radians(np.asarray(yaw, float))
    th = np.radians(np.asarray(pitch, float))
    ph = np.radians(np.asarray(roll, float))

    forward = np.stack([np.cos(th) * np.sin(psi), np.cos(th) * np.cos(psi), np.sin(th)], axis=1)
    starboard = np.stack([
        np.sin(ph) * np.sin(th) * np.sin(psi) + np.cos(ph) * np.cos(psi),
        np.sin(ph) * np.sin(th) * np.cos(psi) - np.cos(ph) * np.sin(psi),
        -np.sin(ph) * np.cos(th),
    ], axis=1)
    dorsal = np.stack([
        np.sin(ph) * np.cos(psi) - np.cos(ph) * np.sin(th) * np.sin(psi),
        -np.sin(ph) * np.sin(psi) - np.cos(ph) * np.sin(th) * np.cos(psi),
        np.cos(ph) * np.cos(th),
    ], axis=1)
    return forward, starboard, dorsal


def _aircraft_mesh(aircraft_json):
    """Return Aircraft_Mesh config dict if present with required landmarks, else None."""
    data = _load_json(aircraft_json)
    if "Aircraft_Mesh" not in data:
        return None
    mesh_cfg = data["Aircraft_Mesh"][0]
    required = ("left_wing", "right_wing", "nose", "tail", "center_of_body")
    if not all(k in mesh_cfg for k in required):
        return None
    return mesh_cfg


def _build_mesh_transform(mesh_cfg, wing_span, body_length=None, ratio=None,
                          tail_height=None, rescale=False):
    """Compute static transform data for mesh rendering. Returns dict or None.

    Derives an axis mapping (forward/port/dorsal) from nose/tail/wing landmarks
    and a mesh->body transform `M_mesh2body`:
      - rescale=False: a single uniform scale = wing_span / mesh wing span.
      - rescale=True: independent per-axis scaling so the body matches bodyLength,
        the wings match wingSpan and the fin matches verticalTailHeight (aspect
        ratio is intentionally not preserved). Axes whose target size is missing
        keep their native scale (factor 1.0).

    The wing-root (where the wings meet the body) is placed `ratio` of the body
    length from the tail when wingsToBodyRatio is given, else derived by projecting
    the wing midpoint onto the body axis. All coordinate keypoints present in
    mesh_cfg are collected as landmarks. Loads OBJ vertices and builds a subsampled
    edge list for wireframe rendering.
    """
    from atraf.mesh_utils import _load_mesh, _faces_to_edges, _subsample_edges

    cob = np.array(mesh_cfg["center_of_body"], float)
    left = np.array(mesh_cfg["left_wing"], float)
    right = np.array(mesh_cfg["right_wing"], float)
    nose = np.array(mesh_cfg["nose"], float)
    tail = np.array(mesh_cfg["tail"], float)
    vtop = np.array(mesh_cfg.get("vertical_tail_top", mesh_cfg["tail"]), float)

    span_mesh = np.linalg.norm(left - right)
    body_mesh = np.linalg.norm(nose - tail)
    if span_mesh < 1e-9 or body_mesh < 1e-9:
        return None

    scale = wing_span / span_mesh

    forward_mesh = (nose - tail) / body_mesh
    port_mesh = (left - right) / span_mesh
    dorsal_mesh = np.cross(forward_mesh, port_mesh)
    dn = np.linalg.norm(dorsal_mesh)
    if dn < 1e-9:
        return None
    dorsal_mesh /= dn

    # Rows: how each body axis (forward, port, dorsal) is expressed in mesh XYZ frame.
    R_mesh2body = np.array([forward_mesh, port_mesh, dorsal_mesh])  # (3, 3)

    # mesh->body transform: uniform scale, or anisotropic per-axis when rescaling.
    if rescale:
        s_body = (body_length / body_mesh) if body_length else 1.0
        s_span = (wing_span / span_mesh) if wing_span else 1.0
        vert_mesh = float(np.dot(vtop - tail, dorsal_mesh))
        s_vert = (tail_height / vert_mesh) if (tail_height and abs(vert_mesh) > 1e-9) else 1.0
        M_mesh2body = np.diag([s_body, s_span, s_vert]) @ R_mesh2body
    else:
        M_mesh2body = scale * R_mesh2body

    # Wing root: along the body per wingsToBodyRatio (0=tail, 1=nose) when given,
    # else the projection of the wing midpoint onto the body axis.
    if ratio is not None:
        wing_root = tail + ratio * (nose - tail)
    else:
        wing_mid = 0.5 * (left + right)
        wing_root = tail + float(np.dot(wing_mid - tail, forward_mesh)) * forward_mesh

    verts, faces = _load_mesh(mesh_cfg["mesh_path"])
    verts_centered = verts - cob

    # Canonical landmarks plus any extra coordinate keypoints in the mesh config
    # (e.g. horizontal stabilizers), each drawn as a marker in the camera view.
    canonical_src = {
        "nose": nose, "tail": tail, "left_wing": left, "right_wing": right,
        "tail_top": vtop, "center": cob, "wing_root": wing_root,
    }
    canonical_source_keys = {"nose", "tail", "left_wing", "right_wing",
                             "vertical_tail_top", "center_of_body"}
    extra = {k: np.array(v, float) for k, v in mesh_cfg.items()
             if k not in canonical_source_keys
             and isinstance(v, (list, tuple)) and len(v) == 3
             and all(isinstance(x, (int, float)) for x in v)}

    lmk_keys = list(canonical_src.keys()) + list(extra.keys())
    lmk_src = list(canonical_src.values()) + list(extra.values())
    lmk_centered = np.array([v - cob for v in lmk_src], dtype=float)  # (K, 3)

    edges = _subsample_edges(_faces_to_edges(faces), max_edges=6000)

    return {
        "scale": scale,
        "M_mesh2body": M_mesh2body,
        "R_mesh2body": R_mesh2body,
        "cob": cob,
        "verts_centered": verts_centered,
        "lmk_centered": lmk_centered,
        "lmk_keys": lmk_keys,
        "edges": edges,
    }


def _rescaled_mesh_cfg(mesh_cfg, M_mesh2body, cob):
    """Return a copy of mesh_cfg with every coordinate keypoint expressed in the
    rescaled, body-aligned frame (x=forward, y=port, z=dorsal). Non-coordinate
    entries (e.g. mesh_path) are copied unchanged."""
    out = {}
    for k, v in mesh_cfg.items():
        if (isinstance(v, (list, tuple)) and len(v) == 3
                and all(isinstance(x, (int, float)) for x in v)):
            p = M_mesh2body @ (np.array(v, float) - cob)
            out[k] = [round(float(p[0]), 4), round(float(p[1]), 4), round(float(p[2]), 4)]
        else:
            out[k] = v
    return out


def _mesh_project_landmarks(lmk_centered, lmk_keys, M_mesh2ENU, traj_ENU,
                             cam_origin, forward, right, up, intr):
    """Project all mesh landmarks for all frames. Returns (verts_px, depths) dicts."""
    # lmk_world: (n_frames, K, 3); M_mesh2ENU already folds in the (an)isotropic scale.
    lmk_world = traj_ENU[:, None, :] + np.einsum("fij,nj->fni", M_mesh2ENU, lmk_centered)

    verts_px, depths = {}, {}
    for k, name in enumerate(lmk_keys):
        u, v, d = _project_vertices(lmk_world[:, k, :], cam_origin, forward, right, up, intr)
        verts_px[name] = np.stack([u, v], axis=1)
        depths[name] = d
    return verts_px, depths


def _camera_basis(az_deg, el_deg, roll_deg=0.0):
    """Return per-frame camera axes (forward, right, up) as (n, 3) ENU arrays.

    forward = boresight (East, North, Up); right = increasing-azimuth (horizontal);
    up = increasing-elevation. Roll (degrees) rotates right/up about the boresight.
    """
    az = np.radians(np.atleast_1d(az_deg))
    el = np.radians(np.atleast_1d(el_deg))
    roll = np.radians(np.atleast_1d(roll_deg))
    if roll.size == 1:
        roll = np.full(az.shape, roll[0])

    zero = np.zeros_like(az)
    forward = np.stack([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)], axis=1)
    right = np.stack([np.cos(az), -np.sin(az), zero], axis=1)
    up = np.stack([-np.sin(el) * np.sin(az), -np.sin(el) * np.cos(az), np.cos(el)], axis=1)

    cr, sr = np.cos(roll)[:, None], np.sin(roll)[:, None]
    right_r = cr * right + sr * up
    up_r = -sr * right + cr * up
    return forward, right_r, up_r


def _aircraft_vertices(X, Y, Z, yaw, pitch, roll, wing_span, body_length, ratio,
                       vertical_tail_height=0.0):
    """Return the aircraft schematic vertices as a dict of (n, 3) ENU arrays.

    The cross has a body (tail->nose) along the aircraft forward axis and wings
    (left->right) along the starboard axis, meeting at `ratio` of the body length
    from the tail (0 = tail, 1 = nose). A vertical tail rises from the tail along
    the dorsal (up) axis. Forward/starboard/dorsal come from the standard aerospace
    Yaw/Pitch/Roll (heading clockwise from North, pitch up, bank right), so the fin
    leans with roll and pitch.
    """
    center = np.stack([np.asarray(X, float), np.asarray(Y, float), np.asarray(Z, float)], axis=1)
    forward, starboard, dorsal = _aircraft_axes(yaw, pitch, roll)

    tail = center - ratio * body_length * forward
    return {
        "nose": center + (1.0 - ratio) * body_length * forward,
        "tail": tail,
        "right_wing": center + (wing_span / 2.0) * starboard,
        "left_wing": center - (wing_span / 2.0) * starboard,
        "tail_top": tail + vertical_tail_height * dorsal,
        "center": center,
    }


def _project_vertices(points_enu, cam_origin, forward, right, up, intr):
    """Project (n, 3) ENU world points to (u, v) pixels. Returns (u, v, depth)."""
    v = points_enu - cam_origin
    depth = np.sum(v * forward, axis=1)
    xc = np.sum(v * right, axis=1)
    yc = np.sum(v * up, axis=1)
    safe = np.where(depth != 0.0, depth, np.nan)
    u = intr["cx"] + intr["fx"] * (xc / safe)
    px_v = intr["cy"] - intr["fy"] * (yc / safe)
    return u, px_v, depth


def _render_camera_views_video(verts_px, intr, ids, times, az, el, offset_px,
                               output_video, fps=20, n_frames=200, mesh_data=None):
    """Render the schematic camera image sequence to an MP4."""
    import imageio_ffmpeg
    import matplotlib
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.animation import FFMpegWriter

    W, H, cx, cy = intr["W"], intr["H"], intr["cx"], intr["cy"]
    n = len(ids)
    frame_idx = np.linspace(0, n - 1, min(n_frames, n)).astype(int)

    fig = Figure(figsize=(6, 6.4))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)

    # Pre-extract mesh data once outside the frame loop.
    use_mesh_render = mesh_data is not None
    if use_mesh_render:
        from matplotlib.collections import LineCollection
        md_verts = mesh_data["verts_centered"]    # (N, 3)
        md_M = mesh_data["M_mesh2ENU"]            # (n_all, 3, 3), folds in scale
        md_traj = mesh_data["traj_ENU"]           # (n_all, 3)
        md_cam = mesh_data["cam_origin"]          # (3,)
        md_edges = mesh_data["edges"]             # (E_sub, 2)
        i_idx = md_edges[:, 0] if len(md_edges) else np.array([], int)
        j_idx = md_edges[:, 1] if len(md_edges) else np.array([], int)

    title_mode = "mesh" if use_mesh_render else "schematic"
    writer = FFMpegWriter(fps=fps, metadata={"title": f"Camera view ({title_mode})"})
    with writer.saving(fig, output_video, dpi=100):
        for i in frame_idx:
            ax.clear()
            ax.set_facecolor("black")
            ax.set_xlim(0, W)
            ax.set_ylim(H, 0)  # image row 0 at top
            ax.set_aspect("equal")
            # boresight crosshair
            ax.axhline(cy, color="#444444", lw=0.8)
            ax.axvline(cx, color="#444444", lw=0.8)

            if use_mesh_render:
                fwd_f, rgt_f, up_f = _camera_basis(az[i], el[i])
                fwd_f, rgt_f, up_f = fwd_f[0], rgt_f[0], up_f[0]

                # Transform all mesh vertices to world ENU for this frame.
                world_f = md_traj[i] + (md_M[i] @ md_verts.T).T  # (N, 3)

                # Project to camera pixels.
                diff = world_f - md_cam                          # (N, 3)
                depth_f = diff @ fwd_f                           # (N,)
                safe_d = np.where(depth_f != 0, depth_f, np.nan)
                u_f = intr["cx"] + intr["fx"] * (diff @ rgt_f) / safe_d
                v_f = intr["cy"] - intr["fy"] * (diff @ up_f) / safe_d

                # Draw edges where both endpoints are in front of the camera.
                if len(i_idx):
                    vis = (depth_f[i_idx] > 0) & (depth_f[j_idx] > 0)
                    i_v, j_v = i_idx[vis], j_idx[vis]
                    if len(i_v):
                        pts0 = np.stack([u_f[i_v], v_f[i_v]], axis=1)
                        pts1 = np.stack([u_f[j_v], v_f[j_v]], axis=1)
                        segs = np.stack([pts0, pts1], axis=1)
                        ax.add_collection(LineCollection(segs, colors="#00bcd4",
                                                         linewidths=1.0, alpha=1.0,
                                                         zorder=2))

                # Overlay the colored wing/body/fin skeleton so structure is
                # unmistakable even when the wireframe panels are thin. Wings meet
                # the body at the wing root, not the center of mass.
                body = np.array([verts_px["tail"][i], verts_px["nose"][i]])
                left_wing = np.array([verts_px["wing_root"][i], verts_px["left_wing"][i]])
                right_wing = np.array([verts_px["wing_root"][i], verts_px["right_wing"][i]])
                fin = np.array([verts_px["tail"][i], verts_px["tail_top"][i]])
                ax.plot(body[:, 0], body[:, 1], color="#ffffff", lw=2.2, zorder=4)
                ax.plot(left_wing[:, 0], left_wing[:, 1], color="#76ff03", lw=2.6, zorder=4)
                ax.plot(right_wing[:, 0], right_wing[:, 1], color="#ff9100", lw=2.6, zorder=4)
                ax.plot(fin[:, 0], fin[:, 1], color="#ffd600", lw=2.2, zorder=4)

                # All mesh keypoints: 4 prime points in distinct colors, the rest
                # (center, fin tip, stabilizers, wing root, ...) in a common gray.
                prime_colors = {"nose": "#ff1744", "tail": "#2979ff",
                                "left_wing": "#76ff03", "right_wing": "#ff9100"}
                for name, pt in verts_px.items():
                    if name in prime_colors:
                        continue
                    ax.plot(pt[i, 0], pt[i, 1], "o", color="#9e9e9e", ms=3, zorder=5)
                for name, col in prime_colors.items():
                    ax.plot(verts_px[name][i, 0], verts_px[name][i, 1], "o", color=col,
                            ms=6, zorder=6, label=name.replace("_", " "))
                ax.legend(loc="upper right", fontsize=8, facecolor="#222222", labelcolor="white")
            else:
                body = np.array([verts_px["tail"][i], verts_px["nose"][i]])
                left_wing = np.array([verts_px["center"][i], verts_px["left_wing"][i]])
                right_wing = np.array([verts_px["center"][i], verts_px["right_wing"][i]])
                fin = np.array([verts_px["tail"][i], verts_px["tail_top"][i]])
                ax.plot(body[:, 0], body[:, 1], color="#00e5ff", lw=2.0)
                ax.plot(left_wing[:, 0], left_wing[:, 1], color="#76ff03", lw=2.0, label="left wing")
                ax.plot(right_wing[:, 0], right_wing[:, 1], color="#ff9100", lw=2.0, label="right wing")
                ax.plot(fin[:, 0], fin[:, 1], color="#ffd600", lw=2.0)
                ax.plot(verts_px["nose"][i, 0], verts_px["nose"][i, 1], "o", color="#ff1744", ms=5)
                ax.plot(verts_px["center"][i, 0], verts_px["center"][i, 1], "o", color="white", ms=3)
                ax.legend(loc="upper right", fontsize=8, facecolor="#222222", labelcolor="white")

            ax.set_xlabel("u [px]")
            ax.set_ylabel("v [px]")
            ax.set_title(
                f"Camera view ({title_mode})  id={ids[i]}  t={times[i]:.2f} s\n"
                f"Az={az[i]:.2f}°  El={el[i]:.2f}°   "
                f"center offset=({offset_px[i, 0]:+.1f}, {offset_px[i, 1]:+.1f}) px",
                fontsize=10)
            fig.tight_layout()
            writer.grab_frame()
    return output_video


def _join_videos_side_by_side(left_video, right_video, output_video, height=540):
    """Scale both clips to a common height and stack them horizontally into one MP4.

    Both inputs are produced at the same fps and frame count, so the panels stay
    frame-synchronised. Uses the ffmpeg binary bundled with imageio-ffmpeg.
    """
    import subprocess
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    filtergraph = (f"[0:v]scale=-2:{height}[l];[1:v]scale=-2:{height}[r];"
                   f"[l][r]hstack=inputs=2")
    cmd = [ffmpeg, "-y", "-i", left_video, "-i", right_video,
           "-filter_complex", filtergraph, "-c:v", "libx264", "-pix_fmt", "yuv420p",
           output_video]
    subprocess.run(cmd, check=True, capture_output=True)
    return output_video


def generate_camera_views(trajectory, battery=None, gimbal=None, lock_deltas=None,
                          camera=None, aircraft=None, output_json=None,
                          output_video=None, gimbal_video=None, output_combined=None,
                          schematic_view=True, use_mesh=True, rescale=False):
    """Project the aircraft into camera pixels per frame.

    When use_mesh=True (default) and the aircraft JSON contains an Aircraft_Mesh
    section with landmark coordinates, the 3D OBJ mesh is loaded, scaled, and
    projected as a wireframe. Otherwise the schematic cross (body + wings + fin)
    is used. Both paths produce the same output format.

    When rescale=True (mesh path only) the mesh is scaled independently per body
    axis so it matches wingSpan / bodyLength / verticalTailHeight (aspect ratio is
    not preserved), and a full copy of the aircraft config with the rescaled,
    body-aligned landmark coordinates is written to XXX_aircraft_rescaled.json.

    Inputs (paths or dicts): trajectory (required), camera + aircraft (required),
    battery (camera altitude), gimbal (camera pose; computed from trajectory +
    lock_deltas when omitted), lock_deltas (used only to compute the gimbal).

    Writes XXX_camera_view.json (body/wing pixel coordinates per frame) and, when
    requested, XXX_camera_view.mp4. When a gimbal-pose video exists (default
    XXX_gimbal_pose.mp4, or `gimbal_video`), the two clips are joined side by side
    into XXX_combined_view.mp4 (or `output_combined`).
    """
    if camera is None or aircraft is None:
        raise ValueError("camera and aircraft configs are required.")

    route_path = trajectory if isinstance(trajectory, str) else None
    route_data = _load_json(trajectory)
    pts = route_data["trajectory"]
    ids = [p["id"] for p in pts]
    X = np.array([p["X"] for p in pts])
    Y = np.array([p["Y"] for p in pts])
    Z = np.array([p["Z"] for p in pts])
    yaw = np.array([p.get("Yaw", 0.0) for p in pts])
    pitch = np.array([p.get("Pitch", 0.0) for p in pts])
    roll = np.array([p.get("Roll", 0.0) for p in pts])
    times = np.array([p.get("time_stamp", i) for i, p in enumerate(pts)])

    cam_alt = _battery_altitude(_load_json(battery) if battery is not None else None)
    cam_origin = np.array([0.0, 0.0, cam_alt])

    # Camera pointing: gimbal pose if given, else computed from trajectory + lock_deltas.
    if gimbal is not None:
        gdata = _load_json(gimbal)
        by_id = {g["id"]: g for g in gdata["gimbal_poses"]}
        az = np.array([by_id[i]["azimute"] for i in ids])
        el = np.array([by_id[i]["altitude"] for i in ids])
        cam_roll = np.array([by_id[i].get("roll", 0.0) for i in ids], float)
    else:
        e_lock, n_lock, a_lock, _, _ = _camera_positions(route_data, battery, lock_deltas)
        az, el = _gimbal_angles(e_lock, n_lock, a_lock, cam_alt)
        cam_roll = np.zeros(len(ids))

    intr = _camera_intrinsics(camera)
    wing_span, body_length, ratio, tail_height = _aircraft_schema(aircraft)

    # Raw schema to tell which dimensions are explicitly provided (affects rescale
    # per-axis factors and whether the wing root uses wingsToBodyRatio).
    aircraft_data = _load_json(aircraft)
    schema0 = aircraft_data["Aircraft_Schema"][0] if "Aircraft_Schema" in aircraft_data else aircraft_data
    ratio_val = float(schema0["wingsToBodyRatio"]) if "wingsToBodyRatio" in schema0 else None
    target_body = float(schema0["bodyLength"]) if "bodyLength" in schema0 else None
    target_vert = float(schema0["verticalTailHeight"]) if "verticalTailHeight" in schema0 else None

    forward, right, up = _camera_basis(az, el, cam_roll)

    # --- Mesh rendering path (use_mesh=True and Aircraft_Mesh with landmarks present) ---
    if use_mesh or not schematic_view:
        mesh_cfg = _aircraft_mesh(aircraft)
        if mesh_cfg is not None:
            mt = None
            try:
                mt = _build_mesh_transform(mesh_cfg, wing_span, body_length=target_body,
                                           ratio=ratio_val, tail_height=target_vert,
                                           rescale=rescale)
            except (IOError, OSError) as e:
                warnings.warn(f"Cannot load mesh file ({e}); falling back to schematic.")
            if mt is not None:
                fwd_a, stb_a, dsl_a = _aircraft_axes(yaw, pitch, roll)
                # Columns of R_body2ENU are ENU expressions of aircraft body axes.
                R_body2ENU = np.stack([fwd_a, -stb_a, dsl_a], axis=2)   # (n, 3, 3)
                M_mesh2ENU = R_body2ENU @ mt["M_mesh2body"]              # (n, 3, 3)
                traj_ENU = np.stack([X, Y, Z], axis=1)                   # (n, 3)

                verts_px, depths = _mesh_project_landmarks(
                    mt["lmk_centered"], mt["lmk_keys"], M_mesh2ENU, traj_ENU,
                    cam_origin, forward, right, up, intr)

                offset_px = verts_px["center"] - np.array([intr["cx"], intr["cy"]])
                in_w = (verts_px["center"][:, 0] >= 0) & (verts_px["center"][:, 0] < intr["W"])
                in_h = (verts_px["center"][:, 1] >= 0) & (verts_px["center"][:, 1] < intr["H"])
                visible = in_w & in_h & (depths["center"] > 0)

                frames = []
                for k, pid in enumerate(ids):
                    fr = {"id": pid}
                    for name in mt["lmk_keys"]:
                        fr[name] = [round(float(verts_px[name][k, 0]), 2),
                                    round(float(verts_px[name][k, 1]), 2)]
                    fr["visible"] = bool(visible[k])
                    frames.append(fr)
                result = {"camera_view": frames}

                if output_json is None:
                    if route_path is None:
                        raise ValueError("output_json must be provided when trajectory is a dict")
                    output_json = route_path.replace("_trajectory.json", "_camera_view.json")
                with open(output_json, "w") as f:
                    json.dump(result, f, indent=2)

                # Write the rescaled aircraft config (full copy, body-aligned landmarks).
                if rescale:
                    new_data = copy.deepcopy(aircraft_data)
                    new_data["Aircraft_Mesh"][0] = _rescaled_mesh_cfg(
                        mesh_cfg, mt["M_mesh2body"], mt["cob"])
                    if isinstance(aircraft, str):
                        if "_aircraft.json" in aircraft:
                            rescaled_path = aircraft.replace("_aircraft.json", "_aircraft_rescaled.json")
                        else:
                            rescaled_path = aircraft.rsplit(".json", 1)[0] + "_rescaled.json"
                        with open(rescaled_path, "w") as f:
                            json.dump(new_data, f, indent=2)
                    else:
                        warnings.warn("rescale=True but aircraft given as dict; "
                                      "cannot derive a rescaled-JSON output path.")

                if output_video is None and route_path is not None:
                    output_video = route_path.replace("_trajectory.json", "_camera_view.mp4")
                if output_video is not None:
                    mesh_data_for_video = {
                        "verts_centered": mt["verts_centered"],
                        "edges": mt["edges"],
                        "M_mesh2ENU": M_mesh2ENU,
                        "traj_ENU": traj_ENU,
                        "cam_origin": cam_origin,
                    }
                    _render_camera_views_video(verts_px, intr, ids, times, az, el, offset_px,
                                               output_video, mesh_data=mesh_data_for_video)
                    if gimbal_video is None and route_path is not None:
                        gimbal_video = route_path.replace("_trajectory.json", "_gimbal_pose.mp4")
                    if output_combined is None and route_path is not None:
                        output_combined = route_path.replace("_trajectory.json", "_combined_view.mp4")
                    if gimbal_video and os.path.isfile(gimbal_video) and output_combined:
                        _join_videos_side_by_side(gimbal_video, output_video, output_combined)

                return result

    # --- Schematic rendering path (fallback) ---
    verts_enu = _aircraft_vertices(X, Y, Z, yaw, pitch, roll, wing_span, body_length,
                                   ratio, tail_height)

    verts_px, depths = {}, {}
    for name, p_enu in verts_enu.items():
        u, v, d = _project_vertices(p_enu, cam_origin, forward, right, up, intr)
        verts_px[name] = np.stack([u, v], axis=1)
        depths[name] = d

    offset_px = verts_px["center"] - np.array([intr["cx"], intr["cy"]])

    in_w = (verts_px["center"][:, 0] >= 0) & (verts_px["center"][:, 0] < intr["W"])
    in_h = (verts_px["center"][:, 1] >= 0) & (verts_px["center"][:, 1] < intr["H"])
    visible = in_w & in_h & (depths["center"] > 0)

    frames = []
    for k, pid in enumerate(ids):
        frames.append({
            "id": pid,
            "nose": [round(float(verts_px["nose"][k, 0]), 2), round(float(verts_px["nose"][k, 1]), 2)],
            "tail": [round(float(verts_px["tail"][k, 0]), 2), round(float(verts_px["tail"][k, 1]), 2)],
            "left_wing": [round(float(verts_px["left_wing"][k, 0]), 2), round(float(verts_px["left_wing"][k, 1]), 2)],
            "right_wing": [round(float(verts_px["right_wing"][k, 0]), 2), round(float(verts_px["right_wing"][k, 1]), 2)],
            "tail_top": [round(float(verts_px["tail_top"][k, 0]), 2), round(float(verts_px["tail_top"][k, 1]), 2)],
            "center": [round(float(verts_px["center"][k, 0]), 2), round(float(verts_px["center"][k, 1]), 2)],
            "visible": bool(visible[k]),
        })
    result = {"camera_view": frames}

    if output_json is None:
        if route_path is None:
            raise ValueError("output_json must be provided when trajectory is a dict")
        output_json = route_path.replace("_trajectory.json", "_camera_view.json")
    with open(output_json, "w") as f:
        json.dump(result, f, indent=2)

    if output_video is None and route_path is not None:
        output_video = route_path.replace("_trajectory.json", "_camera_view.mp4")
    if output_video is not None:
        _render_camera_views_video(verts_px, intr, ids, times, az, el, offset_px,
                                   output_video)

        # Join the gimbal-pose video and this camera view side by side when available.
        if gimbal_video is None and route_path is not None:
            gimbal_video = route_path.replace("_trajectory.json", "_gimbal_pose.mp4")
        if output_combined is None and route_path is not None:
            output_combined = route_path.replace("_trajectory.json", "_combined_view.mp4")
        if gimbal_video and os.path.isfile(gimbal_video) and output_combined:
            _join_videos_side_by_side(gimbal_video, output_video, output_combined)

    return result


if __name__ == "__main__":
    traj = "/home/borisef/projects/ABT_RoutesGui/ABT_FlightInstructions/TEST4_trajectory.json"
    extra = "/home/borisef/projects/ABT_RoutesGui/ABT_FlightCharacteristics/ABT4_extra.json"
    battery = "/home/borisef/projects/ABT_RoutesGui/ABT_FlightInstructions/TEST4_battery.json"

    deltas = generate_lock_deltas(traj, extra)
    print(f"Generated {len(deltas['trajectory'])} lock delta entries.")
    print("First delta entry:", deltas["trajectory"][0])

    lock_deltas_path = traj.replace("_trajectory.json", "_lock_deltas.json")
    poses = generate_camera_pose(traj, battery_json=battery, lock_deltas_json=lock_deltas_path)
    print(f"Generated {len(poses['gimbal_poses'])} gimbal poses.")
    print("First pose:", poses["gimbal_poses"][0])

    video = generate_camera_pose_video(traj, battery_json=battery, lock_deltas_json=lock_deltas_path)
    print("Saved video:", video)

    gimbal_path = traj.replace("_trajectory.json", "_gimbal_pose.json")
    camera_cfg = "/home/borisef/projects/ABT_RoutesGui/ABT_FlightCharacteristics/ABT4_camera.json"
    aircraft_cfg = "/home/borisef/projects/ABT_RoutesGui/ABT_FlightCharacteristics/ABT4_aircraft.json"
    views = generate_camera_views(
        traj, battery=battery, gimbal=gimbal_path, lock_deltas=lock_deltas_path,
        camera=camera_cfg, aircraft=aircraft_cfg, output_json=None, output_video=None, gimbal_video=None,
        output_combined=None, schematic_view=True, use_mesh=True, rescale=True)
    print(f"Generated {len(views['camera_view'])} camera-view frames.")
    print("First frame:", views["camera_view"][0])
