r"""GUI for building ABT flight-instruction JSON files and plotting the top-down route.

Usage:
    python FlightRouteGui.py [path\to\instructions.json]

Direct mode edits FlightSegments; waypoint mode generates and verifies those same
instructions from requested arrival poses. JSON keeps route/coordinate metadata
in type and waypoint inputs in a separate coordinates list.

The editor uses north/east variable names in both coordinate modes. Their displayed
values mean meters in BAT and latitude/longitude degrees in GEO. read_state and
planning_waypoints are the conversion boundary: the simulator always receives
local NED meters, with the GEO ABT origin as its geographic reference. Plotting
converts the simulated trace back to latitude/longitude when needed.

See README.md for schemas, model limitations, dependency setup, and test commands.
"""

import json
import math
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from PlotFlightRoute import (
    configure_route_axes, draw_altitude_profile, draw_route, draw_waypoints,
    expand_target_segment, generate_waypoint_segments,
    geographic_to_local, infer_target_groups, local_to_geographic, simulate, validate_geographic_position,
    DT,
)

def find_data_dir():
    """Find the project's DataFiles directory or fall back to the current directory."""
    starts = [Path.cwd()]
    if getattr(sys, "frozen", False):
        starts.append(Path(sys.executable).resolve().parent)
    starts.append(Path(__file__).resolve().parent)
    for start in starts:
        for parent in (start, *start.parents):
            if (parent / "DataFiles").is_dir():
                return parent / "DataFiles"
    return Path.cwd()


TRAJECTORY_DELTA = 0.1  # seconds between trajectory samples

DATA_DIR = find_data_dir()
INSTRUCTIONS_DIR = DATA_DIR / "ABT_FlightInstructions"
CHARACTERISTICS_DIR = DATA_DIR / "ABT_FlightCharacteristics"
DEFAULT_CHARACTERISTICS_FILE = CHARACTERISTICS_DIR / "ABT0.json"

# (JSON key, column title, default value)
SEGMENT_FIELDS = [
    ("startTimeInSeconds", "Start [s]", 0.0),
    ("targetBankInDegrees", "Bank [°]", 0.0),
    ("rollRateInDegreesPerSecond", "Roll rate [°/s]", 8.0),
    ("targetPitchInDegrees", "Pitch [°]", 0.0),
    ("pitchRateInDegreesPerSecond", "Pitch rate [°/s]", 4.0),
    ("targetSpeedInMetersPerSecond", "Speed [m/s]", 40.0),
    ("accelerationInMetersPerSecondSquared", "Accel [m/s²]", 3.0),
]

# (state key, label, default value)
STATE_FIELDS = [
    ("north", "North [m]", 0.0),
    ("east", "East [m]", 0.0),
    ("down", "Down [m] (neg = up)", 0.0),
    ("roll", "Roll [°]", 0.0),
    ("pitch", "Pitch [°]", 0.0),
    ("heading", "Heading [°] (0 = N)", 0.0),
    ("speed", "Speed [m/s]", 40.0),
    ("end_time", "End time [s] (blank = auto)", ""),
]

ORIGIN_FIELDS = [
    ("North", "north", 0.0),
    ("East", "east", 0.0),
    ("Down", "down", 0.0),
    ("Pitch", "pitch", 0.0),
    ("Roll", "roll", 0.0),
    ("Speed", "speed", 40.0),
    ("Heading", "heading", 0.0),
]

WAYPOINT_FIELDS = [
    ("north", "North [m]", 0.0),
    ("east", "East [m]", 0.0),
    ("down", "Down [m] (NED)", 0.0),
    ("azimuth", "Azimuth [°]", 0.0),
    ("speed", "Speed [m/s]", 40.0),
]

LEG_LIMIT_FIELDS = [
    ("maxBankAngle", "Max bank [°]", 30.0),
    ("maxRollRate", "Roll rate [°/s]", 8.0),
    ("maxPitchUpAngle", "Max pitch up [°]", 20.0),
    ("maxPitchDownAngle", "Max pitch down [°]", 20.0),
    ("maxPitchRate", "Pitch rate [°/s]", 4.0),
    ("maxAccel", "Max accel [m/s²]", 3.0),
]


LIMIT_KEYS = ["maxRollAngle", "maxRollRate", "maxPitchUpAngle", "maxPitchUpRate", "maxPitchDownAngle",
              "maxPitchDownRate", "maxSpeed", "maxAccel", "minSpeed", "minAccel"]


class RouteNavigationToolbar(NavigationToolbar2Tk):
    def __init__(self, canvas, window, reset_callback, **kwargs):
        """Build the Matplotlib toolbar with an application-specific original-view reset."""
        self.reset_callback = reset_callback
        super().__init__(canvas, window, **kwargs)

    def home(self, *_args):
        """Restore the application's original limits for both route plots."""
        self.reset_callback()


def load_limits(path):
    """Load supported flight limits from the first characteristics entry."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        limits = data["Flight_Characteristics"][0]
    except OSError as e:
        raise ValueError(f"Cannot read file: {e}")
    except (KeyError, IndexError, TypeError):
        raise ValueError("Expected a 'Flight_Characteristics' list with at least one entry.")
    found = {k: float(limits[k]) for k in LIMIT_KEYS if isinstance(limits.get(k), (int, float))}
    if not found:
        raise ValueError(f"No known limit fields found ({', '.join(LIMIT_KEYS)}).")
    return found


def parse_route_type(data):
    """Return (path_mode, coordinate_type, editable_waypoints) from route metadata.

    Missing type metadata defaults to direct instructions/BAT for old files.
    Waypoint inputs normally live in the top-level coordinates list; the older
    type.coordinates location and positive-up altitude key remain readable.
    GEO JSON latitude/longitude keys map to the editor's north/east variables;
    these are degrees until planning_waypoints projects them to local meters.
    Raises ValueError for unsupported metadata, units/ranges, or waypoint limits.
    """
    route_type = data.get("type", {})
    if not isinstance(route_type, dict):
        raise ValueError("'type' must be an object.")
    path_mode = route_type.get("pathType", "direct instructions")
    coordinate_type = route_type.get("coordinateType", "BAT")
    if path_mode not in ("direct instructions", "waypoints based"):
        raise ValueError("'type.pathType' must be 'direct instructions' or 'waypoints based'.")
    if coordinate_type not in ("BAT", "GEO"):
        raise ValueError("'type.coordinateType' must be 'BAT' or 'GEO'.")

    raw_waypoints = data.get("coordinates", route_type.get("coordinates", []))
    if path_mode == "direct instructions":
        return path_mode, coordinate_type, []
    if not isinstance(raw_waypoints, list) or not raw_waypoints:
        raise ValueError("Waypoint routes require a non-empty 'coordinates' list.")

    defaults = {key: default for key, _, default in LEG_LIMIT_FIELDS}
    waypoints = []
    for index, raw in enumerate(raw_waypoints, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"Waypoint {index} must be an object.")
        waypoint = {}
        for key, label, _ in WAYPOINT_FIELDS:
            try:
                json_key = {"north": "latitude", "east": "longitude"}.get(key, key) \
                    if coordinate_type == "GEO" else key
                raw_value = raw[json_key] if key != "down" or "down" in raw else -float(raw["altitude"])
                value = float(raw_value)
            except (KeyError, TypeError, ValueError):
                raise ValueError(f"Waypoint {index}: '{label}' must be a number.")
            if not math.isfinite(value):
                raise ValueError(f"Waypoint {index}: '{label}' must be finite.")
            waypoint[key] = value
        if coordinate_type == "GEO":
            validate_geographic_position(waypoint["north"], waypoint["east"])
        for key, label, _ in LEG_LIMIT_FIELDS:
            try:
                value = float(raw.get(key, defaults[key]))
            except (TypeError, ValueError):
                raise ValueError(f"Waypoint {index}: '{label}' must be a number.")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Waypoint {index}: '{label}' must be finite and >= 0.")
            waypoint[key] = value
        if waypoint["speed"] <= 0 or waypoint["maxRollRate"] <= 0 \
                or waypoint["maxPitchRate"] <= 0 or waypoint["maxAccel"] <= 0:
            raise ValueError(f"Waypoint {index}: speed, rates, and acceleration must be > 0.")
        if any(waypoint[key] >= 89 for key in ("maxBankAngle", "maxPitchUpAngle", "maxPitchDownAngle")):
            raise ValueError(f"Waypoint {index}: maximum bank and pitch angles must be below 89°.")
        waypoint["azimuth"] %= 360.0
        waypoints.append(waypoint)
    return path_mode, coordinate_type, waypoints


def build_route_type(path_mode, coordinate_type):
    """Build the common route-mode and coordinate-type header."""
    return {"pathType": path_mode, "coordinateType": coordinate_type}


def limit_warnings(segments, limits, initial_state=None, downs=None):
    """Return flight-limit and ground-contact warnings for the simulated route."""
    lim = (limits or {}).get
    checks = [
        ("targetBankInDegrees", "maxRollAngle", lambda v, m: abs(v) > m, "|bank| > {:g}°"),
        ("rollRateInDegreesPerSecond", "maxRollRate", lambda v, m: v > m, "roll rate > {:g}°/s"),
        ("targetPitchInDegrees", "maxPitchUpAngle", lambda v, m: v > m, "pitch > {:g}°"),
        ("targetPitchInDegrees", "maxPitchDownAngle", lambda v, m: v < -m, "pitch < -{:g}°"),
        ("targetSpeedInMetersPerSecond", "maxSpeed", lambda v, m: v > m, "speed > {:g} m/s"),
        ("targetSpeedInMetersPerSecond", "minSpeed", lambda v, m: v < m, "speed < {:g} m/s"),
    ]
    warnings = []
    if initial_state is not None:
        min_speed = lim("minSpeed")
        if min_speed is not None and initial_state["speed"] < min_speed:
            warnings.append(f"Origin position: speed < {min_speed:g} m/s")
        if initial_state["down"] >= 0:
            warnings.append("Origin position: altitude is at or below 0 m (ground level)")
        elif downs is not None and any(down >= 0 for down in downs[1:]):
            warnings.append("Flight course: altitude reaches or falls below 0 m (ground level)")

    prev = None
    for i, seg in enumerate(segments, start=1):
        for key, limit_key, failed, text in checks:
            if lim(limit_key) is not None and failed(seg[key], lim(limit_key)):
                warnings.append(f"Segment {i}: {text.format(lim(limit_key))}")

        # Pitch rate / acceleration limits depend on the direction of change from the previous segment.
        prev_pitch = prev["targetPitchInDegrees"] if prev else None
        pitch_key = "maxPitchDownRate" if prev_pitch is not None and seg["targetPitchInDegrees"] < prev_pitch \
            else "maxPitchUpRate"
        if lim(pitch_key) is not None and seg["pitchRateInDegreesPerSecond"] > lim(pitch_key):
            warnings.append(f"Segment {i}: pitch rate > {lim(pitch_key):g}°/s")

        decelerating = prev is not None and seg["targetSpeedInMetersPerSecond"] < prev["targetSpeedInMetersPerSecond"]
        if decelerating and lim("minAccel") is not None:
            if seg["accelerationInMetersPerSecondSquared"] > abs(lim("minAccel")):
                warnings.append(f"Segment {i}: deceleration > {abs(lim('minAccel')):g} m/s²")
        elif not decelerating and lim("maxAccel") is not None:
            if seg["accelerationInMetersPerSecondSquared"] > lim("maxAccel"):
                warnings.append(f"Segment {i}: accel > {lim('maxAccel'):g} m/s²")
        prev = seg
    return warnings


def waypoint_limit_warnings(waypoints, limits):
    """Return warnings when waypoint targets or maneuver caps exceed aircraft limits."""
    if not limits:
        return []
    warnings = []
    checks = [
        ("speed", "maxSpeed", "speed > {:g} m/s"),
        ("speed", "minSpeed", "speed < {:g} m/s"),
        ("maxBankAngle", "maxRollAngle", "max bank angle > {:g}°"),
        ("maxRollRate", "maxRollRate", "max roll rate > {:g}°/s"),
        ("maxPitchUpAngle", "maxPitchUpAngle", "max pitch up > {:g}°"),
        ("maxPitchDownAngle", "maxPitchDownAngle", "max pitch down > {:g}°"),
        ("maxPitchRate", "maxPitchUpRate", "max pitch rate > {:g}°/s"),
        ("maxPitchRate", "maxPitchDownRate", "max pitch rate > {:g}°/s"),
        ("maxAccel", "maxAccel", "max acceleration > {:g} m/s²"),
        ("maxAccel", "minAccel", "max deceleration > {:g} m/s²"),
    ]
    for index, waypoint in enumerate(waypoints, start=1):
        for waypoint_key, limit_key, text in checks:
            limit = limits.get(limit_key)
            if limit is None:
                continue
            allowed = abs(limit) if limit_key == "minAccel" else limit
            exceeds = waypoint[waypoint_key] < limit if limit_key == "minSpeed" \
                else waypoint[waypoint_key] > allowed
            if exceeds:
                warnings.append(f"Waypoint {index}: {text.format(allowed)}")
    return warnings


class FlightRouteApp(tk.Tk):
    """Own editable route data, file lifecycle, Tk controls, and simulated plots.

    segments is always the instruction schedule saved as FlightSegments.
    waypoints retains user input in the selected coordinate system, including
    incoming-leg limits; it is never overwritten with projected coordinates.
    _loading_route and _loading_state suppress destructive mode resets and
    dirty-state notifications while loading values into traced Tk variables.
    """

    def __init__(self, initial_file=None):
        """Initialize the editor window, load default limits, and open the requested route."""
        super().__init__()
        # Fit the window to the screen so the plot toolbar is never pushed off the bottom.
        width = min(1500, self.winfo_screenwidth() - 40)
        height = min(900, self.winfo_screenheight() - 100)
        self.geometry(f"{width}x{height}+10+10")
        self.minsize(900, 600)
        if self.winfo_screenheight() < 1000:
            self.state("zoomed")
        self.limits = None
        self.limits_path = None
        self.file_path = None
        self.dirty = False
        self._loading_state = False
        self._original_view = None
        self._pan_drag = None
        self.target_groups = {}
        self.segments = []
        self.waypoints = []
        self.path_mode = "direct instructions"
        self._loading_route = False

        self._build_menu()
        self._build_layout()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        if DEFAULT_CHARACTERISTICS_FILE.is_file():
            self.set_limits_file(DEFAULT_CHARACTERISTICS_FILE, show_errors=False)
        else:
            self.set_limits_file(None)

        if initial_file:
            self.open_file(Path(initial_file))
        else:
            self.new_file()

    # ---------- layout ----------
    def _build_menu(self):
        """Create the application menu and register its keyboard shortcuts."""
        menu = tk.Menu(self)
        file_menu = tk.Menu(menu, tearoff=False)
        file_menu.add_command(label="New", accelerator="Ctrl+N", command=self.new_file)
        file_menu.add_command(label="Open...", accelerator="Ctrl+O", command=self.open_dialog)
        file_menu.add_command(label="Save", accelerator="Ctrl+S", command=self.save)
        file_menu.add_command(label="Save As...", command=self.save_as)
        file_menu.add_command(label="Customized Save...", command=self.customized_save)
        file_menu.add_separator()
        file_menu.add_command(label="Export plot image...", command=self.export_plot)
        file_menu.add_command(label="Select limits file...", command=self.choose_limits_file)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_close)
        menu.add_cascade(label="File", menu=file_menu)
        self.config(menu=menu)
        self.bind_all("<Control-n>", lambda e: self.new_file())
        self.bind_all("<Control-o>", lambda e: self.open_dialog())
        self.bind_all("<Control-s>", lambda e: self.save())

    def _build_layout(self):
        """Construct the segment editor, initial-state controls, status area, and plot canvas."""
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        left = ttk.Frame(paned, padding=8)
        right = ttk.Frame(paned, padding=4)
        paned.add(left, weight=0)
        paned.add(right, weight=1)

        toolbar = ttk.Frame(left)
        toolbar.pack(fill=tk.X)
        for text, cmd in [("New", self.new_file), ("Open...", self.open_dialog),
                          ("Save", self.save), ("Save As...", self.save_as),
                          ("Customized Save...", self.customized_save)]:
            ttk.Button(toolbar, text=text, command=cmd).pack(side=tk.LEFT, padx=2)
        self.file_label = ttk.Label(left, foreground="#444")
        self.file_label.pack(fill=tk.X, pady=(6, 4))

        header = ttk.Frame(left)
        header.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(header, text="Coordinate type").pack(side=tk.LEFT, padx=(2, 6))
        self.coordinate_type_var = tk.StringVar(value="BAT")
        coordinate_type = ttk.Combobox(
            header, textvariable=self.coordinate_type_var, values=("BAT", "GEO"),
            state="readonly", width=8)
        coordinate_type.pack(side=tk.LEFT)
        self.coordinate_type_var.trace_add("write", self.on_coordinate_type_changed)

        self.route_tabs = ttk.Notebook(left)
        self.route_tabs.pack(fill=tk.BOTH, expand=True)
        self.route_tabs.bind("<<NotebookTabChanged>>", self.on_route_tab_changed)

        # Direct-instruction editor
        seg_box = ttk.Frame(self.route_tabs, padding=6)
        self.route_tabs.add(seg_box, text="Direct instructions")
        columns = [key for key, _, _ in SEGMENT_FIELDS]
        self.tree = ttk.Treeview(seg_box, columns=columns, show="headings", height=6, selectmode="browse")
        for key, title, _ in SEGMENT_FIELDS:
            self.tree.heading(key, text=title)
            self.tree.column(key, width=88, anchor=tk.CENTER)
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.tree.bind("<<TreeviewSelect>>", self.on_select)

        # Segment editor
        editor = ttk.Frame(seg_box)
        editor.pack(fill=tk.X, pady=(6, 0))
        self.seg_vars = {}
        for col, (key, title, default) in enumerate(SEGMENT_FIELDS):
            ttk.Label(editor, text=title).grid(row=0, column=col, padx=2, sticky=tk.W)
            var = tk.StringVar(value=f"{default:g}")
            ttk.Entry(editor, textvariable=var, width=11).grid(row=1, column=col, padx=2)
            self.seg_vars[key] = var

        targets = ttk.LabelFrame(seg_box, text="Optional target (blank = direct segment)", padding=4)
        targets.pack(fill=tk.X, pady=(6, 0))
        self.target_vars = {}
        for col, (key, label) in enumerate([
            ("azimuth", "Turn to azimuth [°]"),
            ("altitude", "Level at altitude [m] (neg = up)"),
        ]):
            ttk.Label(targets, text=label).grid(row=0, column=col, padx=2, sticky=tk.W)
            var = tk.StringVar(value="")
            ttk.Entry(targets, textvariable=var, width=16).grid(row=1, column=col, padx=2, sticky=tk.W)
            self.target_vars[key] = var

        shift_box = ttk.Frame(seg_box)
        shift_box.pack(fill=tk.X, pady=(4, 0))
        self.shift_following_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(shift_box, text="Shift following segments by",
                        variable=self.shift_following_var).pack(side=tk.LEFT, padx=2)
        self.shift_seconds_var = tk.StringVar(value="50")
        ttk.Entry(shift_box, textvariable=self.shift_seconds_var, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(shift_box, text="s (negative shortens)").pack(side=tk.LEFT, padx=2)

        buttons = ttk.Frame(seg_box)
        buttons.pack(fill=tk.X, pady=(6, 0))
        for text, cmd in [("Add", self.add_segment), ("Update selected", self.update_segment),
                          ("Delete selected", self.delete_segment), ("Clear all", self.clear_segments)]:
            ttk.Button(buttons, text=text, command=cmd).pack(side=tk.LEFT, padx=2)

        self._build_waypoint_tab()

        # Flight limits file
        limits_box = ttk.LabelFrame(left, text="Flight limits (characteristics file)", padding=6)
        limits_box.pack(fill=tk.X, pady=(8, 0))
        self.limits_label = ttk.Label(limits_box, text="(none)", width=60, anchor=tk.W)
        self.limits_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(limits_box, text="None", command=lambda: self.set_limits_file(None)).pack(side=tk.RIGHT, padx=2)
        ttk.Button(limits_box, text="Browse...", command=self.choose_limits_file).pack(side=tk.RIGHT, padx=2)

        # Initial state
        state_box = ttk.LabelFrame(left, text="Initial state", padding=6)
        state_box.pack(fill=tk.X, pady=(8, 0))
        self.state_vars = {}
        self.state_labels = {}
        for i, (key, label, default) in enumerate(STATE_FIELDS):
            r, c = divmod(i, 2)
            widget = ttk.Label(state_box, text=label)
            widget.grid(row=r, column=c * 2, sticky=tk.W, padx=(0, 4), pady=2)
            self.state_labels[key] = widget
            var = tk.StringVar(value=f"{default:g}" if default != "" else "")
            entry = ttk.Entry(state_box, textvariable=var, width=12)
            entry.grid(row=r, column=c * 2 + 1, sticky=tk.W, padx=(0, 16), pady=2)
            entry.bind("<Return>", lambda e: self.plot_route())
            self.state_vars[key] = var
            if key in {state_key for _, state_key, _ in ORIGIN_FIELDS}:
                var.trace_add("write", self.on_origin_changed)

        system_box = ttk.LabelFrame(left, text="Battery position", padding=6)
        system_box.pack(fill=tk.X, pady=(8, 0))
        self.system_vars = {}
        self.system_labels = {}
        for col, (key, label) in enumerate((("north", "North [m]"), ("east", "East [m]"))):
            widget = ttk.Label(system_box, text=label)
            widget.grid(row=0, column=col * 2, sticky=tk.W, padx=(0, 4))
            self.system_labels[key] = widget
            var = tk.StringVar(value="")
            entry = ttk.Entry(system_box, textvariable=var, width=12)
            entry.grid(row=0, column=col * 2 + 1, sticky=tk.W, padx=(0, 16))
            entry.bind("<Return>", lambda e: self.replot())
            self.system_vars[key] = var
            var.trace_add("write", self.on_origin_changed)

        ttk.Button(left, text="Plot route", command=self.plot_route).pack(fill=tk.X, pady=(8, 0))

        self.status = tk.Text(left, height=6, wrap=tk.WORD, relief=tk.FLAT, background=self.cget("background"))
        self.status.pack(fill=tk.X, pady=(8, 0))
        self.status.configure(state=tk.DISABLED)

        # Plot area
        self.figure = Figure(figsize=(10, 10), dpi=100)
        self.ax, self.altitude_ax = self.figure.subplots(
            2, 1, gridspec_kw={"height_ratios": [3, 1]})
        self.canvas = FigureCanvasTkAgg(self.figure, master=right)
        self.canvas.mpl_connect("scroll_event", self.zoom_plot)
        self.canvas.mpl_connect("button_press_event", self.start_plot_pan)
        self.canvas.mpl_connect("motion_notify_event", self.drag_plot)
        self.canvas.mpl_connect("button_release_event", self.stop_plot_pan)
        toolbar_frame = ttk.Frame(right)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        self.toolbar = RouteNavigationToolbar(
            self.canvas, toolbar_frame, self.reset_view, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side=tk.LEFT, fill=tk.X)
        ttk.Button(toolbar_frame, text="Reset view", command=self.reset_view).pack(side=tk.RIGHT, padx=4)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_waypoint_tab(self):
        """Create the waypoint list and its target and incoming-leg limit editor."""
        panel = ttk.Frame(self.route_tabs, padding=6)
        self.route_tabs.add(panel, text="Waypoints")

        columns = tuple(key for key, _, _ in WAYPOINT_FIELDS)
        self.waypoint_tree = ttk.Treeview(panel, columns=columns, show="headings", height=5,
                                          selectmode="browse")
        for key, title, _ in WAYPOINT_FIELDS:
            self.waypoint_tree.heading(key, text=title)
            self.waypoint_tree.column(key, width=82, anchor=tk.CENTER)
        self.waypoint_tree.pack(fill=tk.BOTH, expand=True)
        self.waypoint_tree.bind("<<TreeviewSelect>>", self.on_waypoint_select)

        fields = ttk.LabelFrame(panel, text="Waypoint target", padding=4)
        fields.pack(fill=tk.X, pady=(6, 0))
        self.waypoint_vars = {}
        self.waypoint_labels = {}
        for column, (key, label, default) in enumerate(WAYPOINT_FIELDS):
            widget = ttk.Label(fields, text=label)
            widget.grid(row=0, column=column, padx=2, sticky=tk.W)
            self.waypoint_labels[key] = widget
            variable = tk.StringVar(value=f"{default:g}")
            ttk.Entry(fields, textvariable=variable, width=12).grid(row=1, column=column, padx=2)
            self.waypoint_vars[key] = variable

        limits = ttk.LabelFrame(panel, text="Incoming-leg limits", padding=4)
        limits.pack(fill=tk.X, pady=(6, 0))
        self.leg_limit_vars = {}
        for column, (key, label, default) in enumerate(LEG_LIMIT_FIELDS):
            row, col = divmod(column, 3)
            ttk.Label(limits, text=label).grid(row=row * 2, column=col, padx=2, sticky=tk.W)
            variable = tk.StringVar(value=f"{default:g}")
            ttk.Entry(limits, textvariable=variable, width=12).grid(row=row * 2 + 1, column=col, padx=2)
            self.leg_limit_vars[key] = variable

        buttons = ttk.Frame(panel)
        buttons.pack(fill=tk.X, pady=(6, 0))
        for text, command in [
            ("Add", self.add_waypoint), ("Update selected", self.update_waypoint),
            ("Delete selected", self.delete_waypoint), ("Move up", lambda: self.move_waypoint(-1)),
            ("Move down", lambda: self.move_waypoint(1)),
        ]:
            ttk.Button(buttons, text=text, command=command).pack(side=tk.LEFT, padx=2)

    def on_route_tab_changed(self, _event=None):
        """Track whether the active route is direct-instruction or waypoint based."""
        if not hasattr(self, "route_tabs"):
            return
        selected = self.route_tabs.tab(self.route_tabs.select(), "text")
        new_mode = "waypoints based" if selected == "Waypoints" else "direct instructions"
        if new_mode != self.path_mode:
            self.path_mode = new_mode
            if not self._loading_route:
                self.mark_dirty()

    def on_coordinate_type_changed(self, *_args):
        """Update labels and reset positions rather than reinterpret their units.

        User-initiated changes clear origin/battery coordinates and waypoint
        inputs, preserving direct instructions. A file load updates labels only;
        its saved coordinates are populated separately under _loading_route.
        """
        if not hasattr(self, "state_labels"):
            return
        geographic = self.coordinate_type_var.get() == "GEO"
        labels = ("Latitude [deg]", "Longitude [deg]") if geographic else ("North [m]", "East [m]")
        for key, label in zip(("north", "east"), labels):
            self.state_labels[key].configure(text=label)
            self.system_labels[key].configure(text=label)
            self.waypoint_labels[key].configure(text=label)
            self.waypoint_tree.heading(key, text=label)
        if not self._loading_route:
            self._loading_state = True
            try:
                for key in ("north", "east"):
                    self.state_vars[key].set("0")
                    self.system_vars[key].set("")
                    self.waypoint_vars[key].set("0")
                self.state_vars["down"].set("0")
                self.waypoint_vars["down"].set("0")
            finally:
                self._loading_state = False
            self.waypoints = []
            self.refresh_waypoints()
            if self.path_mode == "waypoints based":
                self.segments = []
                self.target_groups.clear()
                self.refresh_tree()
            self.mark_dirty()
            self.replot()

    def planning_waypoints(self, state):
        """Return local-meter waypoint records without changing the editable inputs.

        GEO uses state['geo_reference'], the ABT origin's WGS84 latitude/longitude
        pair. BAT records are already local. Rates, Down, and target speed keep
        the same units in both modes.
        """
        if self.coordinate_type_var.get() != "GEO":
            return self.waypoints
        waypoints = []
        for waypoint in self.waypoints:
            north, east = geographic_to_local(waypoint["north"], waypoint["east"], state["geo_reference"])
            waypoints.append(dict(waypoint, north=north, east=east))
        return waypoints

    def read_waypoint_editor(self):
        """Parse the waypoint target and its incoming-leg maneuver limits."""
        waypoint = {}
        for key, label, _ in WAYPOINT_FIELDS:
            try:
                value = float(self.waypoint_vars[key].get())
            except ValueError:
                raise ValueError(f"'{label}' must be a number.")
            if not math.isfinite(value):
                raise ValueError(f"'{label}' must be finite.")
            waypoint[key] = value
        if self.coordinate_type_var.get() == "GEO":
            validate_geographic_position(waypoint["north"], waypoint["east"])
        if waypoint["speed"] <= 0:
            raise ValueError("Waypoint speed must be > 0.")
        waypoint["azimuth"] %= 360.0
        for key, label, _ in LEG_LIMIT_FIELDS:
            try:
                value = float(self.leg_limit_vars[key].get())
            except ValueError:
                raise ValueError(f"'{label}' must be a number.")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"'{label}' must be finite and >= 0.")
            waypoint[key] = value
        if waypoint["maxRollRate"] <= 0 or waypoint["maxPitchRate"] <= 0 or waypoint["maxAccel"] <= 0:
            raise ValueError("Roll rate, pitch rate, and acceleration must be > 0.")
        if any(waypoint[key] >= 89 for key in ("maxBankAngle", "maxPitchUpAngle", "maxPitchDownAngle")):
            raise ValueError("Maximum bank and pitch angles must be < 89°.")
        return waypoint

    def refresh_waypoints(self, select_index=None):
        """Redraw waypoint rows and optionally select one."""
        self.waypoint_tree.delete(*self.waypoint_tree.get_children())
        for index, waypoint in enumerate(self.waypoints):
            self.waypoint_tree.insert(
                "", tk.END, iid=str(index),
                values=[f"{waypoint[key]:g}" for key, _, _ in WAYPOINT_FIELDS])
        if select_index is not None and 0 <= select_index < len(self.waypoints):
            self.waypoint_tree.selection_set(str(select_index))
            self.waypoint_tree.see(str(select_index))

    def on_waypoint_select(self, _event=None):
        """Load a selected waypoint and its incoming-leg limits into the editor."""
        selection = self.waypoint_tree.selection()
        if not selection:
            return
        waypoint = self.waypoints[int(selection[0])]
        for key, _, _ in WAYPOINT_FIELDS:
            self.waypoint_vars[key].set(f"{waypoint[key]:g}")
        for key, _, _ in LEG_LIMIT_FIELDS:
            self.leg_limit_vars[key].set(f"{waypoint[key]:g}")

    def selected_waypoint_index(self):
        """Return the selected waypoint index, or None when no row is selected."""
        selection = self.waypoint_tree.selection()
        return int(selection[0]) if selection else None

    def add_waypoint(self):
        """Append a validated waypoint and regenerate its flight instructions."""
        try:
            waypoint = self.read_waypoint_editor()
        except ValueError as error:
            messagebox.showerror("Invalid waypoint", str(error), parent=self)
            return
        self.waypoints.append(waypoint)
        self.refresh_waypoints(len(self.waypoints) - 1)
        self.waypoints_changed()

    def update_waypoint(self):
        """Replace the selected waypoint and regenerate the route."""
        index = self.selected_waypoint_index()
        if index is None:
            messagebox.showinfo("Update waypoint", "Select a waypoint first.", parent=self)
            return
        try:
            self.waypoints[index] = self.read_waypoint_editor()
        except ValueError as error:
            messagebox.showerror("Invalid waypoint", str(error), parent=self)
            return
        self.refresh_waypoints(index)
        self.waypoints_changed()

    def delete_waypoint(self):
        """Delete the selected waypoint and regenerate the remaining route."""
        index = self.selected_waypoint_index()
        if index is None:
            return
        self.waypoints.pop(index)
        self.refresh_waypoints(min(index, len(self.waypoints) - 1))
        self.waypoints_changed()

    def move_waypoint(self, offset):
        """Move the selected waypoint earlier or later in the route order."""
        index = self.selected_waypoint_index()
        target_index = index + offset if index is not None else -1
        if index is None or not 0 <= target_index < len(self.waypoints):
            return
        self.waypoints[index], self.waypoints[target_index] = \
            self.waypoints[target_index], self.waypoints[index]
        self.refresh_waypoints(target_index)
        self.waypoints_changed()

    def waypoints_changed(self):
        """Regenerate the instruction schedule after a waypoint edit."""
        self.mark_dirty()
        if not self.waypoints:
            self.segments = []
            self.target_groups.clear()
            self.refresh_tree()
            self.replot()
            return
        try:
            self.generate_waypoint_route(show_errors=False)
        except ValueError as error:
            self.segments = []
            self.target_groups.clear()
            self.refresh_tree()
            self.replot()
            warnings = waypoint_limit_warnings(self.waypoints, self.limits)
            lines = [f"Waypoint route needs attention: {error}"]
            if self.limits:
                lines.extend(f"  • {warning}" for warning in warnings)
            else:
                lines.append("WARNING: Aircraft characteristics not loaded; waypoint limits cannot be checked.")
            self.set_status(lines, error=True)

    def generate_waypoint_route(self, show_errors=True):
        """Generate ordinary timed flight segments from the ordered waypoint list."""
        if not self.waypoints:
            if show_errors:
                messagebox.showerror("Generate route", "Add at least one waypoint.", parent=self)
                return False
            raise ValueError("Add at least one waypoint.")
        try:
            state = self.read_state()
            waypoints = self.planning_waypoints(state)
            segments = generate_waypoint_segments(waypoints, state, waypoints)
        except (ValueError, KeyError) as error:
            if show_errors:
                messagebox.showerror("Generate route", str(error), parent=self)
                return False
            raise ValueError(str(error)) from error
        self.segments = segments
        self.target_groups.clear()
        self.refresh_tree()
        self.mark_dirty()
        self.replot()
        return True

    # ---------- helpers ----------
    def set_status(self, lines, error=False):
        """Replace the status panel contents and style them as normal or error feedback."""
        self.status.configure(state=tk.NORMAL)
        self.status.delete("1.0", tk.END)
        self.status.insert(tk.END, "\n".join(lines))
        self.status.configure(state=tk.DISABLED, foreground="#b00020" if error else "#333")

    def update_title(self):
        """Refresh the window title and displayed route path, including the dirty marker."""
        name = self.file_path.name if self.file_path else "Untitled"
        self.title(f"{'*' if self.dirty else ''}{name} - ABT Flight Route Builder")
        self.file_label.config(text=str(self.file_path) if self.file_path else "(unsaved file)")

    def mark_dirty(self):
        """Mark the current route as modified and refresh its title."""
        self.dirty = True
        self.update_title()

    def on_origin_changed(self, *_args):
        """Mark the route modified when an origin-state field changes outside file loading."""
        if not self._loading_state:
            self.mark_dirty()

    def set_state_values(self, values):
        """Set initial-state controls from values, using defaults for omitted fields."""
        self._loading_state = True
        try:
            for key, _, default in STATE_FIELDS:
                value = values.get(key, default)
                self.state_vars[key].set(f"{value:g}" if value != "" else "")
        finally:
            self._loading_state = False

    def refresh_tree(self, select_index=None):
        """Sort segments by start time and rebuild the table, optionally selecting a row."""
        self.segments.sort(key=lambda s: s["startTimeInSeconds"])
        self.tree.delete(*self.tree.get_children())
        for i, seg in enumerate(self.segments):
            self.tree.insert("", tk.END, iid=str(i), values=[f"{seg[k]:g}" for k, _, _ in SEGMENT_FIELDS])
        if select_index is not None and 0 <= select_index < len(self.segments):
            self.tree.selection_set(str(select_index))
            self.tree.see(str(select_index))

    def read_segment_editor(self):
        """Parse and validate the segment editor controls into a flight-segment dictionary."""
        seg = {}
        for key, title, _ in SEGMENT_FIELDS:
            try:
                seg[key] = float(self.seg_vars[key].get())
            except ValueError:
                raise ValueError(f"'{title}' must be a number.")
        if seg["startTimeInSeconds"] < 0:
            raise ValueError("Start time must be >= 0.")
        labels = {k: t for k, t, _ in SEGMENT_FIELDS}
        for key in ("rollRateInDegreesPerSecond", "pitchRateInDegreesPerSecond",
                    "accelerationInMetersPerSecondSquared"):
            if seg[key] < 0:
                raise ValueError(f"'{labels[key]}' must be >= 0 (use 0 for no change).")
        if seg["targetSpeedInMetersPerSecond"] <= 0:
            raise ValueError(f"'{labels['targetSpeedInMetersPerSecond']}' must be > 0.")
        return seg

    def read_target_options(self):
        """Parse azimuth and Down-signed altitude targets for automatic level-off expansion."""
        targets = {}
        for key, label in (("target_azimuth", "Turn to azimuth"),
                           ("target_altitude", "Level at altitude")):
            text = self.target_vars[key.removeprefix("target_")].get().strip()
            if not text:
                continue
            try:
                value = float(text)
            except ValueError:
                raise ValueError(f"'{label}' must be a number.")
            if not math.isfinite(value):
                raise ValueError(f"'{label}' must be finite.")
            targets[key] = value % 360.0 if key == "target_azimuth" else -value
        return targets

    def read_following_shift(self):
        """Return the enabled signed time adjustment for segments following the selection."""
        if not self.shift_following_var.get():
            return 0.0
        try:
            shift = float(self.shift_seconds_var.get().strip())
        except ValueError:
            raise ValueError("The following-segment time shift must be a number.")
        if not math.isfinite(shift):
            raise ValueError("The following-segment time shift must be finite.")
        return shift

    def read_system_location(self):
        """Return battery (north, east) meters, or None when both fields are blank.

        In GEO, project the entered WGS84 degrees relative to the ABT origin.
        A partially filled pair, invalid number, or invalid geographic range
        raises ValueError; absent battery data does not affect the flight model.
        """
        values = {key: var.get().strip() for key, var in self.system_vars.items()}
        if not values["north"] and not values["east"]:
            return None
        if not values["north"] or not values["east"]:
            raise ValueError("Enter both coordinates for the battery position.")
        try:
            north, east = float(values["north"]), float(values["east"])
        except ValueError:
            raise ValueError("Battery position coordinates must be numbers.")
        if not math.isfinite(north) or not math.isfinite(east):
            raise ValueError("Battery position coordinates must be finite.")
        if self.coordinate_type_var.get() == "GEO":
            return geographic_to_local(north, east, self.read_state()["geo_reference"])
        return north, east

    def selected_index(self):
        """Return the selected table row's segment index, or None when nothing is selected."""
        sel = self.tree.selection()
        return int(sel[0]) if sel else None

    def find_index(self, seg):
        """Find a segment's current list index by object identity."""
        return next(i for i, s in enumerate(self.segments) if s is seg)

    def target_group_for(self, seg):
        """Return the generated lock group containing a segment, if it has one."""
        return next((group for group in self.target_groups.values()
                     if any(item is seg for item in group["segments"])), None)

    # ---------- segment actions ----------
    def on_select(self, _event=None):
        """Copy the selected segment's values into the segment editor controls."""
        i = self.selected_index()
        if i is None:
            return
        group = self.target_group_for(self.segments[i])
        segment = group["segments"][0] if group else self.segments[i]
        for key, _, _ in SEGMENT_FIELDS:
            self.seg_vars[key].set(f"{segment[key]:g}")
        targets = group["targets"] if group else {}
        self.target_vars["azimuth"].set(f"{targets['target_azimuth']:g}" if "target_azimuth" in targets else "")
        self.target_vars["altitude"].set(f"{-targets['target_altitude']:g}" if "target_altitude" in targets else "")
        self.shift_following_var.set(False)

    def add_segment(self):
        """Validate an instruction and append its direct or automatically expanded segments."""
        try:
            seg = self.read_segment_editor()
            targets = self.read_target_options()
            state = self.read_state()
            additions = expand_target_segment(self.segments, seg, state, **targets)
        except ValueError as e:
            messagebox.showerror("Invalid segment", str(e), parent=self)
            return
        existing_times = {s["startTimeInSeconds"] for s in self.segments}
        addition_times = [item["startTimeInSeconds"] for item in additions]
        if len(set(addition_times)) != len(addition_times) or any(t in existing_times for t in addition_times):
            messagebox.showerror("Invalid segment", "A segment already starts at one of these times.", parent=self)
            return
        self.segments.extend(additions)
        if targets:
            group = {"segments": additions, "targets": targets}
            self.target_groups[id(additions[0])] = group
        self.refresh_tree()
        self.refresh_tree(self.find_index(additions[0]))
        # Suggest the next start time for quick entry.
        self.seg_vars["startTimeInSeconds"].set(f"{additions[-1]['startTimeInSeconds'] + 10:g}")
        for var in self.target_vars.values():
            var.set("")
        self.shift_following_var.set(False)
        self.mark_dirty()
        self.replot()

    def update_segment(self):
        """Replace a selected direct segment or regenerate its complete targeted maneuver."""
        i = self.selected_index()
        if i is None:
            messagebox.showinfo("Update segment", "Select a segment in the table first.", parent=self)
            return
        try:
            seg = self.read_segment_editor()
            targets = self.read_target_options()
            state = self.read_state()
            time_shift = self.read_following_shift()
        except ValueError as e:
            messagebox.showerror("Invalid segment", str(e), parent=self)
            return
        original = self.segments[i]
        group = self.target_group_for(original)
        replaced = group["segments"] if group else [original]
        replaced_ids = {id(item) for item in replaced}
        remaining = [item for item in self.segments if id(item) not in replaced_ids]
        old_group_end = max(item["startTimeInSeconds"] for item in replaced)
        prefix = [item for item in remaining if item["startTimeInSeconds"] < seg["startTimeInSeconds"]]
        stationary = [item for item in remaining if item["startTimeInSeconds"] <= old_group_end]
        following = [item for item in remaining if item["startTimeInSeconds"] > old_group_end]
        shifted_following = [
            dict(item, startTimeInSeconds=item["startTimeInSeconds"] + time_shift)
            for item in following
        ]
        if any(item["startTimeInSeconds"] < 0 for item in shifted_following):
            messagebox.showerror("Invalid segment", "The time shift would make a start time negative.", parent=self)
            return
        try:
            additions = expand_target_segment(prefix, seg, state, **targets)
        except ValueError as e:
            messagebox.showerror("Invalid segment", str(e), parent=self)
            return
        shifted_following.sort(key=lambda item: item["startTimeInSeconds"])
        if shifted_following and additions[-1]["startTimeInSeconds"] >= shifted_following[0]["startTimeInSeconds"]:
            messagebox.showerror(
                "Invalid segment", "The updated maneuver overlaps a following segment.", parent=self)
            return
        existing_times = {item["startTimeInSeconds"] for item in [*stationary, *shifted_following]}
        addition_times = [item["startTimeInSeconds"] for item in additions]
        if len(set(addition_times)) != len(addition_times) or any(t in existing_times for t in addition_times):
            messagebox.showerror("Invalid segment", "A segment already starts at one of these times.", parent=self)
            return

        if group:
            self.target_groups.pop(id(group["segments"][0]), None)
        self.segments = sorted([*stationary, *shifted_following, *additions],
                               key=lambda item: item["startTimeInSeconds"])
        if targets:
            self.target_groups[id(additions[0])] = {"segments": additions, "targets": targets}
        self.refresh_tree()
        self.refresh_tree(self.find_index(additions[0]))
        self.shift_following_var.set(False)
        self.mark_dirty()
        self.replot()

    def delete_segment(self):
        """Delete the selected segment and refresh the table and route plot."""
        i = self.selected_index()
        if i is None:
            return
        deleted = self.segments.pop(i)
        for base_id, group in tuple(self.target_groups.items()):
            if any(item is deleted for item in group["segments"]):
                del self.target_groups[base_id]
        self.refresh_tree(min(i, len(self.segments) - 1))
        self.mark_dirty()
        self.replot()

    def clear_segments(self):
        """Ask for confirmation, then remove all segments from the current route."""
        if self.segments and messagebox.askyesno("Clear all", "Delete all segments?", parent=self):
            self.segments.clear()
            self.target_groups.clear()
            self.refresh_tree()
            self.mark_dirty()
            self.replot()

    # ---------- file actions ----------
    def confirm_discard(self):
        """Ask whether to save pending edits and return whether the caller may proceed."""
        if not self.dirty:
            return True
        answer = messagebox.askyesnocancel("Unsaved changes", "Save changes first?", parent=self)
        if answer is None:
            return False
        return self.save() if answer else True

    def new_file(self):
        """Start a blank route after resolving any unsaved changes."""
        if not self.confirm_discard():
            return
        self._loading_route = True
        self.path_mode = "direct instructions"
        self.coordinate_type_var.set("BAT")
        self.waypoints = []
        self.route_tabs.select(0)
        self.refresh_waypoints()
        self._loading_route = False
        self.file_path = None
        self.segments = []
        self.target_groups.clear()
        self.set_state_values({})
        self.shift_following_var.set(False)
        for var in self.system_vars.values():
            var.set("")
        self.dirty = False
        self.refresh_tree()
        self.update_title()
        self.replot()

    def open_dialog(self):
        """Prompt for an instructions JSON file and open it if selected."""
        if not self.confirm_discard():
            return
        path = filedialog.askopenfilename(
            parent=self, title="Open flight instructions",
            initialdir=INSTRUCTIONS_DIR if INSTRUCTIONS_DIR.is_dir() else None,
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if path:
            self.open_file(Path(path))

    def open_file(self, path):
        """Load route metadata, instructions, origin, battery, and optional waypoints.

        BAT origin/battery records use North/East; GEO records require explicit
        Latitude/Longitude degrees. Earlier files labeled GEO but containing
        only North/East cannot be interpreted safely and are rejected. Parsing
        occurs before replacing the current editable route. Errors use a dialog.
        """
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            segments = [{key: float(raw[key]) for key, _, _ in SEGMENT_FIELDS} for raw in data["FlightSegments"]]
            path_mode, coordinate_type, waypoints = parse_route_type(data)
            raw_origin = data.get("OriginPosition", {})
            if not isinstance(raw_origin, dict):
                raise ValueError("'OriginPosition' must be an object.")
            state_values = {}
            for json_key, state_key, default in ORIGIN_FIELDS:
                if coordinate_type == "GEO" and state_key in ("north", "east"):
                    json_key = "Latitude" if state_key == "north" else "Longitude"
                    if json_key not in raw_origin:
                        raise ValueError("GEO OriginPosition requires Latitude and Longitude in degrees.")
                if json_key in raw_origin:
                    state_values[state_key] = float(raw_origin[json_key])
                elif state_key == "speed" and segments:
                    state_values[state_key] = segments[0]["targetSpeedInMetersPerSecond"]
                else:
                    state_values[state_key] = default
            if coordinate_type == "GEO":
                validate_geographic_position(state_values["north"], state_values["east"])
            raw_battery = data.get("BatteryPosition")
            battery_values = {}
            if raw_battery is not None:
                if not isinstance(raw_battery, dict):
                    raise ValueError("'BatteryPosition' must be an object.")
                keys = ("Latitude", "Longitude") if coordinate_type == "GEO" else ("North", "East")
                battery_values = {key: float(raw_battery[json_key])
                                  for key, json_key in zip(("north", "east"), keys)}
                if not all(math.isfinite(value) for value in battery_values.values()):
                    raise ValueError("Battery position coordinates must be finite.")
                if coordinate_type == "GEO":
                    validate_geographic_position(battery_values["north"], battery_values["east"])
        except (OSError, ValueError, KeyError, TypeError) as e:
            messagebox.showerror("Open failed", f"Could not read {path}:\n{e}", parent=self)
            return
        self.file_path = path
        self.segments = segments
        self._loading_route = True
        self.path_mode = path_mode
        self.coordinate_type_var.set(coordinate_type)
        self.waypoints = waypoints
        self.route_tabs.select(1 if path_mode == "waypoints based" else 0)
        self.refresh_waypoints()
        self.set_state_values(state_values)
        self._loading_route = False
        self.shift_following_var.set(False)
        try:
            self.target_groups = infer_target_groups(segments, self.read_state())
        except ValueError as e:
            self.target_groups = {}
            messagebox.showwarning("Target locks", f"Could not infer saved target locks:\n{e}", parent=self)
        for key, var in self.system_vars.items():
            var.set(f"{battery_values[key]:.12g}" if key in battery_values else "")
        self.dirty = False
        self.refresh_tree()
        self.update_title()
        self.replot()

    def save(self):
        """Save to the current route path, prompting for a path when the route is untitled."""
        if self.file_path is None:
            return self.save_as()
        return self.write_file(self.file_path)

    def save_as(self):
        """Prompt for a destination path and save the current route there."""
        path = filedialog.asksaveasfilename(
            parent=self, title="Save flight instructions", defaultextension=".json",
            initialdir=INSTRUCTIONS_DIR if INSTRUCTIONS_DIR.is_dir() else None,
            initialfile=self.file_path.name if self.file_path else "NEW_ROUTE.json",
            filetypes=[("JSON files", "*.json")])
        if not path:
            return False
        return self.write_file(Path(path))

    def customized_save(self):
        """Save flight segments, initial state, and battery position to three separate JSON files.

        Given a chosen path myname.json the method writes:
          myname.json         — FlightSegments only
          myname_inistate.json — OriginPosition
          myname_battery.json  — BatteryPosition (null when not set)
        """
        if not self.segments:
            messagebox.showerror("Save failed", "Add at least one segment before saving.", parent=self)
            return
        path = filedialog.asksaveasfilename(
            parent=self, title="Customized Save — choose base filename",
            defaultextension=".json",
            initialdir=INSTRUCTIONS_DIR if INSTRUCTIONS_DIR.is_dir() else None,
            initialfile=self.file_path.name if self.file_path else "myroute.json",
            filetypes=[("JSON files", "*.json")])
        if not path:
            return
        base = Path(path)
        stem = base.stem
        folder = base.parent
        try:
            state = self.read_state()
        except ValueError as e:
            messagebox.showerror("Save failed", str(e), parent=self)
            return
        try:
            battery_location = self.read_system_location()
        except ValueError as e:
            messagebox.showerror("Save failed", str(e), parent=self)
            return

        # Load optional overrides from {characteristics_stem}_extra.json.
        extra_config = {}
        if self.limits_path is not None:
            extra_path = self.limits_path.with_name(self.limits_path.stem + "_extra.json")
            if extra_path.is_file():
                try:
                    extra_config = json.loads(extra_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    pass

        initial_altitude = extra_config.get("InitialAltitude", 1000)
        battery_altitude = extra_config.get("BatteryAltitude", 0)
        delta_seconds = float(extra_config.get("delta_seconds", TRAJECTORY_DELTA))

        # Use BatteryLocation from extra config when the GUI has no battery set.
        battery_from_extra = False
        if battery_location is None and "BatteryLocation" in extra_config:
            loc = extra_config["BatteryLocation"]
            try:
                battery_location = (float(loc["North"]), float(loc["East"]))
                battery_from_extra = True
            except (KeyError, TypeError, ValueError):
                pass

        segments_data = {
            "FlightSegments": [
                {key: float(seg[key]) for key, _, _ in SEGMENT_FIELDS}
                for seg in self.segments
            ]
        }

        origin = {json_key: float(state[state_key]) for json_key, state_key, _ in ORIGIN_FIELDS}
        geographic = self.coordinate_type_var.get() == "GEO"
        if geographic:
            del origin["North"], origin["East"]
            origin["Latitude"], origin["Longitude"] = state["geo_reference"]
        origin["Altitude"] = initial_altitude
        inistate_data = {"OriginPosition": origin}

        if battery_location is not None:
            if battery_from_extra:
                battery_value = {"North": battery_location[0], "East": battery_location[1]}
            else:
                keys = ("Latitude", "Longitude") if geographic else ("North", "East")
                battery_value = {
                    json_key: float(self.system_vars[key].get())
                    for key, json_key in zip(("north", "east"), keys)
                }
            battery_value["Altitude"] = battery_altitude
        else:
            battery_value = None
        battery_data = {"BatteryPosition": battery_value}

        # Build trajectory: sample simulation at delta_seconds intervals.
        # World frame: battery = [0,0,0], X=East, Y=North, Z=Up.
        full_traj = simulate(self.segments, state, include_state=True)
        t_all, n_all, e_all, d_all, _, rol_all, pit_all, hdg_all, _ = full_traj
        bat_north = battery_location[0] if battery_location is not None else 0.0
        bat_east  = battery_location[1] if battery_location is not None else 0.0
        step = max(1, round(delta_seconds / DT))
        trajectory = []
        for rec_id, idx in enumerate(range(0, len(t_all), step)):
            trajectory.append({
                "id": rec_id,
                "time_stamp": t_all[idx],
                "X": e_all[idx] - bat_east,
                "Y": n_all[idx] - bat_north,
                "Z": initial_altitude + state["down"] - d_all[idx],
                "Yaw": hdg_all[idx] % 360.0,
                "Pitch": pit_all[idx],
                "Roll": rol_all[idx],
            })
        trajectory_data = {"trajectory": trajectory}

        route_png = folder / f"{stem}_route.png"
        try:
            (folder / f"{stem}.json").write_text(
                json.dumps(segments_data, indent=2) + "\n", encoding="utf-8")
            (folder / f"{stem}_inistate.json").write_text(
                json.dumps(inistate_data, indent=2) + "\n", encoding="utf-8")
            (folder / f"{stem}_battery.json").write_text(
                json.dumps(battery_data, indent=2) + "\n", encoding="utf-8")
            (folder / f"{stem}_trajectory.json").write_text(
                json.dumps(trajectory_data, indent=2) + "\n", encoding="utf-8")
            self.figure.savefig(str(route_png), dpi=150)
        except OSError as e:
            messagebox.showerror("Save failed", str(e), parent=self)
            return
        messagebox.showinfo(
            "Customized Save",
            f"Saved five files in {folder}:\n"
            f"  {stem}.json\n"
            f"  {stem}_inistate.json\n"
            f"  {stem}_battery.json\n"
            f"  {stem}_trajectory.json  ({len(trajectory)} samples @ {TRAJECTORY_DELTA} s)\n"
            f"  {stem}_route.png",
            parent=self)

    def write_file(self, path):
        """Regenerate waypoint controls when needed and save a validated route JSON.

        Every file contains type, OriginPosition, and FlightSegments. Waypoint
        mode adds a top-level coordinates list; BatteryPosition is optional.
        Origin/battery geographic keys are capitalized, while waypoint geographic
        keys are lowercase. Returns False on validation or I/O failure; the dirty
        flag is cleared only after the file has been written successfully.
        """
        if self.path_mode == "waypoints based":
            if not self.generate_waypoint_route():
                return False
        if not self.segments:
            messagebox.showerror("Save failed", "Add at least one segment before saving.", parent=self)
            return False
        try:
            state = self.read_state()
            battery = self.read_system_location()
        except ValueError as e:
            messagebox.showerror("Save failed", str(e), parent=self)
            return False
        origin = {json_key: float(state[state_key]) for json_key, state_key, _ in ORIGIN_FIELDS}
        geographic = self.coordinate_type_var.get() == "GEO"
        if geographic:
            del origin["North"], origin["East"]
            origin["Latitude"], origin["Longitude"] = state["geo_reference"]
        data = {
            "type": build_route_type(self.path_mode, self.coordinate_type_var.get()),
        }
        if self.path_mode == "waypoints based":
            data["coordinates"] = [
                {({"north": "latitude", "east": "longitude"}.get(key, key) if geographic else key):
                 float(waypoint[key]) for key, _, _ in (*WAYPOINT_FIELDS, *LEG_LIMIT_FIELDS)}
                for waypoint in self.waypoints
            ]
        if battery is not None:
            keys = ("Latitude", "Longitude") if geographic else ("North", "East")
            data["BatteryPosition"] = {
                json_key: float(self.system_vars[key].get())
                for key, json_key in zip(("north", "east"), keys)
            }
        data.update({
            "OriginPosition": origin,
            "FlightSegments": [{key: float(seg[key]) for key, _, _ in SEGMENT_FIELDS} for seg in self.segments],
        })
        try:
            path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        except OSError as e:
            messagebox.showerror("Save failed", str(e), parent=self)
            return False
        self.file_path = path
        self.dirty = False
        self.update_title()
        return True

    def export_plot(self):
        """Prompt for an image destination and export the currently displayed plot."""
        stem = self.file_path.stem if self.file_path else "route"
        path = filedialog.asksaveasfilename(
            parent=self, title="Export plot", defaultextension=".png",
            initialdir=self.file_path.parent if self.file_path else None,
            initialfile=f"{stem}_route_profile.png",
            filetypes=[("PNG image", "*.png"), ("SVG image", "*.svg"), ("PDF", "*.pdf")])
        if path:
            self.figure.savefig(path, dpi=150)

    def on_close(self):
        """Close the editor if the user accepts or resolves any unsaved changes."""
        if self.confirm_discard():
            self.destroy()

    # ---------- limits ----------
    def choose_limits_file(self):
        """Prompt for a flight-characteristics JSON file and load it as the limits source."""
        start = self.limits_path.parent if self.limits_path else CHARACTERISTICS_DIR
        path = filedialog.askopenfilename(
            parent=self, title="Select flight characteristics (limits) file",
            initialdir=start if start.is_dir() else None,
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if path:
            self.set_limits_file(Path(path))

    def set_limits_file(self, path, show_errors=True):
        """Load or clear flight limits, update their label, and refresh the plot."""
        if path is None:
            self.limits, self.limits_path = None, None
            self.limits_label.config(text="(none – limit checks disabled)")
        else:
            try:
                limits = load_limits(path)
            except ValueError as e:
                if show_errors:
                    messagebox.showerror("Invalid limits file", f"{path}\n\n{e}", parent=self)
                return
            self.limits, self.limits_path = limits, Path(path)
            self.limits_label.config(text=str(self.limits_path))
        if hasattr(self, "segments"):
            self.replot()

    # ---------- plotting ----------
    def read_state(self):
        """Return a validated simulation state, converting GEO display values to NED.

        GEO origin latitude/longitude is retained in geo_reference; local north
        and east are then zero. Down stays in meters and headings stay in degrees.
        Without an explicit end time, waypoint plots stop at validated arrival;
        direct schedules continue 12 seconds after their final command starts.
        Raises ValueError for invalid/nonfinite controls or excessive duration.
        """
        state = {}
        for key, label, _ in STATE_FIELDS:
            text = self.state_vars[key].get().strip()
            if key == "end_time" and not text:
                last_start = self.segments[-1]["startTimeInSeconds"] if self.segments else 0.0
                state[key] = max(0.01, last_start) if self.path_mode == "waypoints based" and self.segments \
                    else last_start + 12.0
                continue
            try:
                state[key] = float(text)
            except ValueError:
                raise ValueError(f"Initial state '{label}' must be a number.")
            if not math.isfinite(state[key]):
                raise ValueError(f"Initial state '{label}' must be finite.")
        if self.coordinate_type_var.get() == "GEO":
            validate_geographic_position(state["north"], state["east"])
            state["geo_reference"] = (state["north"], state["east"])
            state["north"], state["east"] = 0.0, 0.0
        if state["speed"] < 0:
            raise ValueError("Initial speed must be >= 0.")
        if state["end_time"] <= 0:
            raise ValueError("End time must be > 0.")
        if state["end_time"] / 0.01 > 2_000_000:
            raise ValueError("End time is too long to simulate.")
        return state

    def zoom_plot(self, event):
        """Zoom the hovered plot around the pointer in or out by one scroll step."""
        axis = event.inaxes
        if self.toolbar.mode or axis not in (self.ax, self.altitude_ax) \
                or event.xdata is None or event.ydata is None:
            return
        factor = {"up": 1 / 1.2, "down": 1.2}.get(event.button)
        if factor is None:
            return
        x_min, x_max = axis.get_xlim()
        y_min, y_max = axis.get_ylim()
        axis.set_xlim(event.xdata + (x_min - event.xdata) * factor,
                      event.xdata + (x_max - event.xdata) * factor, auto=True)
        axis.set_ylim(event.ydata + (y_min - event.ydata) * factor,
                      event.ydata + (y_max - event.ydata) * factor, auto=True)
        self.canvas.draw_idle()

    def start_plot_pan(self, event):
        """Begin direct left-button panning when a plot axis is clicked."""
        if self.toolbar.mode or event.button != 1 or event.inaxes not in (self.ax, self.altitude_ax):
            return
        if event.x is None or event.y is None:
            return
        axis = event.inaxes
        self._pan_drag = (axis, event.x, event.y, axis.get_xlim(), axis.get_ylim())

    def drag_plot(self, event):
        """Move the active plot's view by the pointer's drag distance."""
        if self._pan_drag is None or event.x is None or event.y is None:
            return
        axis, start_x, start_y, x_limits, y_limits = self._pan_drag
        bounds = axis.bbox
        if bounds.width <= 0 or bounds.height <= 0:
            return
        x_shift = -(event.x - start_x) * (x_limits[1] - x_limits[0]) / bounds.width
        y_shift = -(event.y - start_y) * (y_limits[1] - y_limits[0]) / bounds.height
        axis.set_xlim(x_limits[0] + x_shift, x_limits[1] + x_shift, auto=True)
        axis.set_ylim(y_limits[0] + y_shift, y_limits[1] + y_shift, auto=True)
        self.canvas.draw_idle()

    def stop_plot_pan(self, _event=None):
        """Finish direct panning when the mouse button is released."""
        self._pan_drag = None

    def reset_view(self):
        """Restore both plots to the axis limits captured after the last route redraw."""
        if self._original_view is None:
            return
        for axis, x_limits, y_limits in self._original_view:
            axis.set_xlim(x_limits, auto=True)
            axis.set_ylim(y_limits, auto=True)
        self._pan_drag = None
        self.canvas.draw_idle()

    def plot_route(self):
        """Regenerate waypoint controls from the current origin before plotting."""
        if self.path_mode == "waypoints based" and self.waypoints:
            self.waypoints_changed()
        else:
            self.replot()

    def replot(self):
        """Render current instructions and markers; do not generate or fit a route.

        plot_route performs regeneration when requested by the user. Both plots
        here use the same local-meter simulation; only top-down display positions
        are converted to degrees in GEO mode. Waypoint status reports the nearest
        3D sample, not the fitter's recorded leg boundary, and is diagnostic only.
        """
        self._original_view = None
        self._pan_drag = None
        self.ax.clear()
        self.altitude_ax.clear()
        if not self.segments:
            if self.path_mode == "waypoints based" and self.waypoints:
                self.ax.set_axis_on()
                try:
                    state = self.read_state()
                    draw_waypoints(self.ax, self.planning_waypoints(state), state.get("geo_reference"))
                    configure_route_axes(self.ax, state.get("geo_reference"))
                except ValueError as error:
                    self.canvas.draw_idle()
                    self.set_status([str(error)], error=True)
                    return
                self.ax.set_title("Waypoint locations")
                self.ax.grid(True, alpha=0.4)
                self.altitude_ax.text(0.5, 0.5, "Altitude profile", ha="center", va="center",
                                      transform=self.altitude_ax.transAxes, fontsize=13, color="#777")
                self.altitude_ax.set_axis_off()
                self.canvas.draw_idle()
                lines = ["No generated instructions; waypoint locations are shown."]
                warnings = waypoint_limit_warnings(self.waypoints, self.limits)
                if self.limits:
                    if warnings:
                        lines.append(f"Exceeds {self.limits_path.name} limits:")
                        lines.extend(f"  • {warning}" for warning in warnings)
                else:
                    lines.append("WARNING: Aircraft characteristics not loaded; waypoint limits cannot be checked.")
                self.set_status(lines, error=bool(warnings) or not self.limits)
                return
            self.ax.text(0.5, 0.5, "Add flight segments or open a file", ha="center", va="center",
                         transform=self.ax.transAxes, fontsize=13, color="#777")
            self.altitude_ax.text(0.5, 0.5, "Altitude profile", ha="center", va="center",
                                  transform=self.altitude_ax.transAxes, fontsize=13, color="#777")
            self.ax.set_axis_off()
            self.altitude_ax.set_axis_off()
            self.canvas.draw_idle()
            self.set_status(["No segments."])
            return
        self.ax.set_axis_on()
        self.altitude_ax.set_axis_on()
        try:
            state = self.read_state()
            system_location = self.read_system_location()
        except ValueError as e:
            self.canvas.draw_idle()
            self.set_status([str(e)], error=True)
            return
        full_result = simulate(self.segments, state, include_state=True)
        result = full_result[:5]
        name = self.file_path.stem if self.file_path else "Untitled"
        draw_route(self.ax, name, self.segments, state, result, system_location=system_location)
        if self.path_mode == "waypoints based":
            draw_waypoints(self.ax, self.planning_waypoints(state), state.get("geo_reference"))
        draw_altitude_profile(self.altitude_ax, self.segments, state, result)
        self.figure.tight_layout()
        self.canvas.draw()
        self._original_view = tuple(
            (axis, axis.get_xlim(), axis.get_ylim()) for axis in (self.ax, self.altitude_ax))

        _, norths, easts, downs, _ = result
        lines = []
        if self.path_mode == "waypoints based":
            lines.append("WARNING: Real-aircraft flight may have waypoint position and azimuth errors; verify before use.")
            for index, waypoint in enumerate(self.planning_waypoints(state), start=1):
                closest = min(
                    range(len(norths)),
                    key=lambda sample: (
                        (norths[sample] - waypoint["north"]) ** 2
                        + (easts[sample] - waypoint["east"]) ** 2
                        + (downs[sample] - waypoint["down"]) ** 2),
                )
                track_azimuth = full_result[7][closest] % 360.0
                azimuth_error = (track_azimuth - waypoint["azimuth"] + 180.0) % 360.0 - 180.0
                range_error = math.hypot(
                    norths[closest] - waypoint["north"], easts[closest] - waypoint["east"])
                lines.append(
                    f"WP{index} closest approach: {range_error:.1f} m; Down error "
                    f"{downs[closest] - waypoint['down']:+.1f} m; azimuth error {azimuth_error:+.2f}°")
        if "geo_reference" in state:
            latitude, longitude = local_to_geographic(norths[-1], easts[-1], state["geo_reference"])
            lines.append(f"End Lat/Lon: ({latitude:.6f}, {longitude:.6f}) deg; Down {downs[-1]:.1f} m")
        else:
            lines.append(f"End NED: ({norths[-1]:.1f}, {easts[-1]:.1f}, {downs[-1]:.1f}) m  "
                         f"(height above origin {-downs[-1]:.1f} m, climbed {downs[0] - downs[-1]:+.1f} m)")
        warnings = limit_warnings(self.segments, self.limits, state, downs)
        limits_unavailable = self.path_mode == "waypoints based" and not self.limits
        if self.path_mode == "waypoints based":
            warnings.extend(waypoint_limit_warnings(self.waypoints, self.limits))
            if limits_unavailable:
                lines.append("WARNING: Aircraft characteristics not loaded; waypoint limits cannot be checked.")
        if warnings:
            heading = f"Exceeds {self.limits_path.name} limits:" if self.limits_path else "Flight route warnings:"
            lines.append(heading)
            lines.extend(f"  • {w}" for w in warnings)
        self.set_status(lines, error=bool(warnings) or limits_unavailable)


def main():
    """Create the route editor and run its Tk event loop."""
    app = FlightRouteApp(sys.argv[1] if len(sys.argv) > 1 else None)
    app.mainloop()


if __name__ == "__main__":
    main()
