r"""Simulate ABT instruction schedules, fit waypoint arrivals, and render routes.

Usage:
    python PlotFlightRoute.py [path\to\instructions.json]

The simulation uses local North/East/Down meters, seconds, degrees, and m/s.
Headings increase clockwise from North; positive bank turns right and positive
pitch climbs. Down is negative above the vertical reference, not MSL altitude.
Integration uses a fixed 0.01-second step and gravity of 9.80665 m/s^2.

Waypoint planning selects the shortest geometric path from all six Dubins families,
then fits a compact instruction schedule against the rate-limited simulator.
The fitted trace is not necessarily the shortest dynamically feasible trajectory.
GEO input/output uses WGS84, but flight dynamics remain in a local flat plane.
Plots show the simulated instructions, not a separately drawn ideal Dubins curve.

The standalone CLI prompts for a local NED state; geographic file input is handled
by FlightRouteGui. This model is not a real-aircraft performance guarantee.
"""

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from geographiclib.geodesic import Geodesic

GRAVITY = 9.80665
DT = 0.01


def validate_geographic_position(latitude, longitude):
    """Validate WGS84 degrees, excluding poles where the local plot aspect is singular.

    Raises ValueError for nonfinite values, latitude outside (-90, 90), or
    longitude outside [-180, 180]. Inputs are decimal degrees, not radians or DMS.
    """
    if not math.isfinite(latitude) or not -90.0 < latitude < 90.0:
        raise ValueError("Latitude must be finite and strictly between -90 and 90 degrees.")
    if not math.isfinite(longitude) or not -180.0 <= longitude <= 180.0:
        raise ValueError("Longitude must be finite and between -180 and 180 degrees.")


def geographic_to_local(latitude, longitude, reference):
    """Return (north, east) meters relative to a (latitude, longitude) reference.

    WGS84 inverse geodesics supply distance and initial azimuth. Resolving those
    into North/East gives an azimuthal equidistant projection centered on the ABT
    origin. Distances from that origin are preserved, not arbitrary distances
    between remote projected points. All geographic inputs are decimal degrees.
    """
    validate_geographic_position(latitude, longitude)
    validate_geographic_position(*reference)
    inverse = Geodesic.WGS84.Inverse(reference[0], reference[1], latitude, longitude)
    azimuth = math.radians(inverse["azi1"])
    return inverse["s12"] * math.cos(azimuth), inverse["s12"] * math.sin(azimuth)


def local_to_geographic(north, east, reference):
    """Return WGS84 (latitude, longitude) degrees for local North/East meters.

    Uses the geodesic direct problem with the same reference as
    geographic_to_local. Longitude is normalized by GeographicLib to [-180, 180].
    """
    validate_geographic_position(*reference)
    destination = Geodesic.WGS84.Direct(
        reference[0], reference[1], math.degrees(math.atan2(east, north)), math.hypot(north, east))
    return destination["lat2"], destination["lon2"]


def route_plot_position(north, east, reference=None):
    """Return plot (x, y): (east, north) meters or (longitude, latitude) degrees.

    Longitudes are unwrapped around the reference for plotting across the date
    line. Axis values can therefore exceed 180 degrees even though saved input
    coordinates remain normalized WGS84 longitude.
    """
    if reference is None:
        return east, north
    latitude, longitude = local_to_geographic(north, east, reference)
    longitude = reference[1] + (longitude - reference[1] + 180.0) % 360.0 - 180.0
    return longitude, latitude


def configure_route_axes(ax, reference=None):
    """Set meter or degree axes using an approximate local geographic aspect.

    The GEO aspect corrects longitude scale using cos(reference latitude). It is
    intended for local flight routes, not a global map projection or basemap.
    """
    if reference is None:
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("East [m]")
        ax.set_ylabel("North [m]")
    else:
        ax.set_aspect(1.0 / math.cos(math.radians(reference[0])), adjustable="datalim")
        ax.set_xlabel("Longitude [deg]")
        ax.set_ylabel("Latitude [deg]")
        ax.ticklabel_format(axis="both", useOffset=False, style="plain")


def ask_float(prompt, default):
    """Prompt repeatedly until the user enters a number or accepts the default."""
    while True:
        text = input(f"{prompt} [{default}]: ").strip()
        if not text:
            return float(default)
        try:
            return float(text)
        except ValueError:
            print("  Please enter a number.")


def ask_json_path():
    """Return the instructions path from the command line or prompt until one exists."""
    if len(sys.argv) > 1:
        return Path(sys.argv[1].strip('"'))
    while True:
        path = Path(input("Path to flight instructions JSON: ").strip().strip('"'))
        if path.is_file():
            return path
        print(f"  File not found: {path}")


def load_segments(path):
    """Load flight segments from JSON, sort them by start time, and reject empty routes."""
    data = json.loads(path.read_text(encoding="utf-8"))
    segments = sorted(data["FlightSegments"], key=lambda s: s["startTimeInSeconds"])
    if not segments:
        raise ValueError("No FlightSegments found in file.")
    return segments


def ask_initial_state(segments):
    """Prompt for the initial aircraft state and simulation duration."""
    print("\nInitial state (press Enter to accept the default):")
    north = ask_float("  North position [m]", 0.0)
    east = ask_float("  East position [m]", 0.0)
    down = ask_float("  Down position [m] (negative = above origin)", 0.0)
    roll = ask_float("  Roll [deg]", 0.0)
    pitch = ask_float("  Pitch [deg]", 0.0)
    heading = ask_float("  Heading [deg, 0 = North]", 0.0)
    speed = ask_float("  Speed [m/s]", segments[0]["targetSpeedInMetersPerSecond"])
    end_default = segments[-1]["startTimeInSeconds"] + 12.0
    end_time = ask_float("  Simulation end time [s]", end_default)
    if speed <= 0:
        raise ValueError("Initial speed must be positive.")
    return dict(north=north, east=east, down=down, roll=roll, pitch=pitch,
                heading=heading, speed=speed, end_time=end_time)


def step_toward(value, target, max_step):
    """Move a value toward its target by at most max_step, without overshooting."""
    return value + max(-max_step, min(max_step, target - value))


def simulate(segments, state, include_state=False):
    """Integrate a sorted, nonempty instruction schedule from time zero.

    Args:
        segments: Chronological FlightSegments records using the documented JSON
            instruction fields. Rates are nonnegative magnitudes.
        state: Finite north/east/down meters, roll/pitch/heading degrees, speed
            m/s, and positive end_time seconds. Coordinates are always local,
            including when GEO is selected in the editor.
        include_state: Append roll, pitch, heading, and speed histories when true.

    Returns:
        Parallel lists (times, norths, easts, downs, segment_ids), optionally
        followed by (rolls, pitches, headings, speeds). Heading histories are
        unwrapped degrees. The last sample is on the fixed DT integration grid.

    Bank, pitch, and speed move toward their commands at the supplied rates.
    Heading rate is g*tan(bank)/speed; horizontal speed is speed*cos(pitch).
    The model does not include wind, stall, load-factor limits, or actuator lag
    beyond the explicit rate limits. Callers must validate the input schedule.
    """
    north, east, down = state["north"], state["east"], state["down"]
    roll = math.radians(state["roll"])
    pitch = math.radians(state["pitch"])
    heading = math.radians(state["heading"])
    speed = state["speed"]

    times, norths, easts, downs, seg_ids = [0.0], [north], [east], [down], [0]
    if include_state:
        rolls, pitches = [state["roll"]], [state["pitch"]]
        headings, speeds = [state["heading"]], [speed]
    active = 0
    t = 0.0
    while t < state["end_time"] - 1e-9:
        # Grid-aligned event times must not slip a full step due to float accumulation.
        while active + 1 < len(segments) and t + 1e-9 >= segments[active + 1]["startTimeInSeconds"]:
            active += 1
        seg = segments[active]
        if t + 1e-9 >= seg["startTimeInSeconds"]:
            roll = step_toward(roll, math.radians(seg["targetBankInDegrees"]),
                               math.radians(seg["rollRateInDegreesPerSecond"]) * DT)
            pitch = step_toward(pitch, math.radians(seg["targetPitchInDegrees"]),
                                math.radians(seg["pitchRateInDegreesPerSecond"]) * DT)
            speed = step_toward(speed, seg["targetSpeedInMetersPerSecond"],
                                abs(seg["accelerationInMetersPerSecondSquared"]) * DT)

        heading += GRAVITY * math.tan(roll) / max(speed, 1e-3) * DT
        horizontal = speed * math.cos(pitch)
        north += horizontal * math.cos(heading) * DT
        east += horizontal * math.sin(heading) * DT
        # NED: Down is negative above the origin, so climbing (positive pitch) makes Down more negative.
        down -= speed * math.sin(pitch) * DT
        t += DT

        times.append(t)
        norths.append(north)
        easts.append(east)
        downs.append(down)
        seg_ids.append(active)
        if include_state:
            rolls.append(math.degrees(roll))
            pitches.append(math.degrees(pitch))
            headings.append(math.degrees(heading))
            speeds.append(speed)
    result = times, norths, easts, downs, seg_ids
    if include_state:
        return (*result, rolls, pitches, headings, speeds)
    return result


def _mod2pi(angle):
    """Normalize a radian angle into [0, 2*pi)."""
    return angle % (2.0 * math.pi)


def _dubins_path_segments(start, goal, radius):
    """Select the shortest geometric path from LSL, RSR, LSR, RSL, RLR, and LRL.

    start and goal are Cartesian (east, north, heading_radians) poses. Here the
    heading is counterclockwise from East, unlike the simulator's clockwise
    azimuth from North. Radius is a positive meter value.

    Returns three (kind, length_meters) pairs. L/R are constant-radius arcs and
    S is straight; the middle primitive is also an arc in RLR/LRL. These are
    ideal geometric lengths, not rate-limited command durations. Loops and turns
    over 180 degrees are allowed. No external Dubins library is used.
    """
    start_x, start_y, start_heading = start
    goal_x, goal_y, goal_heading = goal
    dx, dy = (goal_x - start_x) / radius, (goal_y - start_y) / radius
    distance = math.hypot(dx, dy)
    direction = math.atan2(dy, dx)
    alpha = _mod2pi(start_heading - direction)
    beta = _mod2pi(goal_heading - direction)
    sin_alpha, sin_beta = math.sin(alpha), math.sin(beta)
    cos_alpha, cos_beta = math.cos(alpha), math.cos(beta)
    cos_delta = math.cos(alpha - beta)

    # Normalize translation and radius so candidate lengths use unit-radius geometry.
    candidates = []
    p_squared = 2 + distance ** 2 - 2 * cos_delta + 2 * distance * (sin_alpha - sin_beta)
    if p_squared >= -1e-10:
        p_squared = max(0.0, p_squared)
        angle = math.atan2(cos_beta - cos_alpha, distance + sin_alpha - sin_beta)
        candidates.append(("LSL", _mod2pi(-alpha + angle), math.sqrt(p_squared),
                           _mod2pi(beta - angle)))

    p_squared = 2 + distance ** 2 - 2 * cos_delta + 2 * distance * (-sin_alpha + sin_beta)
    if p_squared >= -1e-10:
        p_squared = max(0.0, p_squared)
        angle = math.atan2(cos_alpha - cos_beta, distance - sin_alpha + sin_beta)
        candidates.append(("RSR", _mod2pi(alpha - angle), math.sqrt(p_squared),
                           _mod2pi(-beta + angle)))

    p_squared = -2 + distance ** 2 + 2 * cos_delta + 2 * distance * (sin_alpha + sin_beta)
    if p_squared >= -1e-10:
        p = math.sqrt(max(0.0, p_squared))
        angle = math.atan2(-cos_alpha - cos_beta, distance + sin_alpha + sin_beta) \
            - math.atan2(-2.0, p)
        candidates.append(("LSR", _mod2pi(-alpha + angle), p, _mod2pi(-beta + angle)))

    p_squared = -2 + distance ** 2 + 2 * cos_delta - 2 * distance * (sin_alpha + sin_beta)
    if p_squared >= -1e-10:
        p = math.sqrt(max(0.0, p_squared))
        angle = math.atan2(cos_alpha + cos_beta, distance - sin_alpha - sin_beta) \
            - math.atan2(2.0, p)
        candidates.append(("RSL", _mod2pi(alpha - angle), p, _mod2pi(beta - angle)))

    value = (6 - distance ** 2 + 2 * cos_delta + 2 * distance * (sin_alpha - sin_beta)) / 8
    if abs(value) <= 1 + 1e-10:
        p = _mod2pi(2 * math.pi - math.acos(max(-1.0, min(1.0, value))))
        angle = math.atan2(cos_alpha - cos_beta, distance - sin_alpha + sin_beta)
        t = _mod2pi(alpha - angle + p / 2)
        candidates.append(("RLR", t, p, _mod2pi(alpha - beta - t + p)))

    value = (6 - distance ** 2 + 2 * cos_delta + 2 * distance * (-sin_alpha + sin_beta)) / 8
    if abs(value) <= 1 + 1e-10:
        p = _mod2pi(2 * math.pi - math.acos(max(-1.0, min(1.0, value))))
        angle = math.atan2(cos_alpha - cos_beta, distance + sin_alpha - sin_beta)
        t = _mod2pi(-alpha - angle + p / 2)
        candidates.append(("LRL", t, p, _mod2pi(beta - alpha - t + p)))

    if not candidates:
        raise ValueError("No feasible Dubins path is available for this leg.")
    family, first, middle, last = min(candidates, key=lambda path: path[1] + path[2] + path[3])
    return [(kind, value * radius)
            for kind, value in zip(family, (first, middle, last))]


def _fit_dubins_leg(path_segments, initial_state, waypoint, limits, start_time,
                    control_interval, max_leg_time, radius):
    """Fit a Dubins-seeded schedule to a simultaneous simulated arrival pose.

    Parameter order is primitive durations, magnitudes of each turning bank,
    then a common en-route pitch. A final straight/level settling phase is fitted
    too, so arrival occurs after roll-out and pitch level-off. Durations are
    quantized to DT; control_interval seeds nominal/minimum durations, not an
    online feedback cadence.

    Bounded, damped least squares uses finite-difference Jacobians. Residuals are
    North/East/Down errors in meters and heading error in radians multiplied by
    turn radius, giving the angular component a comparable distance scale.

    Returns (segments, final_state, duration_seconds). Acceptance requires <=10 m
    horizontal error, <=10 m vertical error, <=2 degrees azimuth error, and
    <=0.1 m/s speed error at the same endpoint. Raises ValueError on failure.
    Optimization is local and iteration-limited: failure does not prove that no
    dynamically feasible path exists, and other Dubins families are not retried.
    """
    path_segments = [(kind, length) for kind, length in path_segments if length > 1e-6]
    if not path_segments:
        target_down = float(waypoint["down"] if "down" in waypoint else -waypoint["altitude"])
        if abs(initial_state["down"] - target_down) > 10.0 \
            or abs(initial_state["speed"] - float(waypoint["speed"])) > 0.1:
            raise ValueError("A zero-length Dubins leg cannot change altitude or speed.")
        return [], dict(initial_state), 0.0

    max_bank = abs(float(limits["maxBankAngle"]))
    roll_rate = float(limits["maxRollRate"])
    max_pitch_up = abs(float(limits["maxPitchUpAngle"]))
    max_pitch_down = abs(float(limits["maxPitchDownAngle"]))
    pitch_rate = float(limits["maxPitchRate"])
    acceleration = abs(float(limits["maxAccel"]))
    target_speed = float(waypoint["speed"])
    target_heading = float(waypoint["azimuth"]) % 360.0
    target_down = float(waypoint["down"] if "down" in waypoint else -waypoint["altitude"])
    average_speed = max((initial_state["speed"] + target_speed) / 2.0, 1.0)
    settling_duration = max(max_bank / roll_rate,
                            max(max_pitch_up, max_pitch_down) / pitch_rate,
                            control_interval) + DT
    nominal_durations = np.array([max(control_interval, length / average_speed)
                                  for _, length in path_segments] + [settling_duration])
    # The zero geometric length represents a fitted period of straight, level flight.
    path_segments = [*path_segments, ("S", 0.0)]
    turn_indices = [index for index, (kind, _) in enumerate(path_segments) if kind != "S"]
    initial_pitch = math.degrees(math.atan2(
        initial_state["down"] - target_down, max(sum(length for _, length in path_segments), 1.0)))
    initial_pitch = max(-max_pitch_down, min(max_pitch_up, initial_pitch))
    parameters = np.concatenate((nominal_durations, np.full(len(turn_indices), max_bank), [initial_pitch]))
    scales = np.concatenate((np.maximum(nominal_durations, 5.0), np.full(len(turn_indices), 10.0), [10.0]))
    lower = np.concatenate((np.full(len(path_segments), max(0.05, control_interval * 0.1)),
                            np.zeros(len(turn_indices)), [-max_pitch_down]))
    lower[len(path_segments) - 1] = settling_duration
    upper = np.concatenate((nominal_durations + np.maximum(30.0, nominal_durations * 0.5),
                            np.full(len(turn_indices), max_bank), [max_pitch_up]))

    def evaluate(candidate):
        """Simulate one bounded candidate; return endpoint residual, state, and commands."""
        durations = candidate[:len(path_segments)]
        if float(np.sum(durations)) > max_leg_time:
            return np.full(4, 1e6), dict(initial_state), []
        turn_banks = candidate[len(path_segments):-1]
        target_pitch = float(candidate[-1])
        state = dict(initial_state)
        generated = []
        elapsed = 0.0
        turn_index = 0
        for index, ((kind, _), duration) in enumerate(zip(path_segments, durations)):
            duration = math.ceil(float(duration) / DT - 1e-10) * DT
            target_bank = 0.0
            if kind != "S":
                target_bank = (1.0 if kind == "R" else -1.0) * float(turn_banks[turn_index])
                turn_index += 1
            segment = {
                "startTimeInSeconds": start_time + elapsed,
                "targetBankInDegrees": target_bank,
                "rollRateInDegreesPerSecond": roll_rate,
                "targetPitchInDegrees": 0.0 if index == len(path_segments) - 1 else target_pitch,
                "pitchRateInDegreesPerSecond": pitch_rate,
                "targetSpeedInMetersPerSecond": target_speed,
                "accelerationInMetersPerSecondSquared": acceleration,
            }
            local_segment = dict(segment, startTimeInSeconds=0.0)
            result = simulate([local_segment], {**state, "end_time": duration}, include_state=True)
            state = {
                "north": result[1][-1], "east": result[2][-1], "down": result[3][-1],
                "roll": result[5][-1], "pitch": result[6][-1],
                "heading": result[7][-1] % 360.0, "speed": result[8][-1],
            }
            generated.append(segment)
            elapsed += duration

        heading_error = (state["heading"] - target_heading + 180.0) % 360.0 - 180.0
        residual = np.array([
            state["north"] - float(waypoint["north"]),
            state["east"] - float(waypoint["east"]),
            state["down"] - target_down,
            math.radians(heading_error) * radius,
        ])
        return residual, state, generated

    damping = 1e-2
    residual, final_state, generated = evaluate(parameters)
    best_cost = float(residual @ residual)
    for _ in range(24):
        horizontal_error = math.hypot(residual[0], residual[1])
        heading_error = abs(math.degrees(residual[3] / radius))
        if horizontal_error <= 10.0 and abs(residual[2]) <= 10.0 and heading_error <= 2.0:
            break

        jacobian = np.zeros((4, len(parameters)))
        for column in range(len(parameters)):
            perturbation = 0.03 * scales[column]
            trial = parameters.copy()
            trial[column] = min(upper[column], trial[column] + perturbation)
            if trial[column] == parameters[column]:
                trial[column] = max(lower[column], parameters[column] - perturbation)
            actual_step = trial[column] - parameters[column]
            if actual_step:
                jacobian[:, column] = (evaluate(trial)[0] - residual) / actual_step
        # Solve in scaled parameter units so seconds and degrees have similar influence.
        scaled_jacobian = jacobian * scales
        normal = scaled_jacobian.T @ scaled_jacobian + damping * np.eye(len(parameters))
        try:
            delta = np.linalg.solve(normal, -scaled_jacobian.T @ residual)
        except np.linalg.LinAlgError:
            break
        trial = np.clip(parameters + delta * scales, lower, upper)
        trial_residual, trial_state, trial_segments = evaluate(trial)
        trial_cost = float(trial_residual @ trial_residual)
        if trial_cost < best_cost:
            parameters = trial
            residual, final_state, generated = trial_residual, trial_state, trial_segments
            best_cost = trial_cost
            damping = max(damping / 3.0, 1e-6)
        else:
            damping = min(damping * 10.0, 1e6)

    horizontal_error = math.hypot(residual[0], residual[1])
    heading_error = abs(math.degrees(residual[3] / radius))
    if horizontal_error > 10.0 or abs(residual[2]) > 10.0 or heading_error > 2.0:
        raise ValueError(
            f"Dubins leg could not meet simulated arrival tolerances: "
            f"{horizontal_error:.1f} m horizontal, {abs(residual[2]):.1f} m vertical, "
            f"{heading_error:.1f}° azimuth error.")
    if abs(final_state["speed"] - target_speed) > 0.1:
        raise ValueError("Dubins leg is too short to reach the requested speed.")
    duration = sum(math.ceil(float(value) / DT - 1e-10) * DT
                   for value in parameters[:len(path_segments)])
    return generated, final_state, duration


def generate_waypoint_segments(waypoints, initial_state, leg_limits, control_interval=1.0,
                              max_leg_time=600.0):
    """Generate ordinary FlightSegments records for ordered local-NED waypoints.

    waypoints contain north/east/down meters, azimuth degrees, and speed m/s.
    Legacy altitude (positive up) is accepted when down is omitted. Each item in
    leg_limits constrains the incoming leg of its corresponding waypoint; GEO
    positions must be projected by the GUI before calling this function.

    For each leg, select the shortest six-family geometric Dubins seed and fit
    its commands against simulate. The accepted endpoint state is carried into
    the next leg. A final zero-bank, zero-pitch instruction marks the arrival
    time; it does not stop the aircraft if execution continues past that time.

    Raises ValueError for invalid settings or a leg that cannot satisfy the
    fitter's tolerances. The resulting schedule has no live telemetry feedback
    and cannot guarantee real-aircraft arrival accuracy.
    """
    if not math.isfinite(control_interval) or control_interval <= 0:
        raise ValueError("Control interval must be finite and positive.")
    if not math.isfinite(max_leg_time) or max_leg_time <= 0:
        raise ValueError("Maximum leg time must be finite and positive.")
    if not waypoints:
        raise ValueError("Add at least one waypoint.")
    if len(leg_limits) != len(waypoints):
        raise ValueError("Each waypoint must have flight limits for its incoming leg.")

    state = {key: initial_state[key] for key in
             ("north", "east", "down", "roll", "pitch", "heading", "speed")}
    current_time = 0.0
    generated = []

    for waypoint_index, (waypoint, limits) in enumerate(zip(waypoints, leg_limits), start=1):
        max_bank = abs(float(limits["maxBankAngle"]))
        roll_rate = float(limits["maxRollRate"])
        max_pitch_up = abs(float(limits["maxPitchUpAngle"]))
        max_pitch_down = abs(float(limits["maxPitchDownAngle"]))
        pitch_rate = float(limits["maxPitchRate"])
        acceleration = abs(float(limits["maxAccel"]))
        target_speed = float(waypoint["speed"])
        target_heading = float(waypoint["azimuth"]) % 360.0
        if target_speed <= 0 or roll_rate <= 0 or pitch_rate <= 0 or acceleration <= 0:
            raise ValueError(f"Waypoint {waypoint_index}: speed and rates/acceleration must be positive.")
        if max_bank >= 89 or max_pitch_up >= 89 or max_pitch_down >= 89:
            raise ValueError(f"Waypoint {waypoint_index}: maximum bank and pitch angles must be below 89°.")
        if max_bank <= 0:
            raise ValueError(f"Waypoint {waypoint_index}: maximum bank angle must be > 0° for Dubins planning.")

        planning_speed = max(state["speed"], target_speed, 1.0)
        radius = planning_speed ** 2 / (GRAVITY * math.tan(math.radians(max_bank)))
        start = (state["east"], state["north"], math.radians(90.0 - state["heading"]))
        goal = (float(waypoint["east"]), float(waypoint["north"]),
                math.radians(90.0 - target_heading))
        path_segments = _dubins_path_segments(start, goal, radius)
        try:
            leg_segments, state, leg_duration = _fit_dubins_leg(
                path_segments, state, waypoint, limits, current_time, control_interval,
                max_leg_time, radius)
        except ValueError as error:
            raise ValueError(f"Waypoint {waypoint_index}: {error}") from error
        generated.extend(leg_segments)
        current_time += leg_duration

    generated.append({
        "startTimeInSeconds": current_time,
        "targetBankInDegrees": 0.0,
        "rollRateInDegreesPerSecond": roll_rate,
        "targetPitchInDegrees": 0.0,
        "pitchRateInDegreesPerSecond": pitch_rate,
        "targetSpeedInMetersPerSecond": target_speed,
        "accelerationInMetersPerSecondSquared": acceleration,
    })
    return generated


def expand_target_segment(existing_segments, segment, initial_state, target_azimuth=None, target_altitude=None):
    """Expand optional azimuth and altitude targets into simulator-validated timed segments."""
    if target_azimuth is None and target_altitude is None:
        return [dict(segment)]

    start_time = segment["startTimeInSeconds"]
    if any(item["startTimeInSeconds"] >= start_time for item in existing_segments):
        raise ValueError("Targeted segments must be added after all existing segments.")
    if target_azimuth is not None and segment["rollRateInDegreesPerSecond"] <= 0:
        raise ValueError("Turn-to-azimuth requires a positive roll rate.")
    if target_altitude is not None and segment["pitchRateInDegreesPerSecond"] <= 0:
        raise ValueError("Level-at-altitude requires a positive pitch rate.")
    if target_azimuth is not None and abs(segment["targetBankInDegrees"]) >= 89:
        raise ValueError("Turn-to-azimuth requires a bank angle between -89° and 89°.")

    prefix = sorted([*existing_segments, segment], key=lambda item: item["startTimeInSeconds"])
    baseline_state = {**initial_state, "end_time": max(start_time + DT, DT)}
    baseline = simulate(prefix, baseline_state, include_state=True)
    sample_index = max((i for i, t in enumerate(baseline[0]) if t <= start_time + 1e-9), default=0)
    state_at_start = {
        "north": baseline[1][sample_index],
        "east": baseline[2][sample_index],
        "down": baseline[3][sample_index],
        "roll": baseline[5][sample_index],
        "pitch": baseline[6][sample_index],
        "heading": baseline[7][sample_index],
        "speed": baseline[8][sample_index],
    }
    heading_at_start = state_at_start["heading"]
    altitude_at_start = -state_at_start["down"]
    bank = segment["targetBankInDegrees"]
    pitch = segment["targetPitchInDegrees"]
    roll_rate = segment["rollRateInDegreesPerSecond"]
    pitch_rate = segment["pitchRateInDegreesPerSecond"]

    heading_direction = 1 if bank > 0 else -1
    if target_azimuth is not None and bank == 0:
        raise ValueError("Set a non-zero bank angle to choose the turn direction.")
    if target_azimuth is not None:
        target_azimuth %= 360.0
        if heading_direction > 0:
            target_heading_change = (target_azimuth - heading_at_start) % 360.0
        else:
            target_heading_change = -((heading_at_start - target_azimuth) % 360.0)
        heading_tolerance = max(
            0.1,
            abs(math.degrees(GRAVITY * math.tan(math.radians(bank))
                             / max(state_at_start["speed"], 1e-3) * DT)) * 2,
        )
    else:
        target_heading_change = None
        heading_tolerance = 0.0

    if target_altitude is not None:
        target_altitude_change = target_altitude - altitude_at_start
        altitude_direction = 1 if target_altitude_change >= 0 else -1
        altitude_tolerance = max(
            0.05,
            abs(max(state_at_start["speed"], segment["targetSpeedInMetersPerSecond"])
                * math.sin(math.radians(pitch)) * DT) * 2,
        )
    else:
        target_altitude_change = None
        altitude_direction = 0
        altitude_tolerance = 0.0

    cache = {}

    def evaluate(bank_delay, pitch_delay):
        """Simulate a candidate pair of level-off times and cache its final heading and altitude."""
        key = (None if bank_delay is None else round(bank_delay, 2),
               None if pitch_delay is None else round(pitch_delay, 2))
        if key in cache:
            return cache[key]

        events = {}
        if bank_delay is not None:
            events.setdefault(key[0], {})["targetBankInDegrees"] = 0.0
        if pitch_delay is not None:
            events.setdefault(key[1], {})["targetPitchInDegrees"] = 0.0

        local_segment = dict(segment, startTimeInSeconds=0.0)
        schedule = [local_segment]
        command = local_segment
        for delay in sorted(events):
            command = dict(command, startTimeInSeconds=delay, **events[delay])
            schedule.append(command)

        roll_settle = abs(bank) / roll_rate if bank_delay is not None else 0.0
        pitch_settle = abs(pitch) / pitch_rate if pitch_delay is not None else 0.0
        roll_entry = abs(bank - state_at_start["roll"]) / roll_rate if bank_delay is not None else 0.0
        pitch_entry = abs(pitch - state_at_start["pitch"]) / pitch_rate if pitch_delay is not None else 0.0
        horizon = max(
            (bank_delay or 0.0) + roll_settle,
            (pitch_delay or 0.0) + pitch_settle,
            roll_entry,
            pitch_entry,
            0.5,
        ) + DT
        local_state = {**state_at_start, "end_time": horizon}
        result = simulate(schedule, local_state, include_state=True)
        cache[key] = result[7][-1], -result[3][-1]
        return cache[key]

    def solve_delay(progress, target_change, tolerance, label):
        """Find a level-off delay by bracketing and bisecting the simulated target progress."""
        if abs(target_change) <= tolerance:
            return DT
        low = DT
        low_progress = progress(low)
        if low_progress > target_change + tolerance:
            raise ValueError(f"{label} target is too close to level off at the selected rate.")
        high = 1.0
        high_progress = progress(high)
        while high_progress < target_change - tolerance and high < 600.0:
            high = min(high * 2, 600.0)
            high_progress = progress(high)
        if high_progress < target_change - tolerance:
            raise ValueError(f"The selected maneuver cannot reach the requested {label} within 600 seconds.")
        while high - low > DT:
            middle = (low + high) / 2
            if progress(middle) < target_change:
                low = middle
            else:
                high = middle
        candidates = (low, high)
        return min(candidates, key=lambda delay: abs(progress(delay) - target_change))

    bank_delay = DT if target_azimuth is not None else None
    pitch_delay = DT if target_altitude is not None else None
    heading_error = altitude_error = 0.0
    for _ in range(6):
        if target_azimuth is not None:
            bank_delay = solve_delay(
                lambda delay: heading_direction * (evaluate(delay, pitch_delay)[0] - heading_at_start),
                abs(target_heading_change), heading_tolerance, "azimuth")
        if target_altitude is not None:
            pitch_delay = solve_delay(
                lambda delay: altitude_direction * (evaluate(bank_delay, delay)[1] - altitude_at_start),
                abs(target_altitude_change), altitude_tolerance, "altitude")
        final_heading, final_altitude = evaluate(bank_delay, pitch_delay)
        if target_azimuth is not None:
            heading_error = abs(heading_direction * (final_heading - heading_at_start)
                                - abs(target_heading_change))
        if target_altitude is not None:
            altitude_error = abs(final_altitude - target_altitude)
        if heading_error <= heading_tolerance and altitude_error <= altitude_tolerance:
            break
    else:
        raise ValueError("The azimuth and altitude targets cannot both be met with this segment.")

    generated = [dict(segment)]
    events = {}
    if bank_delay is not None:
        events.setdefault(round(start_time + bank_delay, 2), {})["targetBankInDegrees"] = 0.0
    if pitch_delay is not None:
        events.setdefault(round(start_time + pitch_delay, 2), {})["targetPitchInDegrees"] = 0.0
    command = dict(segment)
    for event_time in sorted(events):
        command = dict(command, startTimeInSeconds=event_time, **events[event_time])
        generated.append(command)
    return generated


def infer_target_groups(segments, initial_state):
    """Infer lock-generated segment groups from level-off transitions in a standard route."""
    if len(segments) < 2:
        return {}
    state = {**initial_state, "end_time": segments[-1]["startTimeInSeconds"] + DT}
    result = simulate(segments, state, include_state=True)
    times = result[0]
    groups = {}
    index = 0

    while index < len(segments) - 1:
        first = segments[index]
        previous = first
        members = [first]
        targets = {}
        next_index = index + 1

        while next_index < len(segments):
            current = segments[next_index]
            unchanged_controls = all(
                current[key] == previous[key]
                for key in previous
                if key not in ("startTimeInSeconds", "targetBankInDegrees", "targetPitchInDegrees")
            )
            bank_unchanged = current["targetBankInDegrees"] == previous["targetBankInDegrees"]
            pitch_unchanged = current["targetPitchInDegrees"] == previous["targetPitchInDegrees"]
            bank_lock = previous["targetBankInDegrees"] != 0 and current["targetBankInDegrees"] == 0
            pitch_lock = previous["targetPitchInDegrees"] != 0 and current["targetPitchInDegrees"] == 0
            if not unchanged_controls or not (bank_unchanged or bank_lock) \
                    or not (pitch_unchanged or pitch_lock) or not (bank_lock or pitch_lock):
                break

            sample = min(range(len(times)), key=lambda i: abs(times[i] - current["startTimeInSeconds"]))
            if bank_lock:
                targets["target_azimuth"] = result[7][sample] % 360.0
            if pitch_lock:
                targets["target_altitude"] = -result[3][sample]
            members.append(current)
            previous = current
            next_index += 1

        if targets:
            groups[id(first)] = {"segments": members, "targets": targets}
            index = next_index
        else:
            index += 1
    return groups


def plot(path, segments, state, result):
    """Render the simulated route, save a PNG beside the input file, and show the plot."""
    times, norths, easts, downs, seg_ids = result
    fig, ax = plt.subplots(figsize=(11, 9))
    draw_route(ax, path.stem, segments, state, result)
    fig.tight_layout()

    out = path.with_name(f"{path.stem}_top_down.png")
    fig.savefig(out, dpi=150)
    print(f"\nSaved plot to {out}")
    print(f"End position NED: ({norths[-1]:.1f}, {easts[-1]:.1f}, {downs[-1]:.1f}) m")
    plt.show()


def draw_route(ax, name, segments, state, result, system_location=None):
    """Draw segment paths, transition markers, endpoints, labels, and axes on a Matplotlib axis."""
    times, norths, easts, downs, seg_ids = result
    reference = state.get("geo_reference")
    if reference is not None:
        positions = [route_plot_position(north, east, reference) for north, east in zip(norths, easts)]
        easts, norths = [position[0] for position in positions], [position[1] for position in positions]
    colors = plt.cm.tab10.colors

    for i, seg in enumerate(segments):
        idx = [k for k, s in enumerate(seg_ids) if s == i]
        if not idx:
            continue
        if idx[0] > 0:
            idx.insert(0, idx[0] - 1)
        end = segments[i + 1]["startTimeInSeconds"] if i + 1 < len(segments) else state["end_time"]
        label = (f"{seg['startTimeInSeconds']:g}-{end:g} s: bank {seg['targetBankInDegrees']:g}°, "
                 f"pitch {seg['targetPitchInDegrees']:g}°, speed {seg['targetSpeedInMetersPerSecond']:g} m/s")
        ax.plot([easts[k] for k in idx], [norths[k] for k in idx], color=colors[i % 10], lw=2.5, label=label)

    ax.plot(easts[0], norths[0], "ko", ms=8)
    ax.annotate("START", (easts[0], norths[0]), xytext=(8, -14), textcoords="offset points", weight="bold")
    for seg in segments[1:]:
        k = min(range(len(times)), key=lambda j: abs(times[j] - seg["startTimeInSeconds"]))
        if seg["startTimeInSeconds"] <= state["end_time"]:
            ax.plot(easts[k], norths[k], "o", mfc="white", mec="k", ms=7)
            ax.annotate(f"{seg['startTimeInSeconds']:g} s", (easts[k], norths[k]),
                        xytext=(8, 6), textcoords="offset points")
    ax.plot(easts[-1], norths[-1], "ks", ms=8)
    if system_location is not None:
        system_north, system_east = system_location
        system_east, system_north = route_plot_position(system_north, system_east, reference)
        ax.plot(system_east, system_north, marker="X", color="#d62728", markersize=7,
            linestyle="None", label="Battery")
    # Put the END label on the side facing the middle of the route so it isn't clipped.
    on_right = easts[-1] > (min(easts) + max(easts)) / 2
    on_top = norths[-1] > (min(norths) + max(norths)) / 2
    ax.annotate(f"END ({times[-1]:.1f} s)\nD {downs[-1]:.0f} m (alt {-downs[-1]:.0f} m)", (easts[-1], norths[-1]),
                xytext=(-10 if on_right else 10, -10 if on_top else 10), textcoords="offset points", weight="bold",
                ha="right" if on_right else "left", va="top" if on_top else "bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))

    # Extra margin so START/END labels stay inside the plot area.
    ax.margins(0.15)
    configure_route_axes(ax, reference)
    ax.grid(True, alpha=0.4)
    origin_label = f"NED=({state['north']:g}, {state['east']:g}, {state['down']:g}) m" \
        if reference is None else f"Lat/Lon=({reference[0]:.6f}, {reference[1]:.6f}) deg, Down={state['down']:g} m"
    ax.set_title(f"{name} – top-down route\n"
                 f"Initial: {origin_label}, "
                 f"roll {state['roll']:g}°, pitch {state['pitch']:g}°, "
                 f"heading {state['heading']:g}°, speed {state['speed']:g} m/s", fontsize=11)
    ax.legend(loc="best", fontsize=9)


def draw_waypoints(ax, waypoints, geo_reference=None):
    """Overlay labeled NED waypoint locations on the top-down route plot."""
    if not waypoints:
        return
    for index, waypoint in enumerate(waypoints, start=1):
        east, north = route_plot_position(float(waypoint["north"]), float(waypoint["east"]), geo_reference)
        ax.plot(east, north, marker="*", markersize=12, color="#e6a700",
                markeredgecolor="#222", linestyle="None", zorder=6,
                label="Waypoints" if index == 1 else None)
        ax.annotate(f"WP{index}", (east, north), xytext=(8, 8),
                    textcoords="offset points", weight="bold", zorder=7)
    ax.legend(loc="best", fontsize=9)


def draw_altitude_profile(ax, segments, state, result):
    """Plot altitude relative to the origin against time, colored by active segment."""
    times, _, _, downs, seg_ids = result
    colors = plt.cm.tab10.colors

    for i, seg in enumerate(segments):
        indices = [k for k, segment_id in enumerate(seg_ids) if segment_id == i]
        if not indices:
            continue
        if indices[0] > 0:
            indices.insert(0, indices[0] - 1)
        end_time = segments[i + 1]["startTimeInSeconds"] if i + 1 < len(segments) else state["end_time"]
        ax.plot([times[k] for k in indices], [-downs[k] for k in indices],
                color=colors[i % len(colors)], lw=2.2, label=f"{seg['startTimeInSeconds']:g}-{end_time:g} s")

    for seg in segments[1:]:
        if seg["startTimeInSeconds"] <= state["end_time"]:
            ax.axvline(seg["startTimeInSeconds"], color="#777", linestyle="--", linewidth=0.8, alpha=0.6)

    ax.axhline(-downs[0], color="#777", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.plot(times[0], -downs[0], "ko", ms=5)
    ax.plot(times[-1], -downs[-1], "ks", ms=5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Altitude relative to origin [m]")
    ax.set_title("Altitude profile")
    ax.grid(True, alpha=0.4)
    ax.legend(loc="best", fontsize=8)


def main():
    """Run the command-line plotting workflow from path selection through display."""
    path = ask_json_path()
    segments = load_segments(path)
    print(f"Loaded {len(segments)} flight segment(s) from {path}")
    state = ask_initial_state(segments)
    plot(path, segments, state, simulate(segments, state))


if __name__ == "__main__":
    main()
