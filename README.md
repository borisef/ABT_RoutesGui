# ABT Flight Route Builder — User Guide

## Overview

ABT Flight Route Builder is a GUI tool for creating, editing, simulating, and validating aircraft flight instruction files in JSON format. Routes can be defined either via direct flight instructions or via waypoints, and the results are displayed as a top-down map view and an altitude profile.

> **Important:** The simulation and constraint checks are model-based. They do not guarantee the performance of a real aircraft and are not a substitute for proper testing and authorization before operational use.

---

## Requirements

- Python 3.9 or later
- Tkinter (included in the standard Python installer on Windows)
- Dependencies listed in `requirements.txt`:
  - `matplotlib >= 3.5`
  - `numpy >= 1.21`
  - `geographiclib >= 2.0`

---

## Installation & Launch

**Without Python (Windows):** Run `FlightRouteGui.exe` directly from the tool folder.

**With Python:**

```powershell
python -m pip install -r requirements.txt
python FlightRouteGui.py
```

---

## Project Structure

```
ABT_RoutesGui/
├── FlightRouteGui.py                      # Main GUI application
├── PlotFlightRoute.py                     # Simulation and plotting engine
├── requirements.txt
├── README.md                              # This file (English)
├── README_heb.md                          # Original documentation (Hebrew)
├── ABT_FlightCharacteristics/
│   └── ABT0.json                          # Aircraft limits (roll, pitch, speed, etc.)
└── ABT_FlightInstructions/
    └── TEST1.json                         # Example flight instructions
```

---

## Interface Layout

The left panel contains route settings, the instruction/waypoint table, the initial state, the flight limits file selector, and status messages. The right panel displays the top-down route view and the altitude-over-time profile.

---

## Coordinate Systems

Select the coordinate type with the **Coordinate type** selector:

| Mode | Description |
|------|-------------|
| **BAT** | Local coordinates in metres: North, East, Down |
| **GEO** | WGS84 latitude/longitude in decimal degrees for the start position, waypoints, and battery position. The tool converts them to a local NED frame for simulation. |

**Conventions (both modes):**
- `Down` is in metres, positive downward. A negative value means above the reference point.
- Heading `0°` is North; angles increase clockwise.
- Positive bank = right turn. Positive pitch = nose up (climb).

---

## Initial State & Battery Position

In the **Initial state** section, enter the aircraft state at the start of the simulation:

| Field | Description |
|-------|-------------|
| North / East / Down | Starting position (or Lat/Lon in GEO mode) |
| Roll | Initial bank angle (°) |
| Pitch | Initial pitch angle (°) |
| Heading | Initial heading (°) |
| Speed | Initial speed (m/s) |
| End time [s] | Simulation duration. Leave blank to use the default. |

In the **Battery position** section you can optionally enter a North/East position (or Lat/Lon in GEO mode). Both fields must be filled or both left empty. The position is shown on the plot but does not affect the simulated flight path.

---

## Route Mode 1 — Direct Instructions

Select the **Direct instructions** tab. Each row in the table represents an instruction that takes effect at a given time and remains active until the next instruction.

### Instruction Fields

| Field | Description |
|-------|-------------|
| `Start [s]` | Start time in seconds (≥ 0, must be unique). |
| `Bank [°]` | Target bank angle; positive = right turn, negative = left. |
| `Roll rate [°/s]` | Rate of bank change; must be ≥ 0. |
| `Pitch [°]` | Target pitch angle; positive = climb. |
| `Pitch rate [°/s]` | Rate of pitch change; must be ≥ 0. |
| `Speed [m/s]` | Target speed; must be > 0. |
| `Accel [m/s²]` | Acceleration toward target speed; must be ≥ 0. |

### Table Operations

- **Add** — add a new instruction.
- **Update selected** — select a row, edit the fields, then click to apply.
- **Delete selected** — remove the selected instruction.
- **Clear all** — remove all instructions.

The table is always sorted by start time.

### Optional Auto-Target Fields

Leave these empty if not needed:

- **Turn to azimuth [°]** — the tool inserts additional instructions to complete the turn at the specified heading. A non-zero bank angle must be set to determine the turn direction.
- **Level at altitude [m] (neg = up)** — the tool inserts an instruction to level off at the given altitude (using the Down sign convention: negative = above reference). The target must be reachable with the specified pitch angle and rate.

Both auto-targets can be combined. An instruction with an auto-target must be placed after all existing instructions.

When updating an existing instruction, you can check **Shift following segments by** and enter a time offset (seconds) to push later instructions forward (positive) or backward (negative).

---

## Route Mode 2 — Waypoints

Select the **Waypoints** tab. For each waypoint, enter:

| Field | Description |
|-------|-------------|
| North / East / Down | Waypoint position in metres (or Lat/Lon in GEO mode) |
| Azimuth | Required heading on arrival (°) |
| Speed | Required speed on arrival (m/s) |
| Incoming-leg limits | Constraints that apply to the leg leading **into** this waypoint |

After filling in the values, click **Add**. Select a waypoint to **Update selected**, **Delete selected**, or reorder with **Move up** / **Move down**. The order of the list defines the flight sequence.

Waypoint positions are shown on the plot even before flight instructions are generated. Click **Plot route** to generate (or regenerate) the instructions and display the route.

---

## Plotting & Constraint Checking

Click **Plot route** to compute and display the route.

- The **top graph** shows the top-down (North/East) view.
- The **bottom graph** shows altitude vs. time.
- The **status area** shows the final position, any constraint violations, and (in Waypoints mode) approach/heading errors.

### Navigation Controls

| Action | How |
|--------|-----|
| Zoom | Scroll the mouse wheel over either graph |
| Pan | Left-click and drag |
| Reset view | Click **Reset view** |

---

## Flight Limits File

The tool automatically loads `DataFiles\ABT_FlightCharacteristics\ABT0.json` if found. Otherwise, click **Browse…** in the **Flight limits** area to select a valid characteristics file.

The file must contain a `Flight_Characteristics` array with at least one entry that includes limits such as maximum roll/pitch angles and rates, speed range, and acceleration range.

Click **None** to disable constraint checking.

Constraint violations appear as warnings; they do not block editing or saving. When no limits file is loaded, the tool indicates that checking is unavailable. Treat warnings seriously and validate values against the actual aircraft specifications.

---

## File Operations

| Action | Description |
|--------|-------------|
| **New** | Start a new empty route |
| **Open…** | Open an existing JSON route file |
| **Save** | Save to the current file (prompts for a path if the route is new) |
| **Save As…** | Save to a different name or location |
| **File → Export plot image…** | Export the graphs as PNG, SVG, or PDF |

Saving requires at least one flight instruction. In Waypoints mode, instructions are regenerated at save time. An asterisk (`*`) in the window title indicates unsaved changes. When opening a file, creating a new route, or closing, the tool prompts to save unsaved changes.

The route file stores: route type, coordinate type, initial state, generated instructions, and (if applicable) waypoints and battery position.

---

## Aircraft Characteristics Example (ABT0.json)

| Parameter | Value |
|-----------|-------|
| Max roll angle | ±30° |
| Max roll rate | 10°/s |
| Max pitch up | 20° at 5°/s |
| Max pitch down | 15° at 5°/s |
| Speed range | 30–55 m/s |
| Acceleration range | −5 to +5 m/s² |
