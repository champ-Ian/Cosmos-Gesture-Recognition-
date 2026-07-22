# Cosmos-Gesture-Recognition-

Cumulative IoT and Data Science final project at COSMOS UCLA. Recognize gestures
from the course's [gesture list](#gesture-list) using a fused multi-sensor IoT
pipeline: mmWave radar, IMU, UWB ranging, and RFID.

This repo currently contains the **data collection framework**. Training and
real-time evaluation scripts (see the project requirements: single-sensor
baseline vs. fused model, confusion matrix, held-out-person evaluation) are the
next step once a real dataset exists.

## Sensing plan

| Sensor | Role |
| --- | --- |
| mmWave radar (TI xWRL6432/IWR6432, from `mmwave_lab`) | Range profile + point cloud of arm/hand motion. |
| IMU (ESP32 Core2) | Wrist/hand-worn accelerometer + gyroscope. |
| UWB (2x Qorvo DWM3001CDK, from `UWB_lab`) | FiRa two-way-ranging distance between a controller and controlee board (e.g. one per wrist). |
| RFID reader + tags | Near-field hand/finger sensing (Soli-style micro-gestures, fist open/close). |

Not every gesture needs every sensor — see `gestures.py` for the
suggested sensor combination per gesture (also printed as a prompt during
collection). Per the project requirements, you should collect at least one
**single-sensor baseline** (e.g. mmWave-only) and compare it against a
**fused model** trained on multiple sensors.

## Gesture list

`pull`, `push`, `clockwise`, `anti_clockwise`, `right`, `left`, `bye_bye`,
`one_arm_boxing`, `clapping`, `two_arm_boxing`, `t_arm`, `raise_arms`, `soli`,
`fist_open` (Making Fist and Open), `palm_up_down`.

These are the canonical (snake_case) names used everywhere in code — see
`gestures.py` for the full registry (display name, spoken instruction,
suggested sensors). Start with a smaller subset (`--gesture pull,push,left,right`)
before collecting all 15.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate       # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### mmWave radar wiring

Same hardware/wiring as `mmwave_lab` (TI xWRL6432 EVM over a USB-to-UART
adapter) — see that repo's `README.md` Section 0 for the pinout and
`swru628.pdf` for the board reference. Find the serial port the same way
(`ls /dev/cu.usbserial*` / `python -m serial.tools.list_ports -v` on macOS,
Device Manager on Windows).

The default radar config (`mmwave/xwrL64xx-evm/near_field_hand_50cm.cfg`,
copied from `mmwave_lab`) streams both a range profile and a point cloud in
the first 50 cm — good for hand/arm gestures directly in front of the board.
Use `--mmwave-cfg` to point at a different `.cfg` (e.g. `point_cloud.cfg` for
a wider field of view) if a gesture needs more range.

### IMU / RFID firmware contract

This repo does not include ESP32 firmware — that is built separately per
group. The collector treats these two boards generically: it just reads
**newline-terminated text lines** from each serial port and timestamps them
on arrival (see `sensors/serial_json_stream.py`). Nothing is parsed or
dropped at collection time, so firmware can keep changing without
re-collecting data.

Recommended (not required) firmware output — one JSON object per line:

```text
IMU:  {"t_ms": 12345, "ax": 0.01, "ay": 0.98, "az": 0.03, "gx": 1.2, "gy": -0.3, "gz": 0.1}
RFID: {"t_ms": 12345, "tag_id": "E200...", "rssi": -41.5}
```

If your firmware prints CSV or something else, that's fine — the raw line is
stored either way. Feature extraction (parsing these lines into numeric
arrays for training) is a separate step once your team locks in the actual
line format; it only needs to change a parser, not recollect data.

### UWB wiring (Qorvo DWM3001CDK FiRa ranging pair)

Unlike IMU/RFID, the UWB kit is **not** a single JSON-line tag: it's the same
two-board Qorvo DWM3001CDK setup as `UWB_lab`, running FiRa two-way ranging
(TWR) between a controller/initiator and a controlee/responder board. Each
board needs its own serial port (`ls /dev/cu.usbmodem*` on macOS), and both
boards must use the same class-sheet-assigned preamble code and channel.

`uwb/uwb_stream.py` drives this the same way `UWB_lab/ranging_experiment_wrapper.py`
does: it launches the vendored `uwb/uwb-qorvo-tools/scripts/fira/run_fira_twr/run_fira_twr.py`
as a subprocess per board (resetting both devices first via UCI), then parses
the controller's `distance: X cm` / `status: Ok (0x0)` stdout lines with a
regex-based log parser. Only the controller reports distance; the controlee
just needs to be running so the controller has something to range against.

Required flags when UWB is enabled:

```text
--uwb-controller-port /dev/cu.usbmodemXXXX
--uwb-controlee-port  /dev/cu.usbmodemYYYY
--uwb-group-id <class-sheet group number>
--uwb-preamble-code <9|10|11|12, from the class sheet>
--uwb-channel <5|9, from the class sheet>
```

`--uwb-fps` (default 50), `--uwb-slot-span` (default 2400), and
`--uwb-slots-per-rr` (default 6) control the ranging timing, same as in
`UWB_lab`. Per-trial logs and device-reset logs are written under
`datasets/<dataset_name>/uwb_logs/`.

## Collecting data

Only pass `--*-port` for sensors you actually have wired up right now — any
sensor without a port is skipped, which is how you'd collect a single-sensor
baseline.

Full four-sensor collection:

```bash
python collect_gesture_dataset.py \
  --collector student01 \
  --mmwave-port /dev/cu.usbserial-AAAA \
  --imu-port /dev/cu.usbserial-BBBB \
  --uwb-controller-port /dev/cu.usbmodemCCCC \
  --uwb-controlee-port /dev/cu.usbmodemDDDD \
  --uwb-group-id 1 --uwb-preamble-code 9 --uwb-channel 5 \
  --rfid-port /dev/cu.usbserial-EEEE \
  --gesture pull,push,clockwise,anti_clockwise \
  --trials 5 \
  --duration 4
```

Single-sensor baseline (mmWave only):

```bash
python collect_gesture_dataset.py \
  --collector student01 \
  --mmwave-port /dev/cu.usbserial-AAAA \
  --trials 5 --duration 4
```

Short smoke test (1 trial, 2 gestures, short window) before a full session:

```bash
python collect_gesture_dataset.py \
  --collector student01 --mmwave-port /dev/cu.usbserial-AAAA \
  --gesture pull,push --trials 1 --duration 2
```

For each trial the script prompts with the gesture name, instructions, and
suggested sensors, waits for Enter, records for `--duration` seconds while
printing live per-sensor sample counts, then asks whether to keep the
recording (`--auto-accept` skips that prompt for unattended runs).

Output layout:

```text
datasets/gesture_dataset_YYYYMMDD_HHMMSS/
  dataset_metadata.json
  trials.csv
  uwb_logs/                 # only if UWB was enabled: device-reset + controller/controlee logs
  sessions/
    gesture_student01_pull_001/
      trial_data.npz
      trial_metadata.json
    ...
```

Each `trial_data.npz` holds, per enabled sensor:
- mmWave: `mmwave_frame_number`, `mmwave_time_s`, `mmwave_range_profile`,
  `mmwave_point_count`, `mmwave_points_xyz`, `mmwave_points_velocity`.
- IMU/RFID: `{sensor}_recv_time_s` and `{sensor}_raw_lines` (raw text, one
  entry per line received during the trial window).
- UWB: `uwb_time_s`, `uwb_sequence`, `uwb_status`, `uwb_distance_cm` (one
  entry per parsed ranging sample from the controller).

### Combining datasets from multiple group members

Per the project requirements ("data from multiple group members when
possible" and evaluating a held-out person), combine each collector's dataset
before training:

```bash
python combine_gesture_datasets.py \
  datasets/gesture_dataset_student01_... \
  datasets/gesture_dataset_student02_... \
  --output datasets/combined_gesture_dataset
```

This concatenates manifests into one `trials.csv` (rewriting session/npz
paths to absolute paths) and reports per-gesture, per-collector, and
per-sensor-combination counts in the combined `dataset_metadata.json`.

## Troubleshooting

- If the radar won't configure after a previous run, power-cycle the EVM and
  rerun — same as `mmwave_lab`.
- If a JSON/IMU/RFID port produces zero lines, double check the baud rate
  (`--imu-baud` / `--rfid-baud`, default 115200) and that the firmware is
  actually printing to USB serial (not just a debug UART).
- If UWB produces zero Ok samples: confirm both boards are on the
  class-sheet-assigned preamble code/channel, that no other terminal/process
  already has either port open, and check
  `datasets/<dataset_name>/uwb_logs/controller/controller_terminal_log.txt`
  for the raw `run_fira_twr.py` output. Use `--uwb-skip-device-reset` only if
  the boards are already known-good — a bad reset is a common cause of a
  silent controller.
- `Ctrl+C` stops the collector cleanly; it sends `sensorStop 0` to the radar,
  resets the UWB boards, and closes all serial ports/subprocesses before
  exiting.
- Use `--min-mmwave-frames` / `--min-sensor-lines` to tune how aggressively
  short/glitchy trials get auto-discarded. `--min-uwb-samples` does the same
  for UWB (based on Ok range samples rather than raw lines).

## Files

- `gestures.py`: canonical gesture registry (names, instructions, suggested sensors).
- `collect_gesture_dataset.py`: main multi-sensor trial-based collector.
- `combine_gesture_datasets.py`: merge datasets from multiple collectors.
- `sensors/serial_json_stream.py`: generic background-thread raw-line reader for IMU/RFID.
- `sensors/common.py`: shared timestamp/manifest/JSON helpers.
- `mmwave/radar_io.py`: TI xWRL6432 UART protocol (adapted from `mmwave_lab`).
- `mmwave/mmwave_stream.py`: background-thread radar frame reader with time-windowed extraction.
- `mmwave/xwrL64xx-evm/*.cfg`: radar configs copied from `mmwave_lab`.
- `uwb/uwb_io.py`: Qorvo FiRa TWR subprocess helpers + ranging log parser (adapted from `UWB_lab/uwb_lab_common.py`).
- `uwb/uwb_stream.py`: background-thread controller+controlee ranging pair reader with time-windowed extraction.
- `uwb/uwb-qorvo-tools/`: vendored Qorvo UCI/FiRa CLI (`run_fira_twr.py`, `reset_device.py`, `uci`/`uqt_utils` libs), copied from `UWB_lab`.

Raw recordings and trained models are ignored by git under `datasets/`,
`sessions/`, and `models/`.
