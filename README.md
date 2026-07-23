# Cosmos-Gesture-Recognition-

Cumulative IoT and Data Science final project at COSMOS UCLA. Recognize gestures
from the course's [gesture list](#gesture-list) using a fused multi-sensor IoT
pipeline: mmWave radar, IMU, UWB ranging, and RFID.

This repo contains the full pipeline: data collection, dataset cutting,
training/evaluation (single-sensor baseline vs. fused model, confusion
matrix, held-out-person evaluation), and live evaluation. The architecture
follows the final-project implementation hints (Shanmu Wang): sensors
stream continuously, a coordinator writes event markers, and cutting into
per-trial windows happens offline -- see
[Collecting data](#collecting-data) below. The feature extractors are
deliberately simple starter/baseline features (mirroring how `UWB_lab` ships
baseline range features and leaves a `_proposal` extractor as a student
TODO) -- expect to replace or extend them once you've looked at your own
recorded data.

## Project layout

Follows the final-project implementation-hints skeleton (Shanmu Wang):
scripts and importable code live under `src/`; runtime data/models/figures
live at the repo root, outside `src/`, and are gitignored.

```text
README.md
requirements.txt
src/
  collect.py               # coordinator: streams all sensors, logs events + continuous per-sensor logs
  extract_features.py      # cuts a raw session into per-trial data, and extracts features from it
  combine_datasets.py      # merge processed datasets from multiple collectors
  train.py                 # train a single-sensor or fused gesture classifier
  evaluate.py               # evaluate an already-trained model against a dataset (no retraining)
  realtime_demo.py          # live sliding-window gesture evaluation using a trained model
  gestures.py               # canonical gesture registry
  gesture_models.py         # shared classifier builder + LateFusionClassifier
  sensors/
    base_reader.py          # shared reader interface (check_error/sample_count/window/close)
    imu_reader.py            # IMU (USB serial)
    uwb_reader.py             # UWB (Qorvo FiRa TWR, subprocess-driven)
    mmwave_reader.py           # mmWave radar (USB serial, binary protocol)
    rfid_reader.py              # RFID (TCP socket, not serial)
    common.py                    # shared timestamp/manifest/JSON helpers + REPO_DIR/output-path constants
  mmwave/radar_io.py         # TI xWRL6432 UART protocol (adapted from mmwave_lab)
  mmwave/xwrL64xx-evm/*.cfg  # radar configs
  uwb/uwb_io.py              # Qorvo FiRa TWR subprocess helpers + ranging log parser
  uwb/uwb-qorvo-tools/       # vendored Qorvo UCI/FiRa CLI
data/
  raw/                     # collect.py output: one folder per session
  processed/               # extract_features.py cut output: per-trial datasets
models/                    # trained .joblib models
results/
  figures/                 # confusion-matrix PNGs
```

There's no `src/sensors/wifi_reader.py` / WiFi CSI support here -- it's in the
general class kit but wasn't part of this project's sensor box.

**Run every script from the repo root** as `python src/<script>.py ...`
(e.g. `python src/collect.py --collector student01 ...`) -- that's what all
the examples below assume. All path defaults (`data/raw`, `data/processed`,
`models/`, `results/figures/`, the mmWave `.cfg`) are resolved relative to
the repo root regardless of your current directory, so this also works if
you `cd src/` first and drop the `src/` prefix instead.

## Sensing plan

| Sensor | Role |
| --- | --- |
| mmWave radar (TI xWRL6432/IWR6432, from `mmwave_lab`) | Range profile + point cloud of arm/hand motion. |
| IMU (ESP32 Core2 + BMI270, from `IMU_lab_students`) | Wrist/hand-worn accelerometer + gyroscope. |
| UWB (3x Qorvo DWM3001CDK, from `UWB_lab`) | FiRa two-way-ranging distance from 1 fixed anchor to 2 worn nodes (one per wrist/arm). |
| RFID reader + tags (from `RFID_Lab`) | Near-field hand/finger sensing (Soli-style micro-gestures, fist open/close). |

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
suggested sensors, and **group**). Start with a smaller subset
(`--gesture pull,push,left,right`) before collecting all 15.

Each gesture also has a `group` (per the final-project implementation
hints):

- **discrete** (`pull`, `push`, `clockwise`, `anti_clockwise`, `right`,
  `left`, `bye_bye`, `t_arm`, `raise_arms`, `fist_open`): one clean instance
  per trial, clear start/end. `collect.py` prompts trial-by-trial.
- **periodic** (`clapping`, `one_arm_boxing`, `two_arm_boxing`,
  `palm_up_down`, `soli`): repeated cycles with no single natural boundary.
  `collect.py` records one long continuous take instead; segmenting that
  into individual cycles happens later, in `extract_features.py cut`.

`fist_open` is graded as discrete here, not periodic -- reasonable people
could call it either way (repeated open/close cycles), and the
implementation-hints slide didn't explicitly list it. Edit `gestures.py`'s
`GestureSpec(..., group=...)` if you'd rather treat it as periodic.

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

The default radar config (`src/mmwave/xwrL64xx-evm/near_field_hand_50cm.cfg`,
copied from `mmwave_lab`) streams both a range profile and a point cloud in
the first 50 cm — good for hand/arm gestures directly in front of the board.
Use `--mmwave-cfg` to point at a different `.cfg` (e.g. `point_cloud.cfg` for
a wider field of view) if a gesture needs more range.

### IMU wiring (ESP32 + BMI270, `IMU_lab_students` firmware)

The IMU board is a USB serial device (`--imu-port`, default baud 115200).
`collect.py` reads **newline-terminated raw text lines** and timestamps
them on arrival (see `src/sensors/imu_reader.py`); nothing is parsed at
collection time (the raw line is kept alongside the parsed columns in the
continuous log), so firmware changes don't force a recollect.

The actual firmware (`IMU_lab_students/main/main.c`) prints one line per
sample, not JSON:

```text
accel[g] x= 0.012 y=-0.034 z= 0.998 | gyro[dps] x= 0.10 y=-0.20 z= 0.05
```

`extract_features.py` parses exactly this format (`parse_imu_line()`/
`_IMU_SAMPLE_RE`), and `collect.py` uses the same parser to fill in the
`ax..gz` columns of the live `imu.csv` log. If your group's firmware prints
something else, update `_IMU_SAMPLE_RE` -- the raw line is always stored
either way, so this only means re-running the cut/feature-extraction step,
not recollecting data.

### RFID wiring (`RFID_Lab` reader, TCP -- not serial)

Unlike every other sensor here, the RFID reader is **not a USB serial
device** -- it's a network device that streams tag reads over a **TCP
socket**, same as `RFID_Lab/reading_from_TCP.py` / `touch_detector_gui.py`.
Connect your laptop to the reader's network (its default address is
`192.168.137.1:9055`) before collecting; `src/sensors/rfid_reader.py` opens
that socket instead of a serial port. Enable it with `--rfid` (override the
address with `--rfid-host`/`--rfid-tcp-port` if needed).

The reader prints one line per tag read, for every tag it sees (not just
ones you care about):

```text
E2806995000040154D38514E 2024-01-01 12:00:00.123 -45 3
```

`<EPC> <timestamp> <RSSI> <read_count>`, space-separated (see
`RFID_Lab/rfid_log_utils.py`). `extract_features.py`'s
`extract_rfid_features()` pools every tag read in the trial window into one
feature vector by default; if your gestures use specific tags (e.g.
Soli-style one-tag-per-finger sensing), filter by EPC first for per-tag
features instead.

### UWB wiring (Qorvo DWM3001CDK FiRa ranging: anchor + node(s))

Like RFID, the UWB kit is not read over generic serial text: it's the same
Qorvo DWM3001CDK boards as `UWB_lab`, running FiRa two-way ranging (TWR)
between one fixed **anchor** (FiRa "controller" role) and one or more worn
**nodes** (FiRa "controlee" role) — this project's kit is 1 anchor + 2 nodes
(e.g. one node per wrist/arm), so each gesture gets two independent
distance-from-anchor signals instead of one. Each board needs its own serial
port (`ls /dev/cu.usbmodem*` on macOS), and every board in the setup must use
the same class-sheet-assigned preamble code and channel.

`src/sensors/uwb_reader.py` drives this the same way
`UWB_lab/ranging_experiment_wrapper.py` does: it launches the vendored
`src/uwb/uwb-qorvo-tools/scripts/fira/run_fira_twr/run_fira_twr.py` as a
subprocess per board (resetting all devices first via UCI), then parses the
anchor's `distance: X cm` / `status: Ok (0x0)` / `mac address: ...` stdout
lines with a regex-based log parser. Only the anchor reports distance;
nodes just need to be running so the anchor has something to range against.

Required flags when UWB is enabled:

```text
--uwb-anchor-port /dev/cu.usbmodemXXXX
--uwb-node-port /dev/cu.usbmodemYYYY   # repeat --uwb-node-port for more nodes
--uwb-group-id <class-sheet group number>
--uwb-preamble-code <9|10|11|12, from the class sheet>
--uwb-channel <5|9, from the class sheet>
```

`--uwb-fps` (default 50), `--uwb-slot-span` (default 2400), and
`--uwb-slots-per-rr` control the ranging timing.

**Single node** (one `--uwb-node-port`): plain FiRa unicast TWR, exactly
`UWB_lab`'s documented controller/controlee lab exercise — `--uwb-slots-per-rr`
defaults to 6, same as that lab.

**Multiple nodes** (two or more `--uwb-node-port` flags, i.e. this project's
1 anchor + 2 nodes): the anchor ranges against all nodes using FiRa
"one-to-many" mode (`uwb-qorvo-tools`'s own
`--node`/`--n_controlees`/`--mac`/`--dest-mac` flags), with each node
assigned a distinct MAC address and every sample tagged with which node's
MAC it came from (`uwb_mac_address` in the cut trial `.npz`, see below).
**This path is not exercised by `UWB_lab`'s documented lab and has not been
verified against physical DWM3001CDK hardware** — `--uwb-slots-per-rr`
defaults to a heuristic (`6 * node count`) that may need tuning. Before
trusting it for real data collection, run a short (`--trials 1 --duration 3`)
smoke test and check `data/raw/<session>/uwb_logs/anchor/anchor_terminal_log.txt`
for `status: Ok` against every node's MAC — if ranging is unstable or a node
never shows `Ok`, try increasing `--uwb-slots-per-rr` first.

Per-board logs and device-reset logs are written under
`data/raw/<session>/uwb_logs/`.

## Step-by-step: getting each sensor ready to collect

Do this once per sensor, in order, before starting a real session. Only pass
the flag for a sensor once you've actually plugged it in — `collect.py`
skips any sensor whose port/flag is omitted, which is also how you build a
single-sensor baseline.

**mmWave radar**
1. Plug the xWRL6432 EVM in over its USB-to-UART adapter.
2. Find the port: `ls /dev/cu.usbserial*` (macOS) / Device Manager (Windows).
3. If it was used in a previous run this session, power-cycle the EVM first
   — it won't accept a new config otherwise.
4. Pass `--mmwave-port <port>` (and `--mmwave-cfg <path>` if you want a
   config other than the default `near_field_hand_50cm.cfg`).

**IMU**
1. Plug in the ESP32 Core2 running the `IMU_lab_students` firmware over
   Type-C.
2. Find the port: `ls /dev/cu.usbserial*` or
   `python -m serial.tools.list_ports -v`.
3. Pass `--imu-port <port>` (`--imu-baud` only if your firmware isn't at the
   default 115200).

**UWB**
1. Plug in the anchor board and every node board over micro-USB — each gets
   its own port.
2. Find each port: `ls /dev/cu.usbmodem*`.
3. Confirm your group's class-sheet-assigned preamble code and channel —
   every board in the setup must match.
4. Pass `--uwb-anchor-port <port>`, one `--uwb-node-port <port>` per node,
   `--uwb-group-id`, `--uwb-preamble-code`, and `--uwb-channel`.

**RFID**
1. Power on the RFID reader and connect your laptop to its network.
2. Confirm it's reachable at `192.168.137.1:9055` (the default — override
   with `--rfid-host`/`--rfid-tcp-port` if your reader is set up
   differently).
3. Pass `--rfid` to enable it.

## Verifying a sensor is actually collecting data

Do this per sensor — one at a time — before trusting it in a real
multi-sensor session. Run a throwaway 1-trial, short-duration session with
only that sensor's flag set:

```bash
python src/collect.py --collector smoketest --mmwave-port /dev/cu.usbserial-AAAA \
  --gesture pull --trials 1 --duration 3 --auto-accept
```

**While it's recording**, watch the console. After the trial window closes,
`collect.py` prints a line like:

```text
Captured trial: mmwave=142f
```

(`imu=NNNL`, `uwb=NNNsamples`, `rfid=NNNL` for the other sensors — one entry
per enabled sensor.) If the count for your sensor is `0`, it's not actually
streaming — stop and check wiring/port before collecting anything real. A
nonzero, steadily-growing count across a couple of test trials is the
live-signal check; the file checks below are the after-the-fact check on the
same session.

Then confirm the written file backs that up, in
`data/raw/session_smoketest_.../`:

- **mmWave** — `mmwave.npz` should have a nonzero frame count:
  ```bash
  python -c "import numpy as np; d = np.load('data/raw/session_smoketest_.../mmwave.npz'); print(d['frame_number'].shape)"
  ```
- **IMU** — `imu.csv` should have more than just the header row, with
  non-placeholder `ax..gz` values:
  ```bash
  wc -l data/raw/session_smoketest_.../imu.csv
  tail -5 data/raw/session_smoketest_.../imu.csv
  ```
- **UWB** — `uwb.csv` should have rows with `status=Ok` and a real
  `distance_cm`; if it's empty, check the anchor's own log:
  ```bash
  tail -5 data/raw/session_smoketest_.../uwb.csv
  grep "status: Ok" data/raw/session_smoketest_.../uwb_logs/anchor/anchor_terminal_log.txt
  ```
- **RFID** — `rfid.csv` should have rows with a real `epc` value (hold a
  tag near the reader during the trial so there's something to read):
  ```bash
  tail -5 data/raw/session_smoketest_.../rfid.csv
  ```

If a sensor comes back empty, see [Troubleshooting](#troubleshooting) below
for that sensor's specific failure modes before re-running the smoke test.
Once every sensor you plan to use passes this check individually, combine
their flags into one real collection session.

## Collecting data

Only pass `--*-port` (and `--rfid`) for sensors you actually have wired up
right now — any sensor without a port is skipped, which is how you'd collect
a single-sensor baseline.

`collect.py` is a **coordinator**: every enabled sensor streams
continuously for the whole session on its own background thread; `collect.py`
owns the clock (`time.monotonic()` from session start) and writes event
markers (`session_start`/`trial_start`/`trial_end`/`trial_accept`/
`trial_reject`) plus each sensor's continuously-growing log. **Nothing is cut
into per-trial windows at collection time** -- that happens afterward, in
`extract_features.py cut` (see below).

Full four-sensor collection (discrete + periodic gestures in one session):

```bash
python src/collect.py \
  --collector student01 \
  --mmwave-port /dev/cu.usbserial-AAAA \
  --imu-port /dev/cu.usbserial-BBBB \
  --uwb-anchor-port /dev/cu.usbmodemCCCC \
  --uwb-node-port /dev/cu.usbmodemDDDD --uwb-node-port /dev/cu.usbmodemEEEE \
  --uwb-group-id 1 --uwb-preamble-code 9 --uwb-channel 5 \
  --rfid \
  --gesture pull,push,clapping \
  --trials 5 --duration 4 --periodic-duration 20
```

Single-sensor baseline (mmWave only):

```bash
python src/collect.py \
  --collector student01 \
  --mmwave-port /dev/cu.usbserial-AAAA \
  --trials 5 --duration 4
```

Short smoke test (1 trial, 2 gestures, short window) before a full session:

```bash
python src/collect.py \
  --collector student01 --mmwave-port /dev/cu.usbserial-AAAA \
  --gesture pull,push --trials 1 --duration 2
```

For each trial the script prompts with the gesture name, instructions,
group (`[discrete]`/`[periodic]`), and suggested sensors, waits for Enter,
records for `--duration` seconds (or `--periodic-duration` for periodic
gestures) while printing live per-sensor sample counts, then asks whether
to keep the recording (`--auto-accept` skips that prompt for unattended
runs).

Output layout:

```text
data/raw/session_<dataset-name>/
  session_metadata.json
  events.csv                  # time_s,event,trial_id,gesture,collector
  trials.csv                  # accepted trials only
  imu.csv                     # time_s,sensor,ax,ay,az,gx,gy,gz,raw_line
  uwb.csv                     # time_s,sensor,sequence,mac_address,status,distance_cm
  rfid.csv                    # time_s,sensor,epc,rssi,read_count,raw_line
  mmwave.npz                  # whole-session frame/point-cloud arrays
  uwb_logs/                   # only if UWB was enabled: device-reset + anchor/node logs
```

Only the sensors you enabled get a log file. `events.csv`/`imu.csv`/
`uwb.csv`/`rfid.csv` all use `time_s` relative to the same session-start
clock, so any tool can slice all of them consistently without needing
hardware-level sync between boards. mmWave doesn't flatten into CSV rows
the way the scalar sensors do (a range profile / point cloud is inherently
array-shaped per frame), so it's saved as one combined `.npz` for the whole
session instead.

### Cutting a session into per-trial data

```bash
python src/extract_features.py cut data/raw/session_student01_.../ \
  --output data/processed/student01_session1
```

Reads `events.csv` for each accepted trial's `[trial_start, trial_end]`
window, slices every enabled sensor's continuous log to that window, and
writes one `trial_data.npz` per trial plus a `trials.csv` manifest —
matching the same per-trial-npz shape `train.py`/`evaluate.py`/
`combine_datasets.py` consume.

**Periodic gestures are segmented here**, not during collection: a long
continuous take gets sliced into fixed-length sub-windows
(`--segment-length`, default 3.0s; `--segment-stride`, default =
`--segment-length`, i.e. non-overlapping), each becoming its own trial row
sharing the same gesture label (`<trial_id>_seg000`, `_seg001`, ...).
Discrete trials pass through unsegmented (their own recorded start/end).
Tune `--segment-length` once you've looked at how long one clap/punch/cycle
actually takes in your recordings.

Each cut `trial_data.npz` holds, per enabled sensor:
- mmWave: `mmwave_frame_number`, `mmwave_time_s`, `mmwave_range_profile`,
  `mmwave_point_count`, `mmwave_points_xyz`, `mmwave_points_velocity`.
- IMU/RFID: `{sensor}_recv_time_s` and `{sensor}_raw_lines` (raw text, one
  entry per line received during the trial window).
- UWB: `uwb_time_s`, `uwb_sequence`, `uwb_mac_address`, `uwb_status`,
  `uwb_distance_cm` (one entry per parsed ranging sample from the anchor;
  with multiple nodes, `uwb_mac_address` is what tells samples from
  different nodes apart).

### Combining datasets from multiple group members

Per the project requirements ("data from multiple group members when
possible" and evaluating a held-out person), cut each collector's raw
session first, then combine the processed datasets:

```bash
python src/combine_datasets.py \
  data/processed/student01_session1 \
  data/processed/student02_session1 \
  --output data/processed/combined
```

This concatenates manifests into one `trials.csv` (rewriting session/npz
paths to absolute paths) and reports per-gesture, per-collector, and
per-sensor-combination counts in the combined `dataset_metadata.json`.

## Training and evaluation

`extract_features.py` also extracts a fixed-length feature vector per
sensor from each cut trial's `.npz` (mmWave: energy/point-count/velocity
summary stats over the trial window; UWB: the same baseline range-shape
stats as `UWB_lab`; IMU: parses `IMU_lab_students`' `accel[g].../gyro[dps]...`
log lines into per-axis summary stats; RFID: parses `RFID_Lab`'s `<EPC>
<timestamp> <RSSI> <read_count>` report lines into RSSI/tag-count summary
stats). These are starting points, not the final word on features --
replace them once you understand what actually separates your gestures.

### Single-sensor baseline vs. fused model

`train.py --sensors` picks which sensors go into the model. One sensor gives
you the single-sensor baseline the project rubric asks for; more than one
gives you a fused model to compare against it:

```bash
# Single-sensor baseline (mmWave only)
python src/train.py data/processed/combined --sensors mmwave

# Fused model (mmWave + IMU)
python src/train.py data/processed/combined --sensors mmwave,imu
```

`--fusion early` (default) concatenates every selected sensor's features into
one vector and trains a single classifier. `--fusion late` trains one
classifier per sensor and averages their predicted probabilities at
prediction time (see the project spec's Fusion type table) -- run both to
compare:

```bash
python src/train.py data/processed/combined --sensors mmwave,imu --fusion late
```

Compare classifiers with `--classifier knn` (default) or `--classifier
svm_linear`, same starter choices as `UWB_lab`. Each run writes a
`.joblib` model + `_summary.json` (accuracy, per-class recall,
classification report, which trials were skipped and why) to `models/`,
and a confusion-matrix PNG to `results/figures/` -- override with
`--model-out`/`--confusion-out`. A simple neural network (e.g. a 1D CNN
over the raw per-sensor time series) is a good next step once KNN/SVM are
working and you want better performance -- not built here yet.

### Held-out-person evaluation

```bash
python src/train.py data/processed/combined --sensors mmwave,imu --test-collector student03
```

Trains on every collector except `student03` and tests only on their trials
-- compare this accuracy against a random train/test split to see whether
the model generalizes to a person it never saw.

### Evaluating an already-trained model

`evaluate.py` loads a saved model and reports metrics against a dataset
**without retraining** -- use it to check a model against data collected
after training (a genuinely new held-out person/session), or to regenerate
a confusion matrix without rerunning `train.py`:

```bash
python src/evaluate.py data/processed/student04_session1 \
  --model models/knn_early_mmwave-imu_20260101_120000.joblib
```

It reads the model's own `sensors`/`fusion` config, so you don't need to
re-specify them. `--test-collector` filters to one collector's trials, same
as `train.py`.

### Live evaluation

`realtime_demo.py` loads a trained model, opens only the sensor streams it
actually needs (from the model's own metadata), and predicts on a sliding
window every `--step-seconds`:

```bash
python src/realtime_demo.py \
  --model models/knn_early_mmwave-imu_20260101_120000.joblib \
  --mmwave-port /dev/cu.usbserial-XXXX \
  --imu-port /dev/cu.usbserial-YYYY \
  --duration 60 --window-seconds 3 --step-seconds 0.5 --vote-window 5
```

`--vote-window` smooths raw per-step predictions by majority vote over that
many recent predictions (1 = show raw predictions). Predictions print live
and are saved to `sessions/eval_<name>/realtime_predictions.csv`.

## Troubleshooting

- If the radar won't configure after a previous run, power-cycle the EVM and
  rerun — same as `mmwave_lab`.
- If IMU produces zero lines, double check `--imu-baud` (default 115200) and
  that the firmware is actually printing to USB serial (not just a debug
  UART).
- If RFID produces zero lines, confirm your laptop is connected to the
  reader's network and reachable at `--rfid-host`:`--rfid-tcp-port` (default
  `192.168.137.1:9055`) -- it's a TCP device, not a serial port, so a wrong
  port number here means "wrong network," not "wrong `/dev/...` path."
- If UWB produces zero Ok samples: confirm all boards are on the
  class-sheet-assigned preamble code/channel, that no other terminal/process
  already has any port open, and check
  `data/raw/<session>/uwb_logs/anchor/anchor_terminal_log.txt` for the
  raw `run_fira_twr.py` output. Use `--uwb-skip-device-reset` only if the
  boards are already known-good — a bad reset is a common cause of a silent
  anchor. With multiple nodes, if only some nodes' MACs ever show
  `status: Ok`, try raising `--uwb-slots-per-rr` first (see the UWB wiring
  section above).
- `Ctrl+C` stops the collector cleanly during a trial recording; it sends
  `sensorStop 0` to the radar, resets the UWB boards, and closes all serial
  ports/subprocesses before exiting. Whatever was already logged up to that
  point in `events.csv`/the per-sensor CSVs is preserved (the session is
  still cuttable), but the interrupted trial itself has no `trial_accept`
  event and won't show up in `trials.csv`.
- Use `--min-mmwave-frames` / `--min-sensor-lines` to tune how aggressively
  short/glitchy trials get auto-discarded during collection. `--min-uwb-samples`
  does the same for UWB (based on Ok range samples rather than raw lines).
  These are a live sanity check during collection, separate from anything
  in the cut step.
- If `extract_features.py cut` produces zero rows for a session, check that
  every accepted trial in `trials.csv` has a matching `trial_start`/
  `trial_end` pair in `events.csv` (an interrupted trial won't).

Raw recordings, trained models, and result figures are ignored by git under
`data/`, `sessions/`, `models/`, and `results/`.
