# UWB FiRa TWR Student Lab Operation

This lab uses Qorvo DWM3001CDK UWB modules. Each board has already been flashed
with the UCI firmware build. Do not re-flash the boards unless the instructor
asks you to do so.

You will install the UWB Qorvo Tools, identify your two serial ports, run a
FiRa two-way ranging experiment, and save the data files for later location
accuracy analysis.

## 1. Install UWB Qorvo Tools

Use Python 3.10. From the SDK folder:

```bash
cd SDK/Tools/uwb-qorvo-tools
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

On Windows PowerShell:

```powershell
cd SDK\Tools\uwb-qorvo-tools
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install .
```

All commands below assume the virtual environment is active and your terminal is
in `SDK/Tools/uwb-qorvo-tools`.

## 2. Find Your Board Ports

Connect both DWM3001CDK boards over USB.

macOS:

```bash
ls /dev/cu.usbmodem*
```

Linux:

```bash
ls /dev/serial/by-id/*
```

Windows:

Use Device Manager and look under `Ports (COM & LPT)`.

In the commands below, replace:

- `CONTROLLER_PORT` with the port for the controller/initiator board.
- `CONTROLEE_PORT` with the port for the controlee/responder board.
- `GROUP_ID` with your assigned lab group number.
- `PREAMBLE_CODE` with the preamble code assigned in the class sheet.
- `UWB_CHANNEL` with the channel assigned in the class sheet.

Form your group and find your assigned preamble code and channel here:

https://docs.google.com/spreadsheets/d/1xEM-HSpBDMrXdBSz2BphMjZDY_hA6qhXVAFCRKVvYuw/edit?usp=sharing

There are eight groups total. The groups are split across preamble codes
`9, 10, 11, 12` and channels `5, 9` to reduce interference.

Example macOS ports:

```bash
export CONTROLLER_PORT=/dev/cu.usbmodemD46FFE3655DD1
export CONTROLEE_PORT=/dev/cu.usbmodemE89E195B6C731
export GROUP_ID=1
export PREAMBLE_CODE=9
export UWB_CHANNEL=5
```

Set these variables in every terminal you use. If your shell does not support
this syntax, replace `$CONTROLLER_PORT`, `$CONTROLEE_PORT`, `$GROUP_ID`,
`$PREAMBLE_CODE`, and `$UWB_CHANNEL` in the commands with the actual values.

## 3. Check Both Boards

Run this once for each board:

```bash
python scripts/device/get_device_info/get_device_info.py -p $CONTROLLER_PORT
python scripts/device/get_device_info/get_device_info.py -p $CONTROLEE_PORT
```

Both boards should report `status: Ok`.

## 4. Group Preamble Code And Channel

Each group must use the preamble code and channel assigned in the class sheet.
Do not use your group number as the preamble code unless the sheet explicitly
assigns that value.

Both boards in the same group must use the same values:

```bash
--channel $UWB_CHANNEL
--preamble-idx $PREAMBLE_CODE
```

Do not use another group's preamble code or channel. This assignment reduces
interference when multiple groups run ranging at the same time.

## 5. Run FiRa TWR

Open two terminals. Activate the same Python environment in both terminals:

```bash
cd SDK/Tools/uwb-qorvo-tools
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
cd SDK\Tools\uwb-qorvo-tools
.\.venv\Scripts\Activate.ps1
```

### Terminal 1: Controlee

Start the controlee/responder first:

```bash
python scripts/fira/run_fira_twr/run_fira_twr.py \
  -p $CONTROLEE_PORT \
  --controlee \
  --channel $UWB_CHANNEL \
  --preamble-idx $PREAMBLE_CODE \
  --aoa-report all-disabled \
  -t 30
```

### Terminal 2: Controller

Start the controller/initiator second:

```bash
python scripts/fira/run_fira_twr/run_fira_twr.py \
  -p $CONTROLLER_PORT \
  --channel $UWB_CHANNEL \
  --preamble-idx $PREAMBLE_CODE \
  --aoa-report all-disabled \
  -t 30
```

The DWM3001CDK board does not support AoA, so this lab disables AoA reporting.

Successful ranging output contains measurements with:

```text
status: Ok (0x0)
distance: <value> cm
```

## 6. Configure The Update Rate

The update rate is controlled by the ranging interval. The default FiRa TWR
script uses:

```text
--slot-span 2400
--slots-per-rr 25
--ranging-span 200
```

The important value for FPS is `--ranging-span`, which is the ranging interval
in milliseconds:

```text
FPS = 1000 / ranging interval in ms
```

With the default `--ranging-span 200`:

```text
1000 / 200 = 5 FPS
```

The default profile also reserves 25 slots per ranging round. With
`--slot-span 2400`, each slot is 2 ms, so the default round has:

```text
25 slots * 2 ms = 50 ms of slot time
```

Because the default ranging interval is 200 ms, the default experiment waits
longer than the minimum slot time and produces about 5 range updates per second.

For one controller and one controlee, fewer slots are needed. This lab uses
6 slots per ranging round:

```text
6 slots * 2 ms = 12 ms minimum round time
```

The ranging interval must be greater than or equal to the required slot time:

```text
--ranging-span >= slot time * slots per ranging round
```

### Example: Change FPS

To configure a target FPS, choose:

```text
--ranging-span = 1000 / target FPS
```

For this lab, use a 50 FPS target and check that your chosen ranging interval
is not shorter than the minimum round time.

```text
target FPS = 50
slot-span = 2400
slots-per-rr = 6
```

### TODO: Configure 50 FPS

Before collecting data, configure your timing variables so the experiment runs
at 50 FPS:

```bash
export SLOT_SPAN=TODO_SLOT_SPAN
export SLOTS_PER_RR=TODO_SLOTS_PER_RR
export RANGING_SPAN=TODO_RANGING_SPAN
```

Set these three timing variables in both the controller terminal and the
controlee terminal.

Check your calculation:

```text
FPS = 1000 / RANGING_SPAN
```

The controller output should show:

```text
ranging interval:   <your computed RANGING_SPAN> ms
```

When collecting data, start the controlee first and run it longer than the
controller. Otherwise, the controller may record timeout messages after the
controlee stops.

For CIR collection, keep this timing margin. Short test runs can be misleading
because Python environment startup and JSON dump time can reduce the actual
overlap between the two boards.

## 7. Configure CIR Collection

This lab also records a small channel impulse response (CIR) window. The CIR
window is configured through calibration parameters on each board:

- `rx_diag_config.cir_n_taps`: total number of CIR bins to report.
- `rx_diag_config.cir_fp_tap_offset`: how many bins before the first-path index
  to include.

For this lab, collect 15 bins starting at first-path index minus 2:

```bash
export CIR_N_TAPS=15
export CIR_FP_TAP_OFFSET=2
```

Apply this setting to both boards:

```bash
python scripts/device/set_cal/set_cal.py \
  -p $CONTROLLER_PORT \
  rx_diag_config.cir_n_taps \
  $CIR_N_TAPS

python scripts/device/set_cal/set_cal.py \
  -p $CONTROLLER_PORT \
  rx_diag_config.cir_fp_tap_offset \
  $CIR_FP_TAP_OFFSET

python scripts/device/set_cal/set_cal.py \
  -p $CONTROLEE_PORT \
  rx_diag_config.cir_n_taps \
  $CIR_N_TAPS

python scripts/device/set_cal/set_cal.py \
  -p $CONTROLEE_PORT \
  rx_diag_config.cir_fp_tap_offset \
  $CIR_FP_TAP_OFFSET
```

Verify both boards:

```bash
python scripts/device/get_cal/get_cal.py \
  -p $CONTROLLER_PORT \
  -f n \
  rx_diag_config.cir_n_taps \
  rx_diag_config.cir_fp_tap_offset

python scripts/device/get_cal/get_cal.py \
  -p $CONTROLEE_PORT \
  -f n \
  rx_diag_config.cir_n_taps \
  rx_diag_config.cir_fp_tap_offset
```

Both commands should print:

```text
rx_diag_config.cir_n_taps           = 15
rx_diag_config.cir_fp_tap_offset    = 2
```

To request CIR in the ranging diagnostic report, add:

```bash
--diag-fields 'metrics|cir'
```

The terminal output should include CIR blocks with:

```text
# CIR Report:
    n_samples :   15
    window (i,q): [...]
```

The current decoder may print `path1_ridx: 254` for this setting. Treat that as
the raw one-byte representation of the requested `-2` first-path offset.

CIR data is much larger than distance data. Keep `CIR_N_TAPS=15` for the 50 FPS
experiment.

## 8. Save JSON Data For Later Analysis

For the data collection run, use `--diag_dump`. The tool writes JSON files named
like `range_data_YY-MM-DD-HHhMMmSSs.json` in the current working directory. With
`--diag-fields 'metrics|cir'`, those JSON files contain both ranging metrics and
the CIR samples.

Create one output folder per group and run:

```bash
mkdir -p lab_logs/group_${GROUP_ID}/controller
mkdir -p lab_logs/group_${GROUP_ID}/controlee
```

Before running the next commands, confirm that `CONTROLLER_PORT`,
`CONTROLEE_PORT`, `GROUP_ID`, `PREAMBLE_CODE`, `UWB_CHANNEL`, `SLOT_SPAN`,
`SLOTS_PER_RR`, `RANGING_SPAN`, `CIR_N_TAPS`, and `CIR_FP_TAP_OFFSET` are set
in each terminal.

The CIR report prints many lines at 50 FPS. During data collection, redirect
the output to a log file instead of displaying every line in the terminal. Wait
until the command finishes and the shell prompt returns.

### Terminal 1: Controlee Data Collection

```bash
cd SDK/Tools/uwb-qorvo-tools
source .venv/bin/activate
cd lab_logs/group_${GROUP_ID}/controlee

python ../../../scripts/fira/run_fira_twr/run_fira_twr.py \
  -p $CONTROLEE_PORT \
  --controlee \
  --channel $UWB_CHANNEL \
  --preamble-idx $PREAMBLE_CODE \
  --aoa-report all-disabled \
  --slot-span $SLOT_SPAN \
  --slots-per-rr $SLOTS_PER_RR \
  --ranging-span $RANGING_SPAN \
  --diag-fields 'metrics|cir' \
  --diag_dump \
  -t 70 \
  > controlee_terminal_log.txt 2>&1
```

### Terminal 2: Controller Data Collection

```bash
cd SDK/Tools/uwb-qorvo-tools
source .venv/bin/activate
cd lab_logs/group_${GROUP_ID}/controller

python ../../../scripts/fira/run_fira_twr/run_fira_twr.py \
  -p $CONTROLLER_PORT \
  --channel $UWB_CHANNEL \
  --preamble-idx $PREAMBLE_CODE \
  --aoa-report all-disabled \
  --slot-span $SLOT_SPAN \
  --slots-per-rr $SLOTS_PER_RR \
  --ranging-span $RANGING_SPAN \
  --diag-fields 'metrics|cir' \
  --diag_dump \
  -t 60 \
  > controller_terminal_log.txt 2>&1
```

Keep all generated `range_data_*.json` files and both terminal log files. The
JSON files contain diagnostic data, including the 15-bin CIR windows. The
terminal logs contain the printed ranging distances used for accuracy analysis.

Recommended files to submit:

```text
lab_logs/group_<GROUP_ID>/
  controller/
    controller_terminal_log.txt
    range_data_*.json
  controlee/
    controlee_terminal_log.txt
    range_data_*.json
```

## 9. If TWR Does Not Work

Reload calibration on both boards if you see any of these symptoms:

- `Device -> Error`
- `RangingTxFailed`
- repeated `RangingRxTimeout`
- `ranging_stop failed: Rejected`
- no `status: Ok (0x0)` measurements

Run the following commands for each board, replacing `PORT` with the board port:

```bash
python scripts/device/reset_calibration/reset_calibration.py \
  -p PORT \
  -y \
  --timeout 6

python scripts/device/load_cal/load_cal.py \
  -p PORT \
  -f scripts/device/load_cal/calib_files/DWM3001CDK/dual-hoe_non_aoa.json

python scripts/device/reset_device/reset_device.py -p PORT
```

Then verify calibration:

```bash
python scripts/device/get_cal/get_cal.py \
  -p PORT \
  ant_set0.tx_power_control \
  ant0.ch9.ant_delay \
  debug.pll_bias_trim
```

Expected values should look like:

```text
ant_set0.tx_power_control = 1
ant0.ch9.ant_delay        = non-zero, usually around 16000
debug.pll_bias_trim       = 4
```

After calibration is restored, rerun the FiRa TWR experiment. If you are
collecting CIR data, repeat Section 7 before the data collection run.

## 10. Cleanup

When finished, stop both scripts and disconnect the boards. Do not delete your
group's `lab_logs/group_<GROUP_ID>` folder.
