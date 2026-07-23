#!/usr/bin/env python3
"""
Multi-sensor gesture data collector: coordinator + continuous per-sensor logs.

Modeled on the final-project implementation hints (Shanmu Wang): all sensors
start streaming first and keep streaming for the whole session; this script
is the "coordinator" that owns the official clock (`time.monotonic()` from
session start) and writes event markers (`session_start`, `trial_start`,
`trial_end`, `trial_accept`, `trial_reject`) to `events.csv`. Each sensor's
samples are logged continuously, tagged with timestamps from that same
clock:
    - imu.csv, rfid.csv: one row per parsed sample, PLUS the original raw
      line -- so nothing is lost even if a parser is wrong or incomplete.
    - uwb.csv: one row per parsed ranging sample (already structured --
      `UwbReader`'s log parser discards the raw text upstream of this).
    - mmwave.npz: one combined per-session array file (frame/point-cloud
      data doesn't flatten into CSV rows the way scalar sensors do).

Cutting each sensor's continuous log into per-trial windows happens
OFFLINE, in `extract_features.py`'s `cut` step -- not here. This script's
only job is to stream, prompt, and log.

Two gesture-collection modes (see `gestures.py`'s `GestureSpec.group`):
    - "discrete" gestures (Pull, Push, Clockwise, ...): trial-by-trial,
      prompt -> countdown -> record `--duration` seconds -> keep/redo.
    - "periodic" gestures (Clapping, boxing, Palm Up-Down, Soli): one long
      continuous take per trial (`--periodic-duration` seconds); segmenting
      that into individual cycles happens later, in the cut step
      (`--segment-length`/`--segment-stride`).

Output layout (`data/raw/session_<collector>_<timestamp>/`):
    session_metadata.json
    events.csv
    trials.csv                 (accepted trials only)
    imu.csv, uwb.csv, rfid.csv, mmwave.npz   (whichever sensors are enabled)
    uwb_logs/                  (UWB anchor/node subprocess logs, if enabled)

Example (run from the repo root; all four sensors, discrete + periodic
gestures in one session):

    python src/collect.py \\
        --collector student01 \\
        --mmwave-port /dev/cu.usbserial-XXXX \\
        --imu-port /dev/cu.usbserial-YYYY \\
        --uwb-anchor-port /dev/cu.usbmodemZZZZ \\
        --uwb-node-port /dev/cu.usbmodemWWWW --uwb-node-port /dev/cu.usbmodemVVVV \\
        --uwb-group-id 1 --uwb-preamble-code 9 --uwb-channel 5 \\
        --rfid \\
        --gesture pull,push,clapping \\
        --trials 5 --duration 4 --periodic-duration 20

Then cut the raw session into a processed dataset:

    python src/extract_features.py cut data/raw/session_student01_... \\
        --output data/processed/student01_session1
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from extract_features import parse_imu_line, parse_rfid_line
from gestures import GESTURES, normalize_gestures
from sensors.common import DATA_PROCESSED_DIR, DATA_RAW_DIR, safe_label, timestamp, write_json
from sensors.imu_reader import ImuReader
from sensors.mmwave_reader import MmwaveReader
from sensors.rfid_reader import RfidReader
from sensors.uwb_reader import UwbReader

# Resolved relative to this file (src/), not the current working directory,
# so `--mmwave-cfg` defaults correctly whether you run this as
# `python src/collect.py` from the repo root or `python collect.py` from
# inside src/.
DEFAULT_MMWAVE_CFG = Path(__file__).resolve().parent / "mmwave" / "xwrL64xx-evm" / "near_field_hand_50cm.cfg"

EVENTS_FIELDNAMES = ["time_s", "event", "trial_id", "gesture", "collector"]
TRIALS_FIELDNAMES = [
    "trial_id",
    "collector",
    "gesture",
    "gesture_group",
    "trial_index",
    "attempt_index",
    "planned_duration_s",
    "actual_duration_s",
    "sensors_enabled",
    "session_dir",
]

IMU_CSV_FIELDNAMES = ["time_s", "sensor", "ax", "ay", "az", "gx", "gy", "gz", "raw_line"]
UWB_CSV_FIELDNAMES = ["time_s", "sensor", "sequence", "mac_address", "status", "distance_cm"]
RFID_CSV_FIELDNAMES = ["time_s", "sensor", "epc", "rssi", "read_count", "raw_line"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream all enabled sensors continuously and log trial event markers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--collector", required=True, help="Student/collector ID.")
    parser.add_argument(
        "--gesture",
        action="append",
        help="Gesture label (see gestures.py). Can be repeated or comma separated. Default: all 15 gestures.",
    )
    parser.add_argument("--trials", type=int, default=5, help="Trials/takes to collect per gesture.")
    parser.add_argument("--duration", type=float, default=4.0, help="Seconds per discrete-gesture trial.")
    parser.add_argument(
        "--periodic-duration",
        type=float,
        default=20.0,
        help="Seconds per periodic-gesture continuous take (segmented later by extract_features.py cut).",
    )
    parser.add_argument("--dataset-name", default=f"session_{timestamp()}")
    parser.add_argument("--out-root", default=str(DATA_RAW_DIR))
    parser.add_argument("--auto-accept", action="store_true")
    parser.add_argument(
        "--min-mmwave-frames", type=int, default=10, help="Only checked if --mmwave-port is set."
    )
    parser.add_argument(
        "--min-sensor-lines",
        type=int,
        default=5,
        help="Minimum raw lines required per enabled IMU/RFID sensor to keep a trial.",
    )
    parser.add_argument(
        "--min-uwb-samples",
        type=int,
        default=5,
        help="Minimum Ok range samples required from UWB, if enabled, to keep a trial.",
    )

    mmwave_group = parser.add_argument_group("mmWave radar (TI xWRL6432)")
    mmwave_group.add_argument("--mmwave-port", help="Radar CLI/data serial port.")
    mmwave_group.add_argument("--mmwave-cfg", type=Path, default=DEFAULT_MMWAVE_CFG)
    mmwave_group.add_argument("--mmwave-baud", type=int, default=115200)
    mmwave_group.add_argument("--no-mmwave-warm-reset", action="store_true")

    imu_group = parser.add_argument_group("IMU (ESP32 Core2 + BMI270)")
    imu_group.add_argument("--imu-port", help="IMU serial port.")
    imu_group.add_argument("--imu-baud", type=int, default=115200)

    uwb_group = parser.add_argument_group("UWB (Qorvo DWM3001CDK FiRa TWR: anchor + node(s))")
    uwb_group.add_argument("--uwb-anchor-port", help="UWB anchor (fixed, FiRa controller) serial port.")
    uwb_group.add_argument(
        "--uwb-node-port",
        action="append",
        help=(
            "UWB node (worn, FiRa controlee) serial port. Repeat for multiple nodes -- this "
            "project's kit is 1 anchor + 2 nodes. NOTE: the multi-node one-to-many path is "
            "unverified against real DWM3001CDK hardware -- smoke-test it first."
        ),
    )
    uwb_group.add_argument(
        "--uwb-group-id", type=int, help="Class-sheet group number (required if UWB is enabled)."
    )
    uwb_group.add_argument(
        "--uwb-preamble-code",
        type=int,
        default=10,
        help="Sheet-assigned FiRa preamble code (one of 9, 10, 11, 12).",
    )
    uwb_group.add_argument(
        "--uwb-channel", type=int, choices=[5, 9], default=9, help="Sheet-assigned UWB channel."
    )
    uwb_group.add_argument("--uwb-fps", type=float, default=50.0, help="Target ranging update rate.")
    uwb_group.add_argument("--uwb-slot-span", type=int, default=2400)
    uwb_group.add_argument(
        "--uwb-slots-per-rr",
        type=int,
        default=None,
        help=(
            "Slots per ranging round. Default: 6 for a single node (matches UWB_lab); "
            "6 * node count for multiple nodes (unverified heuristic)."
        ),
    )
    uwb_group.add_argument(
        "--uwb-skip-device-reset",
        action="store_true",
        help="Skip the UCI device reset before/after ranging (use if it's already known-good).",
    )

    rfid_group = parser.add_argument_group("RFID (RFID_Lab reader, TCP -- not serial)")
    rfid_group.add_argument("--rfid", action="store_true", help="Enable the RFID reader.")
    rfid_group.add_argument(
        "--rfid-host", default="192.168.137.1", help="RFID reader network address (see RFID_Lab)."
    )
    rfid_group.add_argument("--rfid-tcp-port", type=int, default=9055, help="RFID reader TCP port.")
    rfid_group.add_argument(
        "--rfid-epcs",
        nargs="+",
        default=None,
        help=(
            "Only keep reads from these EPCs (your group's own tags), same idea as "
            "RFID_Lab's --epcs/SELECTED_EPCS filter. Reads from any other EPC (e.g. another "
            "group's tags in range) are discarded before they're written to rfid.csv or "
            "counted toward --min-sensor-lines."
        ),
    )

    return parser.parse_args()


def normalize_epcs(epcs: list[str] | None) -> set[str] | None:
    return {epc.upper() for epc in epcs} if epcs else None


def prompt_ready(gesture_name: str, trial_index: int, trials: int, duration_s: float, is_periodic: bool) -> None:
    spec = GESTURES[gesture_name]
    print()
    print("=" * 60)
    print(f"Gesture: {spec.display_name} ({gesture_name}) [{spec.group}]")
    print(f"Trial {trial_index}/{trials}")
    print(f"Suggested sensors: {', '.join(spec.suggested_sensors)}")
    print(spec.instruction)
    if is_periodic:
        print(
            f"This is a PERIODIC gesture -- perform it repeatedly, continuously, for the "
            f"whole take. It will be segmented into individual cycles later."
        )
    input(f"Press Enter to record {duration_s:.1f} seconds...")


def prompt_keep_recording() -> bool:
    while True:
        answer = input("Keep this recording? [Y/n] ").strip().lower()
        if answer in {"", "y", "yes"}:
            return True
        if answer in {"n", "no", "r", "redo"}:
            return False
        print("Please answer Y or N.")


class SensorSet:
    """Opens/holds whichever sensors were requested on the command line."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.mmwave: MmwaveReader | None = None
        self.imu: ImuReader | None = None
        self.uwb: UwbReader | None = None
        self.rfid: RfidReader | None = None

        if bool(args.uwb_anchor_port) != bool(args.uwb_node_port):
            raise SystemExit(
                "UWB needs both --uwb-anchor-port and at least one --uwb-node-port "
                "(it's an anchor+node ranging setup, not a single device)."
            )
        if args.uwb_anchor_port and args.uwb_group_id is None:
            raise SystemExit("--uwb-group-id is required when UWB is enabled.")

        try:
            if args.mmwave_port:
                print(f"Opening mmWave radar on {args.mmwave_port} (cfg: {args.mmwave_cfg})...")
                self.mmwave = MmwaveReader(
                    port_path=args.mmwave_port,
                    cfg_path=args.mmwave_cfg,
                    baud=args.mmwave_baud,
                    warm_reset=not args.no_mmwave_warm_reset,
                )
            if args.imu_port:
                print(f"Opening IMU on {args.imu_port}...")
                self.imu = ImuReader("imu", args.imu_port, args.imu_baud)
            if args.uwb_anchor_port:
                print(
                    f"Opening UWB ranging (anchor {args.uwb_anchor_port}, "
                    f"nodes {', '.join(args.uwb_node_port)})..."
                )
                self.uwb = UwbReader(
                    anchor_port=args.uwb_anchor_port,
                    node_ports=args.uwb_node_port,
                    group_id=args.uwb_group_id,
                    log_dir=Path(args.out_root).expanduser().resolve() / args.dataset_name / "uwb_logs",
                    preamble_code=args.uwb_preamble_code,
                    channel=args.uwb_channel,
                    fps=args.uwb_fps,
                    slot_span=args.uwb_slot_span,
                    slots_per_rr=args.uwb_slots_per_rr,
                    reset_devices_first=not args.uwb_skip_device_reset,
                )
            if args.rfid:
                print(f"Opening RFID reader at {args.rfid_host}:{args.rfid_tcp_port}...")
                self.rfid = RfidReader(args.rfid_host, args.rfid_tcp_port)
        except Exception:
            self.close()
            raise

        if not self.enabled_names():
            self.close()
            raise SystemExit(
                "No sensors enabled. Pass at least one of --mmwave-port, --imu-port, "
                "--uwb-anchor-port/--uwb-node-port, --rfid."
            )

        # Let boards settle and start producing data before the first trial.
        time.sleep(0.5)

    def enabled_names(self) -> list[str]:
        names = []
        if self.mmwave is not None:
            names.append("mmwave")
        if self.imu is not None:
            names.append("imu")
        if self.uwb is not None:
            names.append("uwb")
        if self.rfid is not None:
            names.append("rfid")
        return names

    def check_errors(self) -> None:
        for reader in (self.mmwave, self.imu, self.uwb, self.rfid):
            if reader is not None:
                reader.check_error()

    def status_line(self) -> str:
        parts = []
        if self.mmwave is not None:
            parts.append(f"mmwave={self.mmwave.sample_count}f")
        if self.imu is not None:
            parts.append(f"imu={self.imu.sample_count}L")
        if self.uwb is not None:
            parts.append(f"uwb={self.uwb.sample_count}samples")
        if self.rfid is not None:
            parts.append(f"rfid={self.rfid.sample_count}L")
        return " ".join(parts)

    def close(self) -> None:
        for reader in (self.mmwave, self.imu, self.uwb, self.rfid):
            if reader is not None:
                try:
                    reader.close()
                except Exception as error:  # noqa: BLE001 - best-effort cleanup
                    print(f"Warning: error while closing sensor: {error}")


class SessionLogger:
    """Coordinator: owns events.csv/trials.csv and continuously drains each
    reader's buffer into its own per-session log (imu.csv/uwb.csv/rfid.csv),
    tagged with time_s relative to the fixed session start."""

    def __init__(
        self,
        session_dir: Path,
        sensors: SensorSet,
        session_start: float,
        rfid_epcs: set[str] | None = None,
    ) -> None:
        self.session_dir = session_dir
        self.sensors = sensors
        self.session_start = session_start
        self.rfid_epcs = rfid_epcs
        self._written = {"imu": 0, "uwb": 0, "rfid": 0}
        self._csv_files: dict[str, tuple] = {}

        if sensors.imu is not None:
            self._open_csv("imu", IMU_CSV_FIELDNAMES)
        if sensors.uwb is not None:
            self._open_csv("uwb", UWB_CSV_FIELDNAMES)
        if sensors.rfid is not None:
            self._open_csv("rfid", RFID_CSV_FIELDNAMES)

        self._events_file = open(session_dir / "events.csv", "w", newline="")
        self._events_writer = csv.DictWriter(self._events_file, fieldnames=EVENTS_FIELDNAMES)
        self._events_writer.writeheader()

        self._trials_file = open(session_dir / "trials.csv", "w", newline="")
        self._trials_writer = csv.DictWriter(self._trials_file, fieldnames=TRIALS_FIELDNAMES)
        self._trials_writer.writeheader()

    def _open_csv(self, name: str, fieldnames: list[str]) -> None:
        file = open(self.session_dir / f"{name}.csv", "w", newline="")
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        self._csv_files[name] = (file, writer)

    def write_event(self, event: str, trial_id: str, gesture: str, collector: str, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        self._events_writer.writerow(
            {
                "time_s": f"{now - self.session_start:.6f}",
                "event": event,
                "trial_id": trial_id,
                "gesture": gesture,
                "collector": collector,
            }
        )
        self._events_file.flush()

    def write_trial_row(self, row: dict) -> None:
        self._trials_writer.writerow({name: row.get(name, "") for name in TRIALS_FIELDNAMES})
        self._trials_file.flush()

    def drain(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        if self.sensors.imu is not None:
            self._drain_imu(now)
        if self.sensors.uwb is not None:
            self._drain_uwb(now)
        if self.sensors.rfid is not None:
            self._drain_rfid(now)

    def _drain_imu(self, now: float) -> None:
        lines = self.sensors.imu.window(self.session_start, now)
        new_lines = lines[self._written["imu"] :]
        if not new_lines:
            return
        _, writer = self._csv_files["imu"]
        for rel_t, line in new_lines:
            parsed = parse_imu_line(line)
            ax, ay, az, gx, gy, gz = parsed if parsed is not None else ("",) * 6
            writer.writerow(
                {
                    "time_s": f"{rel_t:.6f}",
                    "sensor": "imu",
                    "ax": ax,
                    "ay": ay,
                    "az": az,
                    "gx": gx,
                    "gy": gy,
                    "gz": gz,
                    "raw_line": line,
                }
            )
        self._csv_files["imu"][0].flush()
        self._written["imu"] = len(lines)

    def _drain_uwb(self, now: float) -> None:
        window = self.sensors.uwb.window(self.session_start, now)
        total = len(window["time_s"])
        start_index = self._written["uwb"]
        if start_index >= total:
            return
        _, writer = self._csv_files["uwb"]
        for i in range(start_index, total):
            writer.writerow(
                {
                    "time_s": f"{window['time_s'][i]:.6f}",
                    "sensor": "uwb",
                    "sequence": int(window["sequence"][i]),
                    "mac_address": window["mac_address"][i],
                    "status": window["status"][i],
                    "distance_cm": float(window["distance_cm"][i]),
                }
            )
        self._csv_files["uwb"][0].flush()
        self._written["uwb"] = total

    def _drain_rfid(self, now: float) -> None:
        lines = self.sensors.rfid.window(self.session_start, now)
        new_lines = lines[self._written["rfid"] :]
        if not new_lines:
            return
        _, writer = self._csv_files["rfid"]
        for rel_t, line in new_lines:
            parsed = parse_rfid_line(line)
            if parsed is not None and self.rfid_epcs is not None and parsed[0].upper() not in self.rfid_epcs:
                continue  # another tag's read (e.g. a different group's) -- not ours, drop it
            epc, rssi, read_count = parsed if parsed is not None else ("", "", "")
            writer.writerow(
                {
                    "time_s": f"{rel_t:.6f}",
                    "sensor": "rfid",
                    "epc": epc,
                    "rssi": rssi,
                    "read_count": read_count,
                    "raw_line": line,
                }
            )
        self._csv_files["rfid"][0].flush()
        self._written["rfid"] = len(lines)

    def finalize(self, final_now: float) -> None:
        self.drain(final_now)
        for file, _ in self._csv_files.values():
            file.close()
        self._events_file.close()
        self._trials_file.close()

        if self.sensors.mmwave is not None:
            data = self.sensors.mmwave.window(self.session_start, final_now)
            np.savez_compressed(
                self.session_dir / "mmwave.npz",
                mmwave_frame_number=data["frame_number"],
                mmwave_time_s=data["time_s"],
                mmwave_range_profile=data["range_profile"],
                mmwave_point_count=data["point_count"],
                mmwave_points_xyz=data["points_xyz"],
                mmwave_points_velocity=data["points_velocity"],
            )


def capture_window(sensors: SensorSet, logger: SessionLogger, duration_s: float) -> tuple[float, float]:
    """Wait out the recording window, printing live sensor counters and draining logs."""
    start = time.monotonic()
    next_status = start
    next_drain = start
    while True:
        elapsed = time.monotonic() - start
        if elapsed >= duration_s:
            break
        sensors.check_errors()
        now = time.monotonic()
        if now >= next_drain:
            logger.drain(now)
            next_drain = now + 1.0
        if now >= next_status:
            remaining = max(0.0, duration_s - elapsed)
            print(
                f"\rRecording: {elapsed:5.1f}/{duration_s:.1f} s, "
                f"{sensors.status_line()}, {remaining:5.1f} s left",
                end="",
                flush=True,
            )
            next_status = now + 0.2
        time.sleep(0.02)
    print()
    return start, time.monotonic()


def trial_ok(
    sensors: SensorSet,
    args: argparse.Namespace,
    trial_start: float,
    trial_end: float,
    rfid_epcs: set[str] | None = None,
) -> str | None:
    """Live per-trial quality check (independent of the continuous logs) -- returns
    None if the trial meets minimum-sample thresholds, else a reason string."""
    if sensors.mmwave is not None:
        mmwave_data = sensors.mmwave.window(trial_start, trial_end)
        if len(mmwave_data["frame_number"]) < args.min_mmwave_frames:
            return f"only {len(mmwave_data['frame_number'])} mmWave frames (need {args.min_mmwave_frames})"
    if sensors.imu is not None:
        imu_lines = sensors.imu.window(trial_start, trial_end)
        if len(imu_lines) < args.min_sensor_lines:
            return f"only {len(imu_lines)} IMU lines (need {args.min_sensor_lines})"
    if sensors.uwb is not None:
        uwb_window = sensors.uwb.window(trial_start, trial_end)
        ok_count = int(np.count_nonzero(uwb_window["status"] == "Ok"))
        if ok_count < args.min_uwb_samples:
            return f"only {ok_count} Ok UWB range samples (need {args.min_uwb_samples})"
    if sensors.rfid is not None:
        rfid_lines = sensors.rfid.window(trial_start, trial_end)
        if rfid_epcs is not None:
            rfid_lines = [
                (rel_t, line)
                for rel_t, line in rfid_lines
                if (parsed := parse_rfid_line(line)) is not None and parsed[0].upper() in rfid_epcs
            ]
        if len(rfid_lines) < args.min_sensor_lines:
            return f"only {len(rfid_lines)} matching RFID lines (need {args.min_sensor_lines})"
    return None


def make_session_metadata(args: argparse.Namespace, gesture_list: list[str], sensors: SensorSet) -> dict:
    return {
        "dataset_name": args.dataset_name,
        "collector": args.collector,
        "gestures": [{"name": g, "group": GESTURES[g].group} for g in gesture_list],
        "trials_per_gesture": args.trials,
        "duration_s": args.duration,
        "periodic_duration_s": args.periodic_duration,
        "sensors_enabled": sensors.enabled_names(),
        "mmwave_cfg_path": str(args.mmwave_cfg) if sensors.mmwave is not None else None,
        "mmwave_range_bin_count": (
            sensors.mmwave.range_config.num_range_bins
            if sensors.mmwave is not None and sensors.mmwave.range_config is not None
            else None
        ),
        "ports": {
            "mmwave": args.mmwave_port,
            "imu": args.imu_port,
            "uwb_anchor": args.uwb_anchor_port,
            "uwb_nodes": args.uwb_node_port,
            "rfid": f"{args.rfid_host}:{args.rfid_tcp_port}" if args.rfid else None,
        },
        "rfid_epcs": sorted(normalize_epcs(args.rfid_epcs)) if sensors.rfid is not None and args.rfid_epcs else None,
        "uwb_config": (
            {
                "group_id": args.uwb_group_id,
                "preamble_code": args.uwb_preamble_code,
                "channel": args.uwb_channel,
                "fps": args.uwb_fps,
                "ranging_span_ms": sensors.uwb.ranging_span_ms,
                "slot_span": args.uwb_slot_span,
                "slots_per_rr": sensors.uwb.slots_per_rr,
                "multi_node": sensors.uwb.multi_node,
            }
            if sensors.uwb is not None
            else None
        ),
        "created_at": timestamp(),
    }


def main() -> int:
    args = parse_args()
    gesture_list = normalize_gestures(args.gesture)
    if args.trials < 1:
        raise SystemExit("--trials must be at least 1.")
    if args.duration <= 0 or args.periodic_duration <= 0:
        raise SystemExit("--duration and --periodic-duration must be positive.")

    session_dir = Path(args.out_root).expanduser().resolve() / args.dataset_name
    session_dir.mkdir(parents=True, exist_ok=True)

    sensors = SensorSet(args)
    rfid_epcs = normalize_epcs(args.rfid_epcs)
    if rfid_epcs is not None:
        print(f"RFID EPC filter active -- keeping only: {', '.join(sorted(rfid_epcs))}")
    session_start = time.monotonic()
    logger = SessionLogger(session_dir, sensors, session_start, rfid_epcs=rfid_epcs)
    logger.write_event("session_start", "", "", args.collector, now=session_start)

    write_json(session_dir / "session_metadata.json", make_session_metadata(args, gesture_list, sensors))

    print(f"Raw session folder: {session_dir}")
    print(f"Collector: {args.collector}")
    print(f"Gestures: {', '.join(gesture_list)}")
    print(f"Sensors enabled: {', '.join(sensors.enabled_names())}")

    interrupted = False
    accepted_total = 0
    try:
        for gesture in gesture_list:
            spec = GESTURES[gesture]
            is_periodic = spec.group == "periodic"
            planned_duration = args.periodic_duration if is_periodic else args.duration

            trial_index = 1
            while trial_index <= args.trials:
                attempt_index = 1
                while True:
                    logger.drain()
                    prompt_ready(gesture, trial_index, args.trials, planned_duration, is_periodic)
                    trial_id = f"{safe_label(args.collector)}_{safe_label(gesture)}_{trial_index:03d}"

                    trial_start, trial_end = capture_window(sensors, logger, planned_duration)
                    logger.write_event("trial_start", trial_id, gesture, args.collector, now=trial_start)
                    logger.write_event("trial_end", trial_id, gesture, args.collector, now=trial_end)
                    logger.drain(trial_end)

                    reject_reason = trial_ok(sensors, args, trial_start, trial_end, rfid_epcs=rfid_epcs)
                    if reject_reason is not None:
                        print(f"Discarded: {reject_reason}.")
                        logger.write_event("trial_reject", trial_id, gesture, args.collector)
                        attempt_index += 1
                        continue

                    print(f"Captured trial: {sensors.status_line()}")
                    keep = args.auto_accept or prompt_keep_recording()
                    if keep:
                        logger.write_event("trial_accept", trial_id, gesture, args.collector)
                        logger.write_trial_row(
                            {
                                "trial_id": trial_id,
                                "collector": args.collector,
                                "gesture": gesture,
                                "gesture_group": spec.group,
                                "trial_index": trial_index,
                                "attempt_index": attempt_index,
                                "planned_duration_s": planned_duration,
                                "actual_duration_s": trial_end - trial_start,
                                "sensors_enabled": ",".join(sensors.enabled_names()),
                                "session_dir": str(session_dir),
                            }
                        )
                        accepted_total += 1
                        print(f"Accepted {gesture} trial {trial_index}/{args.trials} ({trial_id})")
                        trial_index += 1
                        break

                    print("Discarded recording. Redoing the same trial.")
                    logger.write_event("trial_reject", trial_id, gesture, args.collector)
                    attempt_index += 1

    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted. Stopping sensors...")
    except RuntimeError as error:
        interrupted = True
        print(f"\nCapture failed: {error}")
    finally:
        # close() joins each reader's background thread, so whatever it captured up to
        # that point is final -- capture final_now afterward so finalize()'s drain/window
        # calls don't miss samples that arrived while close() was still shutting down.
        sensors.close()
        logger.finalize(time.monotonic())

    print(f"Accepted trials: {accepted_total}")
    print(f"Raw session complete: {session_dir}")
    print(
        "Next: python src/extract_features.py cut "
        f"{session_dir} --output {DATA_PROCESSED_DIR / args.dataset_name}"
    )
    return 1 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
