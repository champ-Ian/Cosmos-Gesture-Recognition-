#!/usr/bin/env python3
"""
Multi-sensor gesture data collector for the COSMOS gesture-recognition project.

Modeled on `mmwave_lab/collect_posture_dataset.py`'s trial-based workflow
(prompt -> record -> keep/redo -> save npz + append manifest row), extended to
run several sensors concurrently on independent background threads:

    - mmWave radar (TI xWRL6432): binary UART frames, decoded with
      `mmwave/radar_io.py` (adapted from mmwave_lab).
    - IMU (ESP32 Core2), RFID reader: newline-delimited serial text, read raw
      by `sensors/serial_json_stream.py` (see that file for the
      expected/recommended JSON-line firmware contract).
    - UWB: a Qorvo DWM3001CDK tag + one-or-more-anchors FiRa ranging setup,
      driven by `uwb/uwb_stream.py` (adapted from `UWB_lab`). This needs a
      tag port (`--uwb-tag-port`) plus at least one anchor port
      (`--uwb-anchor-port`, repeatable for multi-anchor ranging), not one
      single port.

Every enabled sensor streams continuously into its own timestamped buffer.
For each trial, the script records a wall-clock window (`--duration` seconds)
and then slices out of each sensor's buffer only the samples received in that
window. This avoids needing hardware-level sync between boards: everything is
tagged with `time.monotonic()` from the same collection-script clock.

Only pass the `--*-port` flags for sensors you actually have wired up; any
sensor without a port is skipped entirely (useful for single/dual sensor
baselines, per the "single-sensor baseline vs fused model" project
requirement).

Example (all four sensors):

    python collect_gesture_dataset.py \\
        --collector student01 \\
        --mmwave-port /dev/cu.usbserial-XXXX \\
        --imu-port /dev/cu.usbserial-YYYY \\
        --uwb-tag-port /dev/cu.usbmodemZZZZ \\
        --uwb-anchor-port /dev/cu.usbmodemWWWW --uwb-anchor-port /dev/cu.usbmodemVVVV \\
        --uwb-group-id 1 --uwb-preamble-code 9 --uwb-channel 5 \\
        --rfid-port /dev/cu.usbserial-WWWW \\
        --gesture pull,push,clockwise,anti_clockwise \\
        --trials 5 --duration 4

Single-sensor baseline (mmWave only):

    python collect_gesture_dataset.py \\
        --collector student01 --mmwave-port /dev/cu.usbserial-XXXX \\
        --trials 5 --duration 4
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from gestures import GESTURES, normalize_gestures
from mmwave.mmwave_stream import MmwaveStream
from sensors.common import append_manifest, now_text, safe_label, timestamp, write_json
from sensors.serial_json_stream import SerialLineStream
from uwb.uwb_stream import UwbStream

DEFAULT_MMWAVE_CFG = Path("mmwave/xwrL64xx-evm/near_field_hand_50cm.cfg")

MANIFEST_FIELDNAMES = [
    "dataset_name",
    "collector",
    "gesture",
    "input_type",
    "trial_index",
    "attempt_index",
    "duration_s",
    "capture_duration_s",
    "sensors_enabled",
    "mmwave_frame_count",
    "mmwave_mean_frame_rate_hz",
    "imu_line_count",
    "uwb_sample_count",
    "uwb_ok_sample_count",
    "rfid_line_count",
    "npz_path",
    "session_dir",
    "started_at",
    "finished_at",
]

INPUT_TYPE = "cosmos_multi_sensor_gesture"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect synchronized multi-sensor gesture recordings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--collector", required=True, help="Student/collector ID.")
    parser.add_argument(
        "--gesture",
        action="append",
        help=(
            "Gesture label (see gestures.py). Can be repeated or comma separated. "
            "Default: all 15 gestures."
        ),
    )
    parser.add_argument("--trials", type=int, default=5, help="Trials to collect per gesture.")
    parser.add_argument(
        "--duration", type=float, default=4.0, help="Seconds of sensor data captured per trial."
    )
    parser.add_argument("--dataset-name", default=f"gesture_dataset_{timestamp()}")
    parser.add_argument("--out-root", default="datasets")
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

    imu_group = parser.add_argument_group("IMU (ESP32 Core2)")
    imu_group.add_argument("--imu-port", help="IMU serial port.")
    imu_group.add_argument("--imu-baud", type=int, default=115200)

    uwb_group = parser.add_argument_group("UWB (Qorvo DWM3001CDK FiRa TWR: tag + anchor(s))")
    uwb_group.add_argument("--uwb-tag-port", help="UWB tag (worn, FiRa controller) serial port.")
    uwb_group.add_argument(
        "--uwb-anchor-port",
        action="append",
        help=(
            "UWB anchor (fixed, FiRa controlee) serial port. Repeat for multiple anchors "
            "(e.g. --uwb-anchor-port /dev/... --uwb-anchor-port /dev/...) to range the tag "
            "against more than one anchor at once via FiRa one-to-many mode. NOTE: the "
            "one-to-many (multi-anchor) path is unverified against real DWM3001CDK hardware -- "
            "smoke-test it before relying on it for real data collection."
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
            "Slots per ranging round. Default: 6 for a single anchor (matches UWB_lab); "
            "6 * anchor count for multiple anchors (unverified heuristic -- increase this "
            "if multi-anchor ranging is unstable or drops anchors)."
        ),
    )
    uwb_group.add_argument(
        "--uwb-skip-device-reset",
        action="store_true",
        help="Skip the UCI device reset before/after ranging (use if it's already known-good).",
    )

    rfid_group = parser.add_argument_group("RFID")
    rfid_group.add_argument("--rfid-port", help="RFID reader serial port.")
    rfid_group.add_argument("--rfid-baud", type=int, default=115200)

    return parser.parse_args()


def prompt_ready(gesture_name: str, trial_index: int, trials: int, duration_s: float) -> None:
    spec = GESTURES[gesture_name]
    print()
    print("=" * 60)
    print(f"Gesture: {spec.display_name} ({gesture_name})")
    print(f"Trial {trial_index}/{trials}")
    print(f"Suggested sensors: {', '.join(spec.suggested_sensors)}")
    print(spec.instruction)
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
        self.mmwave: MmwaveStream | None = None
        self.imu: SerialLineStream | None = None
        self.uwb: UwbStream | None = None
        self.rfid: SerialLineStream | None = None

        if bool(args.uwb_tag_port) != bool(args.uwb_anchor_port):
            raise SystemExit(
                "UWB needs both --uwb-tag-port and at least one --uwb-anchor-port "
                "(it's a tag+anchor ranging setup, not a single device)."
            )
        if args.uwb_tag_port and args.uwb_group_id is None:
            raise SystemExit("--uwb-group-id is required when UWB is enabled.")

        try:
            if args.mmwave_port:
                print(f"Opening mmWave radar on {args.mmwave_port} (cfg: {args.mmwave_cfg})...")
                self.mmwave = MmwaveStream(
                    port_path=args.mmwave_port,
                    cfg_path=args.mmwave_cfg,
                    baud=args.mmwave_baud,
                    warm_reset=not args.no_mmwave_warm_reset,
                )
            if args.imu_port:
                print(f"Opening IMU on {args.imu_port}...")
                self.imu = SerialLineStream("imu", args.imu_port, args.imu_baud)
            if args.uwb_tag_port:
                print(
                    f"Opening UWB ranging (tag {args.uwb_tag_port}, "
                    f"anchors {', '.join(args.uwb_anchor_port)})..."
                )
                self.uwb = UwbStream(
                    tag_port=args.uwb_tag_port,
                    anchor_ports=args.uwb_anchor_port,
                    group_id=args.uwb_group_id,
                    log_dir=Path(args.out_root).expanduser().resolve()
                    / args.dataset_name
                    / "uwb_logs",
                    preamble_code=args.uwb_preamble_code,
                    channel=args.uwb_channel,
                    fps=args.uwb_fps,
                    slot_span=args.uwb_slot_span,
                    slots_per_rr=args.uwb_slots_per_rr,
                    reset_devices_first=not args.uwb_skip_device_reset,
                )
            if args.rfid_port:
                print(f"Opening RFID on {args.rfid_port}...")
                self.rfid = SerialLineStream("rfid", args.rfid_port, args.rfid_baud)
        except Exception:
            self.close()
            raise

        if not self.enabled_names():
            self.close()
            raise SystemExit(
                "No sensors enabled. Pass at least one of --mmwave-port, --imu-port, "
                "--uwb-tag-port/--uwb-anchor-port, --rfid-port."
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
        for stream in (self.mmwave, self.imu, self.uwb, self.rfid):
            if stream is not None:
                stream.check_error()

    def status_line(self) -> str:
        parts = []
        if self.mmwave is not None:
            parts.append(f"mmwave={self.mmwave.frame_count}f")
        if self.imu is not None:
            parts.append(f"imu={self.imu.line_count}L")
        if self.uwb is not None:
            parts.append(f"uwb={self.uwb.sample_count}samples")
        if self.rfid is not None:
            parts.append(f"rfid={self.rfid.line_count}L")
        return " ".join(parts)

    def close(self) -> None:
        for stream in (self.mmwave, self.imu, self.uwb, self.rfid):
            if stream is not None:
                try:
                    stream.close()
                except Exception as error:  # noqa: BLE001 - best-effort cleanup
                    print(f"Warning: error while closing sensor: {error}")


def capture_trial(sensors: SensorSet, duration_s: float) -> tuple[float, float]:
    """Wait out the recording window, printing live sensor counters."""
    start = time.monotonic()
    next_status = start
    while True:
        elapsed = time.monotonic() - start
        if elapsed >= duration_s:
            break
        sensors.check_errors()
        now = time.monotonic()
        if now >= next_status:
            remaining = max(0.0, duration_s - elapsed)
            print(
                f"\rRecording: {elapsed:4.1f}/{duration_s:.1f} s, "
                f"{sensors.status_line()}, {remaining:4.1f} s left",
                end="",
                flush=True,
            )
            next_status = now + 0.2
        time.sleep(0.02)
    print()
    return start, time.monotonic()


def trial_ok(sensors: SensorSet, args: argparse.Namespace, mmwave_data, imu_lines, uwb_window, rfid_lines) -> str | None:
    """Return None if the trial meets minimum-sample thresholds, else a reason string."""
    if sensors.mmwave is not None and len(mmwave_data["frame_number"]) < args.min_mmwave_frames:
        return f"only {len(mmwave_data['frame_number'])} mmWave frames (need {args.min_mmwave_frames})"
    if sensors.imu is not None and len(imu_lines) < args.min_sensor_lines:
        return f"only {len(imu_lines)} IMU lines (need {args.min_sensor_lines})"
    if sensors.uwb is not None:
        ok_count = int(np.count_nonzero(uwb_window["status"] == "Ok"))
        if ok_count < args.min_uwb_samples:
            return f"only {ok_count} Ok UWB range samples (need {args.min_uwb_samples})"
    if sensors.rfid is not None and len(rfid_lines) < args.min_sensor_lines:
        return f"only {len(rfid_lines)} RFID lines (need {args.min_sensor_lines})"
    return None


def save_trial(
    dataset_dir: Path,
    args: argparse.Namespace,
    sensors: SensorSet,
    gesture: str,
    trial_index: int,
    attempt_index: int,
    trial_start: float,
    trial_end: float,
    mmwave_data,
    imu_lines: list[tuple[float, str]],
    uwb_window: dict,
    rfid_lines: list[tuple[float, str]],
    started_at: str,
    finished_at: str,
) -> dict:
    session_name = f"gesture_{safe_label(args.collector)}_{safe_label(gesture)}_{trial_index:03d}"
    session_dir = dataset_dir / "sessions" / session_name
    session_dir.mkdir(parents=True, exist_ok=True)
    npz_path = session_dir / "trial_data.npz"

    capture_duration_s = max(trial_end - trial_start, 1e-9)
    payload: dict[str, np.ndarray] = {
        "dataset_name": np.array(args.dataset_name),
        "collector": np.array(args.collector),
        "gesture": np.array(gesture),
        "input_type": np.array(INPUT_TYPE),
        "created_unix_s": np.array(time.time()),
        "duration_s": np.array(args.duration),
        "capture_duration_s": np.array(capture_duration_s),
        "sensors_enabled": np.array(",".join(sensors.enabled_names())),
    }

    mmwave_frame_count = 0
    mmwave_mean_fps = 0.0
    if sensors.mmwave is not None:
        mmwave_frame_count = len(mmwave_data["frame_number"])
        mmwave_mean_fps = mmwave_frame_count / capture_duration_s
        payload.update(
            {
                "mmwave_frame_number": mmwave_data["frame_number"],
                "mmwave_time_s": mmwave_data["time_s"],
                "mmwave_range_profile": mmwave_data["range_profile"],
                "mmwave_point_count": mmwave_data["point_count"],
                "mmwave_points_xyz": mmwave_data["points_xyz"],
                "mmwave_points_velocity": mmwave_data["points_velocity"],
                "mmwave_cfg_path": np.array(str(args.mmwave_cfg)),
            }
        )
        if sensors.mmwave.range_config is not None:
            payload["mmwave_range_bin_spacing_m"] = np.array(
                sensors.mmwave.range_config.bin_spacing_m
            )

    if sensors.imu is not None:
        payload["imu_recv_time_s"] = np.array([t for t, _ in imu_lines], dtype=float)
        payload["imu_raw_lines"] = np.array([line for _, line in imu_lines], dtype=object)
    uwb_ok_count = 0
    if sensors.uwb is not None:
        uwb_ok_count = int(np.count_nonzero(uwb_window["status"] == "Ok"))
        payload.update(
            {
                "uwb_time_s": uwb_window["time_s"],
                "uwb_sequence": uwb_window["sequence"],
                "uwb_mac_address": uwb_window["mac_address"],
                "uwb_status": uwb_window["status"],
                "uwb_distance_cm": uwb_window["distance_cm"],
            }
        )
    if sensors.rfid is not None:
        payload["rfid_recv_time_s"] = np.array([t for t, _ in rfid_lines], dtype=float)
        payload["rfid_raw_lines"] = np.array([line for _, line in rfid_lines], dtype=object)

    np.savez_compressed(npz_path, **payload)

    metadata = {
        "dataset_name": args.dataset_name,
        "collector": args.collector,
        "gesture": gesture,
        "input_type": INPUT_TYPE,
        "trial_index": trial_index,
        "attempt_index": attempt_index,
        "duration_s": args.duration,
        "capture_duration_s": capture_duration_s,
        "sensors_enabled": ",".join(sensors.enabled_names()),
        "mmwave_frame_count": mmwave_frame_count,
        "mmwave_mean_frame_rate_hz": mmwave_mean_fps,
        "imu_line_count": len(imu_lines),
        "uwb_sample_count": int(len(uwb_window["time_s"])) if sensors.uwb is not None else 0,
        "uwb_ok_sample_count": uwb_ok_count,
        "rfid_line_count": len(rfid_lines),
        "npz_path": str(npz_path),
        "session_dir": str(session_dir),
        "started_at": started_at,
        "finished_at": finished_at,
    }
    write_json(session_dir / "trial_metadata.json", metadata)
    return metadata


def make_dataset_metadata(args: argparse.Namespace, gesture_list: list[str], sensors: SensorSet) -> dict:
    return {
        "dataset_name": args.dataset_name,
        "collector": args.collector,
        "gestures": gesture_list,
        "input_type": INPUT_TYPE,
        "trials_per_gesture": args.trials,
        "duration_s": args.duration,
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
            "uwb_tag": args.uwb_tag_port,
            "uwb_anchors": args.uwb_anchor_port,
            "rfid": args.rfid_port,
        },
        "uwb_config": (
            {
                "group_id": args.uwb_group_id,
                "preamble_code": args.uwb_preamble_code,
                "channel": args.uwb_channel,
                "fps": args.uwb_fps,
                "ranging_span_ms": sensors.uwb.ranging_span_ms,
                "slot_span": args.uwb_slot_span,
                "slots_per_rr": sensors.uwb.slots_per_rr,
                "multi_anchor": sensors.uwb.multi_anchor,
            }
            if sensors.uwb is not None
            else None
        ),
        "created_at": now_text(),
    }


def main() -> int:
    args = parse_args()
    gesture_list = normalize_gestures(args.gesture)
    if args.trials < 1:
        raise SystemExit("--trials must be at least 1.")
    if args.duration <= 0:
        raise SystemExit("--duration must be positive.")

    dataset_dir = Path(args.out_root).expanduser().resolve() / args.dataset_name
    manifest_path = dataset_dir / "trials.csv"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    sensors = SensorSet(args)

    write_json(dataset_dir / "dataset_metadata.json", make_dataset_metadata(args, gesture_list, sensors))

    print(f"Dataset folder: {dataset_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Collector: {args.collector}")
    print(f"Gestures: {', '.join(gesture_list)}")
    print(f"Sensors enabled: {', '.join(sensors.enabled_names())}")

    interrupted = False
    accepted_total = 0
    try:
        for gesture in gesture_list:
            trial_index = 1
            while trial_index <= args.trials:
                attempt_index = 1
                while True:
                    prompt_ready(gesture, trial_index, args.trials, args.duration)
                    started_at = now_text()
                    trial_start, trial_end = capture_trial(sensors, args.duration)
                    finished_at = now_text()

                    mmwave_data = sensors.mmwave.window(trial_start, trial_end) if sensors.mmwave else None
                    imu_lines = sensors.imu.window(trial_start, trial_end) if sensors.imu else []
                    uwb_window = sensors.uwb.window(trial_start, trial_end) if sensors.uwb else {}
                    rfid_lines = sensors.rfid.window(trial_start, trial_end) if sensors.rfid else []

                    reject_reason = trial_ok(sensors, args, mmwave_data, imu_lines, uwb_window, rfid_lines)
                    if reject_reason is not None:
                        print(f"Discarded: {reject_reason}.")
                        attempt_index += 1
                        continue

                    print(f"Captured trial: {sensors.status_line()}")
                    keep = args.auto_accept or prompt_keep_recording()
                    if keep:
                        row = save_trial(
                            dataset_dir,
                            args,
                            sensors,
                            gesture,
                            trial_index,
                            attempt_index,
                            trial_start,
                            trial_end,
                            mmwave_data,
                            imu_lines,
                            uwb_window,
                            rfid_lines,
                            started_at,
                            finished_at,
                        )
                        append_manifest(manifest_path, row, MANIFEST_FIELDNAMES)
                        accepted_total += 1
                        print(f"Saved {gesture} trial {trial_index}/{args.trials}: {row['session_dir']}")
                        trial_index += 1
                        break

                    print("Discarded recording. Redoing the same trial.")
                    attempt_index += 1

    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted. Stopping sensors...")
    except RuntimeError as error:
        interrupted = True
        print(f"\nCapture failed: {error}")
    finally:
        sensors.close()

    print(f"Accepted trials: {accepted_total}")
    print(f"Dataset complete: {dataset_dir}")
    return 1 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
