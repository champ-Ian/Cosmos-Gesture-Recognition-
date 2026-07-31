#!/usr/bin/env python3
"""
Live/near-real-time gesture evaluation using a model trained by `train.py`.

Modeled on `UWB_lab/eval_realtime.py`: opens whichever sensor streams the
loaded model actually needs (`payload["sensors"]`).

`--capture-mode trigger` (recommended) runs a proper movement-based
segmentation pipeline instead of predicting on an arbitrary fixed-rate
sliding window:

    continuous sensor data -> movement detector -> confirm gesture start ->
    collect the complete gesture -> confirm gesture end -> preprocess ->
    classifier -> display -> re-arm -> back to no-gesture

implemented as an explicit state machine in `run_trigger_state_machine()`:

    IDLE -> START_CONFIRMATION -> COLLECTING -> END_CONFIRMATION ->
    CLASSIFYING -> REARM -> IDLE

See that function's docstring for what each state does. The key ideas: a
single noisy detector tick can't start or end a capture on its own (that
needs --start-confirm-seconds / --end-confirm-seconds of SUSTAINED agreement
first); a --pre-buffer-seconds back-dates the captured window's start so the
very beginning of the gesture -- which happens before the detector can
confirm it -- isn't lost; and once a real gesture is confirmed (long enough
to clear --min-gesture-seconds, not just noise), the classified window is a
full --window-seconds clip from the confirmed start, matching the FIXED
duration every training trial was collected at (see collect.py/
extract_features.py) -- NOT trimmed down to just the confirmed motion span,
since real training trials include natural pre/post-motion stillness within
their labeled duration too, and a motion-trimmed capture -- however tidy --
doesn't resemble that.

`--capture-mode interval` is the older, simpler alternative: predicts every
`--step-seconds` on a fixed-length sliding window regardless of what's
happening, so windows constantly straddle real gesture boundaries. Kept
around as a baseline to compare against, not recommended for actual live use.

Two different "nothing to report" outcomes, and they mean different things:
`no_gesture` = the detector never confirmed movement at all (IDLE/REARM), or
the classifier's own prediction was the trained "resting" class.
`unknown_gesture` = movement WAS confirmed (a real capture happened), but
--confidence-threshold or --temporal-confirm-windows/--temporal-confirm-agreement
couldn't get a reliable, consistent answer for which of the 15 trained
gestures it was. Collapsing these into one label would hide the difference
between "you didn't do anything" and "you did something, we're just not sure
what" -- worth keeping separate rather than convenient to merge.

Predictions are printed live and saved to
`sessions/eval_<name>/realtime_predictions.csv` (including a `gate_reason`
column: `none`/`confidence`/`not_confirmed`, showing which mechanism, if any,
overrode the raw prediction).

Only pass the `--*-port` flags for sensors your model actually uses --
`train.py --sensors` records which ones that is, and this script tells you
plainly if a required port is missing.

Also handles models from `train_cnn_raw.py` (`payload["classifier"] ==
"cnn_raw"`) -- those need resampled raw per-channel sequences, not the flat
summary-stat vector every other model here uses, so window extraction and
prediction both branch on that instead of assuming one shape everywhere.

Example (run from the repo root; a model trained on mmWave + IMU):

    python src/realtime_demo.py \\
      --model models/knn_early_mmwave-imu_20260101_120000.joblib \\
      --mmwave-port /dev/cu.usbserial-XXXX \\
      --imu-port /dev/cu.usbserial-YYYY \\
      --duration 60 --window-seconds 3 --step-seconds 0.5 --vote-window 5 \\
      --confidence-threshold 0.5
"""
from __future__ import annotations

import argparse
import csv
import json
import queue
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from extract_features import (
    RAW_SEQUENCE_STEPS,
    extract_sensor_features,
    extract_sensor_raw_sequence,
    feature_names_for_sensor,
    parse_imu_line,
    parse_rfid_line,
    raw_feature_names_for_sensor,
)
from sensors.common import REPO_DIR, timestamp
from sensors.imu_reader import ImuReader
from sensors.mmwave_reader import MmwaveReader
from sensors.rfid_reader import RfidReader
from sensors.uwb_reader import UwbReader
from train_cnn_raw import RAW_COLUMN_RE

# Resolved relative to this file (src/), not the current working directory --
# see collect.py's DEFAULT_MMWAVE_CFG for why.
DEFAULT_MMWAVE_CFG = Path(__file__).resolve().parent / "mmwave" / "xwrL64xx-evm" / "near_field_hand_50cm.cfg"

# "No movement happened" -- the detector never confirmed a gesture at all
# (IDLE/REARM), or the classifier itself predicted the trained "resting" class.
NO_GESTURE_LABEL = "no_gesture"
# "Movement was confirmed, but the classifier isn't confident/consistent about
# which of the 15 real gestures it was." Different meaning from NO_GESTURE_LABEL
# on purpose -- see --confidence-threshold's help text.
UNKNOWN_GESTURE_LABEL = "unknown_gesture"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Model .joblib from train.py.")
    parser.add_argument("--duration", type=float, default=60.0, help="Seconds to run the live session.")
    parser.add_argument("--window-seconds", type=float, default=3.0, help="Sliding feature-extraction window "
        "(interval mode). In --capture-mode trigger, this is BOTH a hard MAX duration cap on one COLLECTING "
        "episode AND the fixed duration every confirmed (non-noise) gesture gets classified at, matching how "
        "training trials were recorded -- not trimmed down to just the confirmed motion span. Keep this close "
        "to your training trial duration (see collect.py's --duration / --segment-length).")
    parser.add_argument("--step-seconds", type=float, default=0.5, help="Seconds between predictions (interval mode).")
    parser.add_argument(
        "--vote-window",
        type=int,
        default=1,
        help="Majority-vote over this many recent raw predictions. 1 = show raw predictions.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help=(
            "Predictions with confidence below this are reported as 'unknown_gesture' instead of the raw "
            "guess -- movement was already confirmed by the detector, the classifier just isn't sure which "
            "gesture it was, which is different from no movement having happened at all (see 'no_gesture' vs "
            "'unknown_gesture' in the module docstring). Set to 0 to disable (always show the raw guess)."
        ),
    )
    parser.add_argument(
        "--class-confidence-threshold",
        action="append",
        default=[],
        metavar="LABEL=VALUE",
        help=(
            "Per-gesture override of --confidence-threshold, e.g. --class-confidence-threshold "
            "one_arm_boxing=0.2 --class-confidence-threshold t_arm=0.2. Repeatable. A raw prediction "
            "is only gated to 'unknown_gesture' if its confidence is below THAT label's threshold (falls "
            "back to --confidence-threshold for any label not listed) -- lets a gesture that's "
            "reliably correct but chronically low-confidence clear the bar without loosening the "
            "threshold for every other gesture too."
        ),
    )
    parser.add_argument(
        "--capture-mode",
        choices=["interval", "trigger"],
        default="interval",
        help=(
            "'interval' predicts every --step-seconds regardless of what's happening -- windows constantly "
            "straddle real gesture boundaries. 'trigger' (recommended) instead runs the full movement-based "
            "segmentation state machine -- see the module docstring -- so each captured window is properly "
            "gesture-bounded like training data was."
        ),
    )
    parser.add_argument(
        "--calibration-seconds",
        type=float,
        default=2.0,
        help="[trigger mode] Idle period at startup used to measure each sensor's resting activity level. Stay still.",
    )
    parser.add_argument(
        "--trigger-window-seconds",
        type=float,
        default=0.4,
        help="[trigger mode] Short lookback window used to compute the live activity signal each poll tick.",
    )
    parser.add_argument(
        "--trigger-check-interval",
        type=float,
        default=0.1,
        help="[trigger mode] Seconds between activity-signal checks (the state machine's poll rate).",
    )
    parser.add_argument(
        "--trigger-z-threshold",
        type=float,
        default=3.0,
        help="[trigger mode] The movement detector: any sensor's activity this many std-devs above its idle "
        "baseline counts as 'Gesture' for one poll tick. This is a per-tick binary decision, not a smoothed "
        "probability -- START_CONFIRMATION/END_CONFIRMATION are what make state transitions robust to it.",
    )
    parser.add_argument(
        "--pre-buffer-seconds",
        type=float,
        default=0.3,
        help="[trigger mode] When a gesture start is confirmed, back-date the captured window's start by this "
        "much -- recovers the very beginning of the gesture, which happened before the detector had enough "
        "sustained signal to confirm it (see 'preserve the true gesture boundaries' in the module docstring). "
        "Free to do: sensor readers keep their full session history, nothing needs to be pre-recorded.",
    )
    parser.add_argument(
        "--start-confirm-seconds",
        type=float,
        default=0.3,
        help="[trigger mode] IDLE -> COLLECTING only happens after the detector says 'Gesture' for this many "
        "CONSECUTIVE seconds (START_CONFIRMATION) -- a single noisy tick isn't enough to start a capture.",
    )
    parser.add_argument(
        "--end-confirm-seconds",
        type=float,
        default=0.8,
        help="[trigger mode] COLLECTING -> CLASSIFYING only happens after the detector says 'No Gesture' for "
        "this many CONSECUTIVE seconds (END_CONFIRMATION). If activity resumes before that, it's treated as a "
        "brief pause mid-gesture, not a real end, and collection continues rather than splitting into two "
        "captures. Decides WHEN a gesture is considered over (and, via the confirmed motion duration, whether "
        "--min-gesture-seconds treats it as noise) -- it does NOT determine how much data gets classified; "
        "see --window-seconds for that. Kept fairly generous (0.8s, up from an earlier 0.5s) so a brief "
        "mid-gesture lull doesn't get mistaken for the real end.",
    )
    parser.add_argument(
        "--rearm-seconds",
        type=float,
        default=0.5,
        help="[trigger mode] After displaying a result, require this many CONSECUTIVE seconds of confirmed "
        "'No Gesture' before returning to IDLE and allowing a new capture -- prevents residual settling "
        "motion right after a gesture from immediately triggering a second, spurious capture.",
    )
    parser.add_argument(
        "--min-gesture-seconds",
        type=float,
        default=1.0,
        help="[trigger mode] Checked against the ACTUAL confirmed motion duration (gesture_start_time to the "
        "instant 'No Gesture' was first seen in END_CONFIRMATION) -- shorter than this is discarded as noise "
        "before ever being classified. NOTE: confirmed motion duration can never be less than "
        "--start-confirm-seconds + --pre-buffer-seconds (anything shorter never survives START_CONFIRMATION "
        "in the first place) -- keep this comfortably ABOVE that sum, or it can never actually discard "
        "anything. Does not limit how much data gets classified for a real gesture; see "
        "--window-seconds for that (every classified capture is a full --window-seconds clip, matching the "
        "fixed duration training trials were recorded at). Raised from an earlier 0.3s: with "
        "--start-confirm-seconds + --pre-buffer-seconds alone already contributing ~0.6s, 0.3s was barely "
        "filtering anything.",
    )
    parser.add_argument(
        "--temporal-confirm-windows",
        type=int,
        default=1,
        help="[trigger mode] Re-classify this many trailing sub-windows of the same capture (in addition to "
        "the full captured window) and require --temporal-confirm-agreement of them to agree with the main "
        "prediction before confirming it -- catches a result that only looks right when you classify the "
        "whole blob, but flips depending on exactly how much of the capture you look at. 1 = disabled "
        "(classify once, like before).",
    )
    parser.add_argument(
        "--temporal-confirm-agreement",
        type=float,
        default=0.67,
        help="[trigger mode] Fraction of --temporal-confirm-windows sub-classifications (plus the main one) "
        "that must agree with the main prediction for it to be confirmed. Below this, forced to "
        "'unknown_gesture' (movement was real, the classifier just isn't consistent about which gesture).",
    )
    parser.add_argument("--out-root", default=str(REPO_DIR / "sessions"))
    parser.add_argument("--session-name")
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Show a live Tkinter window (big color-coded prediction, confidence bar, "
        "recent-prediction history) instead of terminal-only output.",
    )
    parser.add_argument(
        "--binary-display",
        action="store_true",
        help="Testing aid: collapse the displayed/GUI prediction to just 'gesture' or 'no_gesture' "
        "instead of the specific class name, to isolate whether the no-movement gates work at all, "
        "separate from whether the specific gesture guess is correct. The real predicted label is "
        "still logged in full to the CSV and shown in the bracketed raw-guess debug text -- this only "
        "simplifies the headline display. Not meant to be left on permanently.",
    )

    mmwave_group = parser.add_argument_group("mmWave radar (TI xWRL6432)")
    mmwave_group.add_argument("--mmwave-port")
    mmwave_group.add_argument("--mmwave-cfg", type=Path, default=DEFAULT_MMWAVE_CFG)
    mmwave_group.add_argument("--mmwave-baud", type=int, default=115200)
    mmwave_group.add_argument("--no-mmwave-warm-reset", action="store_true")

    imu_group = parser.add_argument_group("IMU (ESP32 Core2)")
    imu_group.add_argument("--imu-port")
    imu_group.add_argument("--imu-baud", type=int, default=115200)

    uwb_group = parser.add_argument_group(
        "UWB (Qorvo DWM3001CDK FiRa TWR: 2 boards, one worn per wrist -- wrist-to-wrist "
        "distance, no fixed anchor)"
    )
    uwb_group.add_argument("--uwb-anchor-port")
    uwb_group.add_argument("--uwb-node-port", action="append")
    uwb_group.add_argument("--uwb-group-id", type=int)
    uwb_group.add_argument("--uwb-preamble-code", type=int, default=10)
    uwb_group.add_argument("--uwb-channel", type=int, choices=[5, 9], default=9)
    uwb_group.add_argument("--uwb-fps", type=float, default=50.0)
    uwb_group.add_argument("--uwb-slot-span", type=int, default=2400)
    uwb_group.add_argument("--uwb-slots-per-rr", type=int, default=None)
    uwb_group.add_argument("--uwb-skip-device-reset", action="store_true")

    rfid_group = parser.add_argument_group("RFID (RFID_Lab reader, TCP -- not serial)")
    rfid_group.add_argument("--rfid-host", default="192.168.137.1")
    rfid_group.add_argument("--rfid-tcp-port", type=int, default=9055)

    args = parser.parse_args()
    args.class_confidence_thresholds = {}
    for item in args.class_confidence_threshold:
        for part in item.split(","):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                raise SystemExit(f"--class-confidence-threshold expects LABEL=VALUE, got: {part!r}")
            label, value = part.split("=", 1)
            args.class_confidence_thresholds[label.strip()] = float(value)
    return args


def majority_vote(predictions) -> tuple[str, float, dict]:
    """Majority-vote over recent raw predictions; ties break toward the most recent label."""
    counts: dict[str, int] = {}
    for prediction in predictions:
        counts[prediction] = counts.get(prediction, 0) + 1
    max_count = max(counts.values())
    tied = {label for label, count in counts.items() if count == max_count}
    winner = next(label for label in reversed(predictions) if label in tied)
    return winner, max_count / len(predictions), counts


def open_streams(args: argparse.Namespace, required_sensors: list[str], session_dir: Path) -> dict:
    streams: dict = {}
    try:
        if "mmwave" in required_sensors:
            if not args.mmwave_port:
                raise SystemExit("This model needs mmWave -- pass --mmwave-port.")
            print(f"Opening mmWave radar on {args.mmwave_port} (cfg: {args.mmwave_cfg})...")
            streams["mmwave"] = MmwaveReader(
                port_path=args.mmwave_port,
                cfg_path=args.mmwave_cfg,
                baud=args.mmwave_baud,
                warm_reset=not args.no_mmwave_warm_reset,
            )
        if "imu" in required_sensors:
            if not args.imu_port:
                raise SystemExit("This model needs IMU -- pass --imu-port.")
            print(f"Opening IMU on {args.imu_port}...")
            streams["imu"] = ImuReader("imu", args.imu_port, args.imu_baud)
        if "uwb" in required_sensors:
            if not args.uwb_anchor_port or not args.uwb_node_port:
                raise SystemExit("This model needs UWB -- pass --uwb-anchor-port and --uwb-node-port.")
            if args.uwb_group_id is None:
                raise SystemExit("--uwb-group-id is required when UWB is enabled.")
            print(f"Opening UWB (anchor {args.uwb_anchor_port}, nodes {', '.join(args.uwb_node_port)})...")
            streams["uwb"] = UwbReader(
                anchor_port=args.uwb_anchor_port,
                node_ports=args.uwb_node_port,
                group_id=args.uwb_group_id,
                log_dir=session_dir / "uwb_logs",
                preamble_code=args.uwb_preamble_code,
                channel=args.uwb_channel,
                fps=args.uwb_fps,
                slot_span=args.uwb_slot_span,
                slots_per_rr=args.uwb_slots_per_rr,
                reset_devices_first=not args.uwb_skip_device_reset,
            )
        if "rfid" in required_sensors:
            print(f"Opening RFID reader at {args.rfid_host}:{args.rfid_tcp_port}...")
            streams["rfid"] = RfidReader(args.rfid_host, args.rfid_tcp_port)
    except Exception:
        close_streams(streams)
        raise
    return streams


def close_streams(streams: dict) -> None:
    for stream in streams.values():
        try:
            stream.close()
        except Exception as error:  # noqa: BLE001 - best-effort cleanup
            print(f"Warning: error while closing sensor: {error}")


def sensor_activity_score(sensor: str, window) -> float | None:
    """Cheap 'how much motion is in this window' number for trigger-mode --
    NOT a feature vector, just a scalar used to decide whether to bother
    capturing+classifying at all. Returns None if there's not enough data
    in the window to say anything (treated as "no activity detected")."""
    if sensor == "mmwave":
        if len(window["frame_number"]) < 2:
            return None
        range_profile = window["range_profile"]
        energy = np.nansum(range_profile, axis=1) if range_profile.size else np.zeros(len(window["frame_number"]))
        return float(np.std(energy))
    if sensor == "uwb":
        ok = window["status"] == "Ok"
        values = window["distance_cm"][ok]
        if len(values) < 2:
            return None
        return float(np.std(values))
    if sensor == "imu":
        samples = [parse_imu_line(line) for _, line in window]
        samples = [s for s in samples if s is not None]
        if len(samples) < 2:
            return None
        accel_mag = np.linalg.norm(np.array(samples)[:, :3], axis=1)
        return float(np.std(accel_mag))
    if sensor == "rfid":
        rssi = [parse_rfid_line(line)[1] for _, line in window if parse_rfid_line(line) is not None]
        if len(rssi) < 2:
            return None
        return float(np.std(rssi))
    raise ValueError(f"Unknown sensor: {sensor}")


def window_to_feature_input(sensor: str, window) -> dict:
    """Adapt a live stream's `.window()` output to the `{sensor}_*` keys extract_features.py expects."""
    if sensor == "mmwave":
        return {
            "mmwave_frame_number": window["frame_number"],
            "mmwave_range_profile": window["range_profile"],
            "mmwave_point_count": window["point_count"],
            "mmwave_points_velocity": window["points_velocity"],
        }
    if sensor == "uwb":
        return {"uwb_status": window["status"], "uwb_distance_cm": window["distance_cm"]}
    if sensor in ("imu", "rfid"):
        return {f"{sensor}_raw_lines": np.array([line for _, line in window], dtype=object)}
    raise ValueError(f"Unknown sensor: {sensor}")


def window_to_raw_input(sensor: str, window) -> dict:
    """Same idea as `window_to_feature_input`, but for `extract_sensor_raw_sequence`
    (train_cnn_raw.py's raw-resampled-sequence path) instead of `extract_sensor_features`.
    That extractor needs per-sample *timestamps* too (it resamples onto a fixed grid),
    which the scalar path never needed -- every reader's `.window()` already returns
    them, `window_to_feature_input` just didn't pass them through."""
    if sensor == "mmwave":
        return {
            "mmwave_frame_number": window["frame_number"],
            "mmwave_time_s": window["time_s"],
            "mmwave_range_profile": window["range_profile"],
            "mmwave_point_count": window["point_count"],
        }
    if sensor == "uwb":
        return {
            "uwb_status": window["status"],
            "uwb_distance_cm": window["distance_cm"],
            "uwb_time_s": window["time_s"],
        }
    if sensor in ("imu", "rfid"):
        return {
            f"{sensor}_raw_lines": np.array([line for _, line in window], dtype=object),
            f"{sensor}_recv_time_s": np.array([t for t, _ in window], dtype=float),
        }
    raise ValueError(f"Unknown sensor: {sensor}")


def raw_channel_sequences(sensor: str, raw_input: dict) -> dict[str, list[float]]:
    """Extract one sensor's raw resampled sequence and regroup it by channel key
    (e.g. 'imu_ax', matching `train_cnn_raw.py`'s `channel_names`) instead of the
    flat per-step names `raw_feature_names_for_sensor` returns (e.g. 'imu_raw_ax_0').
    Returns {} if this sensor's raw sequence can't be extracted from this window."""
    values = extract_sensor_raw_sequence(sensor, raw_input)
    if values is None:
        return {}
    names = raw_feature_names_for_sensor(sensor)
    channels: dict[str, list[float]] = {}
    for name, value in zip(names, values):
        match = RAW_COLUMN_RE.match(name)
        if not match:
            continue
        sensor_name, field, step = match.group(1), match.group(2), int(match.group(3))
        channel_key = f"{sensor_name}_{field}"
        channels.setdefault(channel_key, [0.0] * RAW_SEQUENCE_STEPS)[step] = value
    return channels


def predict_raw(model, channel_names: list[str], scalar_names: list[str], channel_seqs: dict, scalar_values: dict) -> tuple[str, float | None]:
    """`predict()`'s equivalent for `TorchRawCNNClassifier` models -- these take
    `(X_raw, X_scalar)` as two separate structured arrays instead of one flat
    vector, so they can't share `predict()`/`prediction_confidence()` above."""
    X_raw = np.array([[channel_seqs[channel] for channel in channel_names]], dtype=np.float32)
    X_scalar = np.array([[scalar_values[name] for name in scalar_names]], dtype=np.float32)
    prediction = model.predict(X_raw, X_scalar)[0]
    proba = model.predict_proba(X_raw, X_scalar)[0]
    classes = list(model.classes_)
    confidence = float(proba[classes.index(prediction)]) if prediction in classes else float(max(proba))
    return prediction, confidence


def prediction_confidence(model, sensors: list[str], fusion: str, per_sensor_vectors: dict, prediction: str) -> float | None:
    if fusion == "early":
        vector = []
        for sensor in sensors:
            vector.extend(per_sensor_vectors[sensor])
        if not hasattr(model, "predict_proba"):
            return None
        proba = model.predict_proba([vector])[0]
        classes = list(getattr(model, "classes_", []))
        if not classes and hasattr(model, "steps"):
            classes = list(model.steps[-1][1].classes_)
        return float(proba[classes.index(prediction)]) if prediction in classes else float(max(proba))

    proba = model.predict_proba(per_sensor_vectors)[0]
    classes = list(model.classes_)
    return float(proba[classes.index(prediction)]) if prediction in classes else float(max(proba))


def predict(model, sensors: list[str], fusion: str, per_sensor_vectors: dict) -> tuple[str, float | None]:
    if fusion == "early":
        vector = []
        for sensor in sensors:
            vector.extend(per_sensor_vectors[sensor])
        prediction = model.predict([vector])[0]
    else:
        prediction = model.predict(per_sensor_vectors)[0]
    confidence = prediction_confidence(model, sensors, fusion, per_sensor_vectors, prediction)
    return prediction, confidence


def _classify_window(
    streams: dict,
    window_start: float,
    window_end: float,
    model,
    sensors: list[str],
    fusion: str | None,
    is_raw_cnn: bool,
    channel_names: list[str] | None,
    scalar_names: list[str] | None,
) -> tuple[str, float | None] | None:
    """Extract features over [window_start, window_end] and predict -- no gating,
    no logging, no printing, just (raw_prediction, raw_confidence). Returns None
    if a sensor's data in this window isn't usable (e.g. too few samples).

    Split out from `capture_and_classify` so trigger mode's state machine can
    silently re-probe a growing window (every check interval, while waiting to
    see if the model becomes confident) without spamming the console/CSV/GUI
    the way an official, reported prediction does."""
    ready = True

    if is_raw_cnn:
        channel_seqs: dict = {}
        scalar_values: dict = {}
        for sensor, stream in streams.items():
            window = stream.window(window_start, window_end)
            sensor_channels = raw_channel_sequences(sensor, window_to_raw_input(sensor, window))
            scalar_vector = extract_sensor_features(sensor, window_to_feature_input(sensor, window))
            if not sensor_channels or scalar_vector is None:
                ready = False
                break
            channel_seqs.update(sensor_channels)
            scalar_values.update(zip(feature_names_for_sensor(sensor), scalar_vector))
    else:
        per_sensor_vectors = {}
        for sensor, stream in streams.items():
            window = stream.window(window_start, window_end)
            vector = extract_sensor_features(sensor, window_to_feature_input(sensor, window))
            if vector is None:
                ready = False
                break
            per_sensor_vectors[sensor] = vector

    if not ready:
        return None

    if is_raw_cnn:
        return predict_raw(model, channel_names, scalar_names, channel_seqs, scalar_values)
    return predict(model, sensors, fusion, per_sensor_vectors)


def capture_and_classify(
    streams: dict,
    window_start: float,
    window_end: float,
    model,
    sensors: list[str],
    fusion: str | None,
    is_raw_cnn: bool,
    channel_names: list[str] | None,
    scalar_names: list[str] | None,
    args: argparse.Namespace,
    vote_history: deque,
    writer,
    prediction_file,
    session_start: float,
    gui_queue: "queue.Queue | None" = None,
    baseline: dict[str, tuple[float, float]] | None = None,
) -> None:
    """Extract features over [window_start, window_end], predict, gate on
    --confidence-threshold, log to CSV, and print -- shared by --capture-mode
    interval (called every --step-seconds) and the trigger state machine's
    CLASSIFYING state (called once per confirmed gesture -- by construction,
    real movement was already detected+confirmed for the whole [window_start,
    window_end) span, so there's nothing here to re-check the ACTIVITY of; the
    only thing left to be unsure about is WHICH gesture it was). Does nothing
    if a sensor's data in this window isn't usable (e.g. too few samples).

    `baseline` is currently unused here (kept in the signature for callers
    that still pass it) -- activity/settledness are now the trigger state
    machine's job (START_CONFIRMATION/END_CONFIRMATION), not a post-hoc check
    on the finished window."""
    result = _classify_window(streams, window_start, window_end, model, sensors, fusion, is_raw_cnn, channel_names, scalar_names)
    if result is None:
        return
    raw_prediction, raw_confidence = result
    if raw_prediction == "resting":
        raw_prediction = NO_GESTURE_LABEL
    effective_threshold = args.class_confidence_thresholds.get(raw_prediction, args.confidence_threshold)
    below_threshold = raw_confidence is not None and raw_confidence < effective_threshold

    not_confirmed = False
    agreeing, sub_predictions = 1, [raw_prediction]
    if args.temporal_confirm_windows > 1 and raw_prediction != NO_GESTURE_LABEL:
        total_span = window_end - window_start
        for i in range(1, args.temporal_confirm_windows):
            sub_start = window_start + total_span * (i / args.temporal_confirm_windows)
            sub_result = _classify_window(
                streams, sub_start, window_end, model, sensors, fusion, is_raw_cnn, channel_names, scalar_names
            )
            if sub_result is not None:
                sub_label = "resting" if sub_result[0] == "resting" else sub_result[0]
                sub_predictions.append(NO_GESTURE_LABEL if sub_label == "resting" else sub_label)
        agreeing = sum(1 for p in sub_predictions if p == raw_prediction)
        agreement = agreeing / len(sub_predictions)
        not_confirmed = agreement < args.temporal_confirm_agreement

    if raw_prediction == NO_GESTURE_LABEL:
        gate_reason = "none"
    elif not_confirmed:
        gate_reason = "not_confirmed"
    elif below_threshold:
        gate_reason = "confidence"
    else:
        gate_reason = "none"

    below_threshold = below_threshold or not_confirmed
    # Movement was already confirmed by the state machine before this function was ever
    # called -- a low-confidence/inconsistent guess means "not sure which gesture," not
    # "no gesture happened." NO_GESTURE_LABEL itself (the model's own "resting" class, or
    # interval mode with no confirmed movement at all) is the one case that's genuinely
    # "nothing happened," so it stays as-is rather than becoming "unknown."
    if below_threshold and raw_prediction != NO_GESTURE_LABEL:
        gated_prediction = UNKNOWN_GESTURE_LABEL
    elif below_threshold:
        gated_prediction = NO_GESTURE_LABEL
    else:
        gated_prediction = raw_prediction
    vote_history.append(gated_prediction)
    if args.vote_window <= 1:
        display_prediction, display_confidence = gated_prediction, raw_confidence
        vote_fraction, vote_counts = None, {}
    else:
        display_prediction, vote_fraction, vote_counts = majority_vote(vote_history)
        display_confidence = vote_fraction

    writer.writerow(
        {
            "time_s": f"{window_end - session_start:.3f}",
            "prediction": display_prediction,
            "confidence": "" if display_confidence is None else f"{display_confidence:.4f}",
            "raw_prediction": raw_prediction,
            "raw_confidence": "" if raw_confidence is None else f"{raw_confidence:.4f}",
            "gated_prediction": gated_prediction,
            "gate_reason": gate_reason,
            "confidence_threshold": args.confidence_threshold,
            "captured_seconds": f"{window_end - window_start:.3f}",
            "vote_fraction": "" if vote_fraction is None else f"{vote_fraction:.4f}",
            "vote_count": len(vote_history),
            "vote_window": args.vote_window,
            "vote_counts_json": json.dumps(vote_counts, sort_keys=True),
        }
    )
    prediction_file.flush()

    shown_prediction = display_prediction
    if args.binary_display and display_prediction != NO_GESTURE_LABEL:
        shown_prediction = "gesture"

    if gui_queue is not None:
        gui_queue.put(
            {
                "prediction": shown_prediction,
                "confidence": display_confidence,
                "time_s": window_end - session_start,
                "raw_prediction": raw_prediction,
                "raw_confidence": raw_confidence,
                "below_threshold": below_threshold,
                "vote_fraction": vote_fraction,
                "vote_window": args.vote_window,
            }
        )

    if args.vote_window <= 1:
        conf_text = "" if raw_confidence is None else f" ({raw_confidence:.2f})"
        if not_confirmed:
            below_text = f" [{agreeing}/{len(sub_predictions)} sub-windows agreed, raw guess: {raw_prediction}]"
        elif below_threshold:
            below_text = f" [below {effective_threshold:.2f}, raw guess: {raw_prediction}]"
        else:
            below_text = ""
        print(f"prediction: {shown_prediction}{conf_text}{below_text}", flush=True)
    else:
        vote_text = "" if vote_fraction is None else f" vote={vote_fraction:.2f}"
        raw_text = "" if raw_confidence is None else f" ({raw_confidence:.2f})"
        print(f"prediction: {shown_prediction}{vote_text} | raw: {raw_prediction}{raw_text}", flush=True)


def calibrate_idle_baseline(streams: dict, args: argparse.Namespace) -> dict[str, tuple[float, float]]:
    """[trigger mode] Sample each sensor's activity score for --calibration-seconds
    (the tester should stay still) and return {sensor: (mean, std)}. This is the
    idle baseline every later activity reading gets compared against to decide
    whether a gesture just started."""
    print(f"Calibrating idle baseline -- please stay still for {args.calibration_seconds:.1f}s...")
    samples: dict[str, list[float]] = {sensor: [] for sensor in streams}
    calibration_start = time.monotonic()
    end_time = calibration_start + args.calibration_seconds
    while time.monotonic() < end_time:
        now = time.monotonic()
        window_start = max(calibration_start, now - args.trigger_window_seconds)
        for sensor, stream in streams.items():
            score = sensor_activity_score(sensor, stream.window(window_start, now))
            if score is not None:
                samples[sensor].append(score)
        time.sleep(args.trigger_check_interval)

    baseline: dict[str, tuple[float, float]] = {}
    for sensor, values in samples.items():
        if values:
            mean = float(np.mean(values))
            std = max(float(np.std(values)), 1e-6)
        else:
            mean, std = 0.0, 1e-6
        baseline[sensor] = (mean, std)
    print("Baseline: " + ", ".join(f"{sensor}={mean:.2f}±{std:.2f}" for sensor, (mean, std) in baseline.items()))
    return baseline


def activity_triggered(streams: dict, baseline: dict[str, tuple[float, float]], now: float, args: argparse.Namespace) -> bool:
    """[trigger mode] True if any sensor's current activity is --trigger-z-threshold
    std-devs above its idle baseline -- deliberately liberal (any one sensor can
    trigger) since missed real gestures are worse than an occasional false trigger
    that gets caught by --confidence-threshold downstream instead."""
    window_start = now - args.trigger_window_seconds
    for sensor, stream in streams.items():
        score = sensor_activity_score(sensor, stream.window(window_start, now))
        if score is None:
            continue
        mean, std = baseline[sensor]
        if (score - mean) / std >= args.trigger_z_threshold:
            return True
    return False


def run_trigger_state_machine(
    streams: dict,
    args: argparse.Namespace,
    session_start: float,
    end_time: float,
    stop_event: threading.Event,
    cycle_kwargs: dict,
    gui_queue: "queue.Queue | None",
) -> None:
    """--capture-mode trigger's loop: a movement-based segmentation state machine.

    IDLE -> START_CONFIRMATION -> COLLECTING -> END_CONFIRMATION -> CLASSIFYING -> REARM -> IDLE

    IDLE: waiting, displaying 'no_gesture'. The instant the detector
    (activity_triggered(), one poll-tick binary decision) says "Gesture",
    move to START_CONFIRMATION.

    START_CONFIRMATION: the detector must keep saying "Gesture" for
    --start-confirm-seconds STRAIGHT before a capture actually starts -- one
    noisy tick drops back to IDLE instead of starting a spurious capture.

    COLLECTING: the window's start is back-dated by --pre-buffer-seconds
    (recovering the bit of the gesture that happened before
    START_CONFIRMATION finished confirming it -- free to do, since the sensor
    readers keep their whole session's history, nothing needs to be
    pre-recorded) and capture keeps extending for as long as the detector
    keeps saying "Gesture", up to a hard --window-seconds cap. The instant it
    says "No Gesture", move to END_CONFIRMATION.

    END_CONFIRMATION: the detector must keep saying "No Gesture" for
    --end-confirm-seconds STRAIGHT before the gesture is considered over -- if
    it flips back to "Gesture" first, that was a brief pause mid-gesture, not
    the end, and COLLECTING resumes instead of splitting one gesture into two
    captures. Once confirmed, the ACTUAL confirmed motion duration (from
    gesture_start_time to the instant "No Gesture" was first seen) is checked
    against --min-gesture-seconds -- too short and it's discarded as noise
    right here, without ever being classified. If it's long enough to be real,
    the window that actually gets classified is NOT trimmed to that motion
    span -- it's expanded to a full --window-seconds clip from the confirmed
    start, matching the fixed duration every training trial was recorded at
    (see collect.py/extract_features.py). Training trials include natural
    pre/post-motion stillness within their labeled duration, so a
    motion-trimmed capture, however tidy, is shorter than and doesn't
    resemble anything the model has seen -- CLASSIFYING then has to wait
    (real time, not instantly) until that full window has actually elapsed
    before there's enough buffered data to extract it.

    CLASSIFYING: waits for the full --window-seconds window (from
    gesture_start_time) to actually elapse, then runs `capture_and_classify`
    exactly once. (Noise-length captures never reach this state at all --
    see END_CONFIRMATION above.)

    REARM: after a result is shown (or a too-short capture discarded),
    require --rearm-seconds of CONTINUOUS confirmed "No Gesture" before
    allowing IDLE again -- keeps residual settling motion right after a
    gesture from immediately starting a second, spurious capture."""

    def announce_status(text: str) -> None:
        print(text, flush=True)
        if gui_queue is not None:
            gui_queue.put({"_status": text})

    def push_no_gesture(now: float) -> None:
        print(f"prediction: {NO_GESTURE_LABEL}", flush=True)
        if gui_queue is not None:
            gui_queue.put(
                {
                    "prediction": NO_GESTURE_LABEL,
                    "confidence": None,
                    "time_s": now - session_start,
                    "raw_prediction": NO_GESTURE_LABEL,
                    "raw_confidence": None,
                    "below_threshold": False,
                    "vote_fraction": None,
                    "vote_window": args.vote_window,
                }
            )

    baseline = calibrate_idle_baseline(streams, args)
    announce_status(f"Waiting for gesture onset (z >= {args.trigger_z_threshold})...")
    push_no_gesture(time.monotonic())

    state = "IDLE"
    start_candidate_time = 0.0
    gesture_start_time = 0.0
    gesture_end_time = 0.0
    end_candidate_time = 0.0
    rearm_stable_since = 0.0

    while time.monotonic() < end_time and not stop_event.is_set():
        now = time.monotonic()
        for stream in streams.values():
            stream.check_error()
        triggered = activity_triggered(streams, baseline, now, args)

        if state == "IDLE":
            if triggered:
                state = "START_CONFIRMATION"
                start_candidate_time = now

        elif state == "START_CONFIRMATION":
            if not triggered:
                state = "IDLE"
            elif now - start_candidate_time >= args.start_confirm_seconds:
                gesture_start_time = start_candidate_time - args.pre_buffer_seconds
                announce_status(f"[{start_candidate_time - session_start:.2f}s] gesture start confirmed, collecting...")
                state = "COLLECTING"

        elif state == "COLLECTING":
            if now - gesture_start_time >= args.window_seconds:
                gesture_end_time = gesture_start_time + args.window_seconds
                state = "CLASSIFYING"
            elif not triggered:
                end_candidate_time = now
                state = "END_CONFIRMATION"

        elif state == "END_CONFIRMATION":
            if triggered:
                state = "COLLECTING"  # false alarm -- just a brief pause mid-gesture
            elif now - gesture_start_time >= args.window_seconds:
                gesture_end_time = gesture_start_time + args.window_seconds
                state = "CLASSIFYING"
            elif now - end_candidate_time >= args.end_confirm_seconds:
                confirmed_motion_seconds = end_candidate_time - gesture_start_time
                if confirmed_motion_seconds < args.min_gesture_seconds:
                    # Too brief to be a real gesture attempt -- discard as noise here, based on
                    # how long motion was actually confirmed, rather than expanding it into a
                    # full window and letting the classifier guess at what's likely nothing.
                    announce_status(
                        f"[{end_candidate_time - session_start:.2f}s] discarded {confirmed_motion_seconds:.2f}s "
                        f"capture (shorter than --min-gesture-seconds {args.min_gesture_seconds:.2f}s), likely noise"
                    )
                    rearm_stable_since = 0.0
                    state = "REARM"
                else:
                    # Classify a full --window-seconds window from the confirmed start, NOT
                    # trimmed to end_candidate_time. Every training trial (see extract_features.py
                    # / collect.py) is a fixed-duration clip regardless of how fast the real
                    # motion finished within it, so a motion-trimmed capture -- however tidy --
                    # doesn't resemble anything the model actually trained on. Start/end
                    # confirmation still decide WHEN a real gesture happened (and filter noise via
                    # min_gesture_seconds above); this only changes how much data gets classified
                    # once that's decided.
                    gesture_end_time = gesture_start_time + args.window_seconds
                    state = "CLASSIFYING"

        if state == "CLASSIFYING":
            if now < gesture_end_time:
                pass  # keep waiting for the full fixed window to actually fill before classifying
            else:
                captured_seconds = gesture_end_time - gesture_start_time
                announce_status(
                    f"[{gesture_end_time - session_start:.2f}s] gesture end confirmed, "
                    f"captured {captured_seconds:.2f}s, classifying..."
                )
                capture_and_classify(window_start=gesture_start_time, window_end=gesture_end_time, baseline=baseline, **cycle_kwargs)
                rearm_stable_since = 0.0
                state = "REARM"

        elif state == "REARM":
            if triggered:
                rearm_stable_since = 0.0
            elif rearm_stable_since == 0.0:
                rearm_stable_since = now
            elif now - rearm_stable_since >= args.rearm_seconds:
                state = "IDLE"
                push_no_gesture(now)

        time.sleep(args.trigger_check_interval)


def main() -> int:
    args = parse_args()
    args.vote_window = max(1, int(args.vote_window))

    try:
        import joblib
    except ImportError as exc:
        raise SystemExit(f"Missing joblib/sklearn environment: {exc}")

    payload = joblib.load(args.model)
    model = payload["model"]
    classifier_label = payload.get("classifier_label", payload.get("classifier", "unknown"))
    labels = payload.get("labels", [])
    is_raw_cnn = payload.get("classifier") == "cnn_raw"

    if is_raw_cnn:
        channel_names = payload["channel_names"]
        scalar_names = payload["scalar_names"]
        # channel/scalar names are '{sensor}_{field}' / '{sensor}_{stat}' -- the
        # sensor is always the part before the first underscore (sensor names
        # themselves never contain one), same convention train_cnn_raw.py uses.
        sensors = sorted({name.split("_", 1)[0] for name in channel_names + scalar_names})
        fusion = None
        print(f"Loaded model: {classifier_label} (raw sequences + summary stats: {'+'.join(sensors)})")
    else:
        sensors = payload["sensors"]
        fusion = payload.get("fusion", "early")
        print(f"Loaded model: {classifier_label} ({fusion} fusion: {'+'.join(sensors)})")

    session_name = args.session_name or f"eval_{timestamp()}"
    session_dir = Path(args.out_root).expanduser().resolve() / session_name
    session_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = session_dir / "realtime_predictions.csv"

    print(f"Gesture labels: {', '.join(labels)}")
    print(f"Session folder: {session_dir}")

    streams = open_streams(args, sensors, session_dir)
    time.sleep(0.5)  # let boards start producing data

    stop_event = threading.Event()
    gui_queue: "queue.Queue | None" = queue.Queue() if args.gui else None

    def run_session() -> bool:
        """Runs the capture/predict loop until --duration elapses, a fatal
        error occurs, or `stop_event` is set (only happens if --gui's window
        gets closed early). Returns True if the session was interrupted."""
        vote_history: deque = deque(maxlen=args.vote_window)
        interrupted = False

        with open(predictions_path, "w", newline="") as prediction_file:
            writer = csv.DictWriter(
                prediction_file,
                fieldnames=[
                    "time_s",
                    "prediction",
                    "confidence",
                    "raw_prediction",
                    "raw_confidence",
                    "gated_prediction",
                    "gate_reason",
                    "confidence_threshold",
                    "captured_seconds",
                    "vote_fraction",
                    "vote_count",
                    "vote_window",
                    "vote_counts_json",
                ],
            )
            writer.writeheader()

            session_start = time.monotonic()
            end_time = session_start + args.duration

            cycle_kwargs = dict(
                streams=streams,
                model=model,
                sensors=sensors,
                fusion=fusion,
                is_raw_cnn=is_raw_cnn,
                channel_names=channel_names if is_raw_cnn else None,
                scalar_names=scalar_names if is_raw_cnn else None,
                args=args,
                vote_history=vote_history,
                writer=writer,
                prediction_file=prediction_file,
                session_start=session_start,
                gui_queue=gui_queue,
            )

            try:
                if args.capture_mode == "trigger":
                    run_trigger_state_machine(
                        streams=streams,
                        args=args,
                        session_start=session_start,
                        end_time=end_time,
                        stop_event=stop_event,
                        cycle_kwargs=cycle_kwargs,
                        gui_queue=gui_queue,
                    )
                else:
                    last_prediction_time = 0.0
                    while time.monotonic() < end_time and not stop_event.is_set():
                        now = time.monotonic()
                        for stream in streams.values():
                            stream.check_error()

                        if now - last_prediction_time >= args.step_seconds:
                            window_start = max(session_start, now - args.window_seconds)
                            capture_and_classify(window_start=window_start, window_end=now, **cycle_kwargs)
                            last_prediction_time = now

                        time.sleep(0.02)
            except KeyboardInterrupt:
                interrupted = True
                print("\nInterrupted. Stopping sensors...")
            except RuntimeError as error:
                interrupted = True
                print(f"\nEvaluation failed: {error}")
            finally:
                close_streams(streams)
                if gui_queue is not None:
                    gui_queue.put({"_stop": True})

        return interrupted

    if args.gui:
        from realtime_gui import RealtimeGestureGUI

        result: dict = {}
        worker = threading.Thread(target=lambda: result.update(interrupted=run_session()), daemon=True)
        worker.start()
        gui = RealtimeGestureGUI(
            model_label=classifier_label,
            sensors=sensors,
            gesture_labels=labels,
            no_gesture_label=NO_GESTURE_LABEL,
            duration_s=args.duration,
            unknown_gesture_label=UNKNOWN_GESTURE_LABEL,
        )
        gui.run(gui_queue, stop_event)
        worker.join(timeout=5)
        interrupted = result.get("interrupted", False)
    else:
        interrupted = run_session()

    print(f"Done. Predictions saved to: {predictions_path}")
    return 1 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
