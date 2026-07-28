#!/usr/bin/env python3
"""
Live/near-real-time gesture evaluation using a model trained by `train.py`.

Modeled on `UWB_lab/eval_realtime.py`: opens whichever sensor streams the
loaded model actually needs (`payload["sensors"]`), keeps a sliding time
window per sensor, extracts the same features used during training (see
extract_features.py), predicts every `--step-seconds`, and optionally
smooths raw predictions over `--vote-window` recent predictions via
majority vote. Predictions are printed live and saved to
`sessions/eval_<name>/realtime_predictions.csv`.

Only pass the `--*-port` flags for sensors your model actually uses --
`train.py --sensors` records which ones that is, and this script tells you
plainly if a required port is missing.

Example (run from the repo root; a model trained on mmWave + IMU):

    python src/realtime_demo.py \\
      --model models/knn_early_mmwave-imu_20260101_120000.joblib \\
      --mmwave-port /dev/cu.usbserial-XXXX \\
      --imu-port /dev/cu.usbserial-YYYY \\
      --duration 60 --window-seconds 3 --step-seconds 0.5 --vote-window 5
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import deque
from pathlib import Path

import numpy as np

from extract_features import extract_sensor_features, feature_names_for_sensor
from sensors.common import REPO_DIR, timestamp
from sensors.imu_reader import ImuReader
from sensors.mmwave_reader import MmwaveReader
from sensors.rfid_reader import RfidReader
from sensors.uwb_reader import UwbReader

# Resolved relative to this file (src/), not the current working directory --
# see collect.py's DEFAULT_MMWAVE_CFG for why.
DEFAULT_MMWAVE_CFG = Path(__file__).resolve().parent / "mmwave" / "xwrL64xx-evm" / "near_field_hand_50cm.cfg"

# ---------------------------------------------------------------------------
# Rest / "not a gesture" detection -- a rule-based gate, not a trained class.
#
# Each of these features (already computed by extract_sensor_features(), same
# as what the classifier sees) is ~0 when nothing is moving and grows with
# real motion. If every active sensor's value stays under its threshold, we
# report REST_LABEL directly and skip the classifier entirely for that step.
# ---------------------------------------------------------------------------

REST_LABEL = "resting"

REST_MOTION_FEATURES = {
    "imu": ["imu_accel_mag_std", "imu_gx_std", "imu_gy_std", "imu_gz_std"],
    "mmwave": ["mmwave_velocity_abs_mean", "mmwave_point_count_std"],
    "uwb": ["uwb_mean_abs_step_cm"],
}

# Ballpark fallback thresholds, used when --rest-calibration-seconds is 0 (or
# too short to produce a usable baseline). --rest-calibration-seconds (the
# default) measures your own hardware's noise floor instead and is more
# reliable across different kits/people -- treat these as a rough safety net,
# not the intended source of truth.
DEFAULT_REST_THRESHOLDS = {
    "imu_accel_mag_std": 0.06,
    "imu_gx_std": 5.0,
    "imu_gy_std": 5.0,
    "imu_gz_std": 5.0,
    "mmwave_velocity_abs_mean": 0.05,
    "mmwave_point_count_std": 2.0,
    "uwb_mean_abs_step_cm": 1.5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Model .joblib from train.py.")
    parser.add_argument("--duration", type=float, default=60.0, help="Seconds to run the live session.")
    parser.add_argument("--window-seconds", type=float, default=3.0, help="Sliding feature-extraction window.")
    parser.add_argument("--step-seconds", type=float, default=0.5, help="Seconds between predictions.")
    parser.add_argument(
        "--vote-window",
        type=int,
        default=1,
        help="Majority-vote over this many recent raw predictions. 1 = show raw predictions.",
    )
    parser.add_argument("--out-root", default=str(REPO_DIR / "sessions"))
    parser.add_argument("--session-name")

    rest_group = parser.add_argument_group(
        "Rest / no-gesture detection (rule-based gate, not a trained class -- see REST_MOTION_FEATURES)"
    )
    rest_group.add_argument(
        "--no-rest-detection",
        action="store_true",
        help="Disable rest detection entirely -- always classify into one of the trained gestures.",
    )
    rest_group.add_argument(
        "--rest-calibration-seconds",
        type=float,
        default=2.0,
        help=(
            "Seconds to hold still at startup to calibrate rest thresholds from this hardware's own "
            "noise floor (recommended). Set to 0 to skip calibration and use fixed built-in defaults."
        ),
    )
    rest_group.add_argument(
        "--rest-threshold-multiplier",
        type=float,
        default=4.0,
        help="Calibrated threshold = baseline_mean + multiplier * baseline_std, per motion feature.",
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

    return parser.parse_args()


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


def sensor_rest_ratio(sensor: str, feature_vector: list[float], thresholds: dict[str, float]) -> float | None:
    """Max(|feature value| / threshold) over this sensor's configured motion features.

    >1 means this sensor alone shows enough motion to rule out "resting". None if this
    sensor has no configured motion features (e.g. rfid) -- it's simply not consulted.
    """
    feature_names = REST_MOTION_FEATURES.get(sensor)
    if not feature_names:
        return None
    names = feature_names_for_sensor(sensor)
    ratios = []
    for feat in feature_names:
        value = abs(feature_vector[names.index(feat)])
        threshold = thresholds.get(feat, DEFAULT_REST_THRESHOLDS[feat])
        ratios.append((value / threshold) if threshold > 0 else (0.0 if value == 0 else float("inf")))
    return max(ratios)


def is_at_rest(sensors: list[str], per_sensor_vectors: dict, thresholds: dict[str, float]) -> bool:
    """True if every sensor with configured motion features reports staying under threshold."""
    ratios = [
        ratio
        for sensor in sensors
        if (ratio := sensor_rest_ratio(sensor, per_sensor_vectors[sensor], thresholds)) is not None
    ]
    return bool(ratios) and max(ratios) <= 1.0


def calibrate_rest_thresholds(
    streams: dict,
    calibration_seconds: float,
    multiplier: float,
    window_seconds: float,
    step_seconds: float,
) -> dict[str, float]:
    """Sample motion features while the user holds still to derive per-hardware rest thresholds.

    Falls back to DEFAULT_REST_THRESHOLDS per feature if calibration is skipped (0s) or a
    feature never got enough samples (e.g. that sensor isn't in `streams`).
    """
    if calibration_seconds <= 0:
        return dict(DEFAULT_REST_THRESHOLDS)

    print(f"Calibrating rest baseline -- hold your hands still for {calibration_seconds:.1f}s...")
    samples: dict[str, list[float]] = {feat: [] for feats in REST_MOTION_FEATURES.values() for feat in feats}

    start = time.monotonic()
    end = start + calibration_seconds
    next_sample = start
    while time.monotonic() < end:
        now = time.monotonic()
        if now >= next_sample:
            for sensor, stream in streams.items():
                feats = REST_MOTION_FEATURES.get(sensor)
                if not feats:
                    continue
                window = stream.window(max(start, now - window_seconds), now)
                vector = extract_sensor_features(sensor, window_to_feature_input(sensor, window))
                if vector is None:
                    continue
                names = feature_names_for_sensor(sensor)
                for feat in feats:
                    samples[feat].append(vector[names.index(feat)])
            next_sample = now + step_seconds
        time.sleep(0.02)

    thresholds: dict[str, float] = {}
    for feat, values in samples.items():
        default = DEFAULT_REST_THRESHOLDS[feat]
        if len(values) < 2:
            thresholds[feat] = default
            continue
        arr = np.array(values)
        calibrated = float(np.mean(arr) + multiplier * np.std(arr))
        # A near-perfectly-still calibration take (std ~ 0) shouldn't make the gate trip on
        # ordinary sensor noise later -- never let calibration set the bar below half the default.
        thresholds[feat] = max(calibrated, 0.5 * default)
    print("Rest calibration done:", {k: round(v, 4) for k, v in thresholds.items()})
    return thresholds


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


def main() -> int:
    args = parse_args()
    args.vote_window = max(1, int(args.vote_window))

    try:
        import joblib
    except ImportError as exc:
        raise SystemExit(f"Missing joblib/sklearn environment: {exc}")

    payload = joblib.load(args.model)
    model = payload["model"]
    sensors = payload["sensors"]
    fusion = payload.get("fusion", "early")
    classifier_label = payload.get("classifier_label", payload.get("classifier", "unknown"))
    labels = payload.get("labels", [])

    session_name = args.session_name or f"eval_{timestamp()}"
    session_dir = Path(args.out_root).expanduser().resolve() / session_name
    session_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = session_dir / "realtime_predictions.csv"

    print(f"Loaded model: {classifier_label} ({fusion} fusion: {'+'.join(sensors)})")
    print(f"Gesture labels: {', '.join(labels)}")
    print(f"Session folder: {session_dir}")

    streams = open_streams(args, sensors, session_dir)
    time.sleep(0.5)  # let boards start producing data

    rest_thresholds = None
    if not args.no_rest_detection:
        rest_thresholds = calibrate_rest_thresholds(
            streams, args.rest_calibration_seconds, args.rest_threshold_multiplier,
            args.window_seconds, args.step_seconds,
        )

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
                "vote_fraction",
                "vote_count",
                "vote_window",
                "vote_counts_json",
            ],
        )
        writer.writeheader()

        session_start = time.monotonic()
        end_time = session_start + args.duration
        last_prediction_time = 0.0

        try:
            while time.monotonic() < end_time:
                now = time.monotonic()
                for stream in streams.values():
                    stream.check_error()

                if now - last_prediction_time >= args.step_seconds:
                    window_start = max(session_start, now - args.window_seconds)
                    per_sensor_vectors = {}
                    ready = True
                    for sensor, stream in streams.items():
                        window = stream.window(window_start, now)
                        vector = extract_sensor_features(sensor, window_to_feature_input(sensor, window))
                        if vector is None:
                            ready = False
                            break
                        per_sensor_vectors[sensor] = vector

                    if ready:
                        if rest_thresholds is not None and is_at_rest(sensors, per_sensor_vectors, rest_thresholds):
                            raw_prediction, raw_confidence = REST_LABEL, None
                        else:
                            raw_prediction, raw_confidence = predict(model, sensors, fusion, per_sensor_vectors)
                        vote_history.append(raw_prediction)
                        if args.vote_window <= 1:
                            display_prediction, display_confidence = raw_prediction, raw_confidence
                            vote_fraction, vote_counts = None, {}
                        else:
                            display_prediction, vote_fraction, vote_counts = majority_vote(vote_history)
                            display_confidence = vote_fraction

                        writer.writerow(
                            {
                                "time_s": f"{now - session_start:.3f}",
                                "prediction": display_prediction,
                                "confidence": "" if display_confidence is None else f"{display_confidence:.4f}",
                                "raw_prediction": raw_prediction,
                                "raw_confidence": "" if raw_confidence is None else f"{raw_confidence:.4f}",
                                "vote_fraction": "" if vote_fraction is None else f"{vote_fraction:.4f}",
                                "vote_count": len(vote_history),
                                "vote_window": args.vote_window,
                                "vote_counts_json": json.dumps(vote_counts, sort_keys=True),
                            }
                        )
                        prediction_file.flush()
                        if args.vote_window <= 1:
                            conf_text = "" if raw_confidence is None else f" ({raw_confidence:.2f})"
                            print(f"prediction: {display_prediction}{conf_text}", flush=True)
                        else:
                            vote_text = "" if vote_fraction is None else f" vote={vote_fraction:.2f}"
                            raw_text = "" if raw_confidence is None else f" ({raw_confidence:.2f})"
                            print(f"prediction: {display_prediction}{vote_text} | raw: {raw_prediction}{raw_text}", flush=True)

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

    print(f"Done. Predictions saved to: {predictions_path}")
    return 1 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
