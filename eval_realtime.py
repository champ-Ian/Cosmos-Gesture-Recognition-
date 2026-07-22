#!/usr/bin/env python3
"""
Live/near-real-time gesture evaluation using a model trained by `train.py`.

Modeled on `UWB_lab/eval_realtime.py`: opens whichever sensor streams the
loaded model actually needs (`payload["sensors"]`), keeps a sliding time
window per sensor, extracts the same features used during training (see
features.py), predicts every `--step-seconds`, and optionally smooths raw
predictions over `--vote-window` recent predictions via majority vote.
Predictions are printed live and saved to
`sessions/eval_<name>/realtime_predictions.csv`.

Only pass the `--*-port` flags for sensors your model actually uses --
`train.py --sensors` records which ones that is, and this script tells you
plainly if a required port is missing.

Example (a model trained on mmWave + IMU):

    python eval_realtime.py \\
      --model datasets/combined_gesture_dataset/models/knn_early_mmwave-imu_20260101_120000.joblib \\
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

from features import extract_sensor_features
from mmwave.mmwave_stream import MmwaveStream
from sensors.common import timestamp
from sensors.serial_json_stream import SerialLineStream
from uwb.uwb_stream import UwbStream

DEFAULT_MMWAVE_CFG = Path("mmwave/xwrL64xx-evm/near_field_hand_50cm.cfg")


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
    parser.add_argument("--out-root", default="sessions")
    parser.add_argument("--session-name")

    mmwave_group = parser.add_argument_group("mmWave radar (TI xWRL6432)")
    mmwave_group.add_argument("--mmwave-port")
    mmwave_group.add_argument("--mmwave-cfg", type=Path, default=DEFAULT_MMWAVE_CFG)
    mmwave_group.add_argument("--mmwave-baud", type=int, default=115200)
    mmwave_group.add_argument("--no-mmwave-warm-reset", action="store_true")

    imu_group = parser.add_argument_group("IMU (ESP32 Core2)")
    imu_group.add_argument("--imu-port")
    imu_group.add_argument("--imu-baud", type=int, default=115200)

    uwb_group = parser.add_argument_group("UWB (Qorvo DWM3001CDK FiRa TWR: anchor + node(s))")
    uwb_group.add_argument("--uwb-anchor-port")
    uwb_group.add_argument("--uwb-node-port", action="append")
    uwb_group.add_argument("--uwb-group-id", type=int)
    uwb_group.add_argument("--uwb-preamble-code", type=int, default=10)
    uwb_group.add_argument("--uwb-channel", type=int, choices=[5, 9], default=9)
    uwb_group.add_argument("--uwb-fps", type=float, default=50.0)
    uwb_group.add_argument("--uwb-slot-span", type=int, default=2400)
    uwb_group.add_argument("--uwb-slots-per-rr", type=int, default=None)
    uwb_group.add_argument("--uwb-skip-device-reset", action="store_true")

    rfid_group = parser.add_argument_group("RFID")
    rfid_group.add_argument("--rfid-port")
    rfid_group.add_argument("--rfid-baud", type=int, default=115200)

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
            streams["mmwave"] = MmwaveStream(
                port_path=args.mmwave_port,
                cfg_path=args.mmwave_cfg,
                baud=args.mmwave_baud,
                warm_reset=not args.no_mmwave_warm_reset,
            )
        if "imu" in required_sensors:
            if not args.imu_port:
                raise SystemExit("This model needs IMU -- pass --imu-port.")
            print(f"Opening IMU on {args.imu_port}...")
            streams["imu"] = SerialLineStream("imu", args.imu_port, args.imu_baud)
        if "uwb" in required_sensors:
            if not args.uwb_anchor_port or not args.uwb_node_port:
                raise SystemExit("This model needs UWB -- pass --uwb-anchor-port and --uwb-node-port.")
            if args.uwb_group_id is None:
                raise SystemExit("--uwb-group-id is required when UWB is enabled.")
            print(f"Opening UWB (anchor {args.uwb_anchor_port}, nodes {', '.join(args.uwb_node_port)})...")
            streams["uwb"] = UwbStream(
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
            if not args.rfid_port:
                raise SystemExit("This model needs RFID -- pass --rfid-port.")
            print(f"Opening RFID on {args.rfid_port}...")
            streams["rfid"] = SerialLineStream("rfid", args.rfid_port, args.rfid_baud)
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
    """Adapt a live stream's `.window()` output to the `{sensor}_*` keys features.py expects."""
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
