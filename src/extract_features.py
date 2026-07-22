#!/usr/bin/env python3
"""
Per-sensor feature extraction, plus the offline "cut" step that turns a raw
`collect.py` session into per-trial `trial_data.npz` files.

Modeled on `UWB_lab/uwb_lab_common.py`'s `extract_range_features()`: each
extractor takes the raw per-trial arrays for one sensor and returns a fixed-
length list of floats (or `None` if the trial doesn't have enough data), plus
a matching list of feature names. `train.py`/`evaluate.py`/`realtime_demo.py`
all use `build_feature_vector()` to turn a trial's selected sensors into one
combined vector (early fusion) or a per-sensor dict (late fusion).

These are starter/baseline features, the same way UWB_lab ships baseline
range features and leaves a `_proposal` extractor as a TODO for students --
expect to replace or extend these once you've looked at your own data.
IMU parses `IMU_lab_students`' `accel[g].../gyro[dps]...` log lines; RFID
parses `RFID_Lab`'s `<EPC> <timestamp> <RSSI> <read_count>` report lines --
neither is JSON, despite what earlier revisions of this repo assumed.

Run as a script to cut a raw session (`data/raw/session_.../`, written by
`collect.py`) into a processed dataset (`data/processed/<name>/`) --
see `cut_session()` / the `__main__` block at the bottom.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

from sensors.common import timestamp


def _finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).ravel()
    return values[np.isfinite(values)]


def _summary_stats(values: np.ndarray, prefix: str) -> tuple[list[float], list[str]]:
    """count/mean/std/min/max/first/last/delta -- reused across sensors."""
    values = _finite(values)
    if len(values) == 0:
        stats = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    else:
        stats = [
            float(len(values)),
            float(np.mean(values)),
            float(np.std(values)) if len(values) > 1 else 0.0,
            float(np.min(values)),
            float(np.max(values)),
            float(values[0]),
            float(values[-1]),
            float(values[-1] - values[0]),
        ]
    names = [f"{prefix}_{suffix}" for suffix in ("count", "mean", "std", "min", "max", "first", "last", "delta")]
    return stats, names


# ---------------------------------------------------------------------------
# mmWave radar
# ---------------------------------------------------------------------------

MMWAVE_FEATURE_NAMES = [
    "mmwave_frame_count",
    "mmwave_energy_mean",
    "mmwave_energy_std",
    "mmwave_energy_min",
    "mmwave_energy_max",
    "mmwave_energy_first",
    "mmwave_energy_last",
    "mmwave_energy_delta",
    "mmwave_point_count_mean",
    "mmwave_point_count_std",
    "mmwave_velocity_abs_mean",
    "mmwave_velocity_std",
    "mmwave_velocity_abs_max",
]


def extract_mmwave_features(npz) -> list[float] | None:
    frame_number = npz["mmwave_frame_number"]
    frame_count = len(frame_number)
    if frame_count == 0:
        return None

    range_profile = npz["mmwave_range_profile"]
    energy = np.nansum(range_profile, axis=1) if range_profile.size else np.zeros(frame_count)

    point_count = npz["mmwave_point_count"].astype(float)
    velocity = _finite(npz["mmwave_points_velocity"])

    features = [float(frame_count)]
    features += [
        float(np.mean(energy)),
        float(np.std(energy)) if len(energy) > 1 else 0.0,
        float(np.min(energy)),
        float(np.max(energy)),
        float(energy[0]),
        float(energy[-1]),
        float(energy[-1] - energy[0]),
    ]
    features += [
        float(np.mean(point_count)),
        float(np.std(point_count)) if len(point_count) > 1 else 0.0,
    ]
    if len(velocity) == 0:
        features += [0.0, 0.0, 0.0]
    else:
        features += [
            float(np.mean(np.abs(velocity))),
            float(np.std(velocity)) if len(velocity) > 1 else 0.0,
            float(np.max(np.abs(velocity))),
        ]

    return features


# ---------------------------------------------------------------------------
# UWB ranging
# ---------------------------------------------------------------------------

UWB_FEATURE_NAMES = [f"uwb_{suffix}" for suffix in ("count", "mean_cm", "std_cm", "min_cm", "max_cm", "median_cm", "range_cm", "q25_cm", "q75_cm", "iqr_cm", "first_cm", "last_cm", "delta_cm", "abs_delta_cm", "mean_abs_step_cm", "max_abs_step_cm", "slope_cm_per_sample")]


def extract_uwb_features(npz) -> list[float] | None:
    """Baseline UWB range features (same shape as `UWB_lab`'s `extract_range_features`).

    Pools all `status == "Ok"` distance samples in the trial window. With
    more than one anchor this does NOT distinguish which anchor a sample
    came from (`uwb_mac_address` is available in the trial `.npz` if you want
    to split per anchor) -- that's a deliberate baseline simplification, not
    a claim that anchor identity doesn't matter for your gestures.
    """
    status = npz["uwb_status"]
    distance_cm = npz["uwb_distance_cm"]
    if len(distance_cm) == 0:
        return None

    ok_mask = status == "Ok"
    values = distance_cm[ok_mask].astype(float)
    values = values[np.isfinite(values) & (values > 0) & (values < 60000)]
    if len(values) < 2:
        return None

    diffs = np.diff(values)
    q25, q75 = np.percentile(values, [25, 75])
    x = np.arange(len(values), dtype=float)
    slope = float(np.polyfit(x, values, deg=1)[0])

    return [
        float(len(values)),
        float(values.mean()),
        float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        float(values.min()),
        float(values.max()),
        float(np.median(values)),
        float(values.max() - values.min()),
        float(q25),
        float(q75),
        float(q75 - q25),
        float(values[0]),
        float(values[-1]),
        float(values[-1] - values[0]),
        float(abs(values[-1] - values[0])),
        float(np.mean(np.abs(diffs))) if len(diffs) else 0.0,
        float(np.max(np.abs(diffs))) if len(diffs) else 0.0,
        slope,
    ]


# ---------------------------------------------------------------------------
# IMU: ESP32 + BMI270 (IMU_lab_students firmware) log-line parsing
# ---------------------------------------------------------------------------

# Real firmware output (see IMU_lab_students/main/main.c, README.md):
#   accel[g] x= 0.012 y=-0.034 z= 0.998 | gyro[dps] x= 0.10 y=-0.20 z= 0.05
# Plain ESP_LOGI text, not JSON.
_FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_IMU_SAMPLE_RE = re.compile(
    rf"accel\[g\]\s+x=\s*({_FLOAT_RE})\s+y=\s*({_FLOAT_RE})\s+z=\s*({_FLOAT_RE})"
    rf"\s+\|\s+gyro\[dps\]\s+x=\s*({_FLOAT_RE})\s+y=\s*({_FLOAT_RE})\s+z=\s*({_FLOAT_RE})"
)

IMU_AXES = ("ax", "ay", "az", "gx", "gy", "gz")
IMU_FEATURE_NAMES = ["imu_line_count", "imu_record_count"]
for _axis in IMU_AXES:
    IMU_FEATURE_NAMES += [f"imu_{_axis}_mean", f"imu_{_axis}_std", f"imu_{_axis}_min", f"imu_{_axis}_max"]
IMU_FEATURE_NAMES += ["imu_accel_mag_mean", "imu_accel_mag_std"]


def parse_imu_line(line: str) -> tuple[float, float, float, float, float, float] | None:
    """Parse one `accel[g] ... | gyro[dps] ...` line into (ax,ay,az,gx,gy,gz), or None.

    Shared by `collect.py` (writes parsed columns to the live `imu.csv` log)
    and `_parse_imu_lines()` below (batch parsing for feature extraction).
    """
    match = _IMU_SAMPLE_RE.search(line)
    if not match:
        return None
    values = tuple(float(value) for value in match.groups())
    return values  # type: ignore[return-value]


def _parse_imu_lines(raw_lines) -> np.ndarray:
    """Parse `accel[g] ... | gyro[dps] ...` lines into an (n, 6) [ax,ay,az,gx,gy,gz] array."""
    rows = [parse_imu_line(line) for line in raw_lines]
    rows = [row for row in rows if row is not None]
    return np.array(rows, dtype=float) if rows else np.zeros((0, 6))


def extract_imu_features(npz) -> list[float] | None:
    """Parses the IMU_lab_students BMI270 firmware's `accel[g].../gyro[dps]...` text lines.

    If your group's firmware prints something else, update `_IMU_SAMPLE_RE`
    (or write your own extractor and register it in FEATURE_SPECS).
    """
    raw_lines = npz["imu_raw_lines"]
    line_count = len(raw_lines)
    if line_count == 0:
        return None

    samples = _parse_imu_lines(raw_lines)
    if len(samples) < 2:
        return None

    features = [float(line_count), float(len(samples))]
    for i in range(6):
        values = samples[:, i]
        features += [
            float(np.mean(values)),
            float(np.std(values)) if len(values) > 1 else 0.0,
            float(np.min(values)),
            float(np.max(values)),
        ]

    ax, ay, az = samples[:, 0], samples[:, 1], samples[:, 2]
    magnitude = np.sqrt(ax**2 + ay**2 + az**2)
    features += [float(np.mean(magnitude)), float(np.std(magnitude)) if len(magnitude) > 1 else 0.0]

    return features


# ---------------------------------------------------------------------------
# RFID: RFID_Lab reader log-line parsing (TCP, not serial)
# ---------------------------------------------------------------------------

# Real reader output (see RFID_Lab/rfid_log_utils.py): space-separated
#   <EPC> <timestamp (one or more tokens)> <RSSI> <read_count>
# e.g. "E2806995000040154D38514E 2024-01-01 12:00:00.123 -45 3"


def parse_rfid_line(line: str) -> tuple[str, int, int] | None:
    """Parse one RFID_Lab report line into (epc, rssi, read_count), or None.

    Shared by `collect.py` (writes parsed columns to the live `rfid.csv` log)
    and `_parse_rfid_lines()` below (batch parsing for feature extraction).
    """
    parts = str(line).strip().split()
    if len(parts) < 5:
        return None
    try:
        rssi = int(parts[-2])
        read_count = int(parts[-1])
    except ValueError:
        return None
    return parts[0], rssi, read_count


def _parse_rfid_lines(raw_lines) -> list[tuple[str, int, int]]:
    """Parse RFID_Lab report lines into (epc, rssi, read_count) tuples."""
    records = [parse_rfid_line(line) for line in raw_lines]
    return [record for record in records if record is not None]


RFID_FEATURE_NAMES = [
    "rfid_line_count",
    "rfid_record_count",
    "rfid_unique_tag_count",
    "rfid_rssi_mean",
    "rfid_rssi_std",
    "rfid_rssi_min",
    "rfid_rssi_max",
    "rfid_read_count_sum",
]


def extract_rfid_features(npz) -> list[float] | None:
    """Baseline RFID features: pools every tag read in the trial window together.

    The reader streams a line for every EPC it sees, not just tags you care
    about. This pools all of them -- if your gestures use specific tags
    (e.g. one per finger for Soli-style sensing), filter `_parse_rfid_lines()`
    output by EPC first and compute per-tag features instead.
    """
    raw_lines = npz["rfid_raw_lines"]
    line_count = len(raw_lines)
    if line_count == 0:
        return None

    records = _parse_rfid_lines(raw_lines)
    if len(records) < 2:
        return None

    tag_ids = {epc for epc, _, _ in records}
    rssi = np.array([rssi for _, rssi, _ in records], dtype=float)
    read_count_sum = float(sum(read_count for _, _, read_count in records))

    return [
        float(line_count),
        float(len(records)),
        float(len(tag_ids)),
        float(np.mean(rssi)),
        float(np.std(rssi)) if len(rssi) > 1 else 0.0,
        float(np.min(rssi)),
        float(np.max(rssi)),
        read_count_sum,
    ]


# ---------------------------------------------------------------------------
# Combined feature vector
# ---------------------------------------------------------------------------

FEATURE_SPECS = {
    "mmwave": (extract_mmwave_features, MMWAVE_FEATURE_NAMES),
    "imu": (extract_imu_features, IMU_FEATURE_NAMES),
    "uwb": (extract_uwb_features, UWB_FEATURE_NAMES),
    "rfid": (extract_rfid_features, RFID_FEATURE_NAMES),
}


def feature_names_for_sensor(sensor: str) -> list[str]:
    return list(FEATURE_SPECS[sensor][1])


def extract_sensor_features(sensor: str, npz) -> list[float] | None:
    extractor, _ = FEATURE_SPECS[sensor]
    return extractor(npz)


def build_feature_vector(npz, sensors: list[str]) -> list[float] | None:
    """Early-fusion feature vector: concatenate every requested sensor's features.

    Returns None if ANY requested sensor's features can't be extracted from
    this trial (missing data, too few samples, etc.) -- the trial gets
    skipped rather than silently zero-filled.
    """
    combined: list[float] = []
    for sensor in sensors:
        sensor_features = extract_sensor_features(sensor, npz)
        if sensor_features is None:
            return None
        combined.extend(sensor_features)
    return combined


def feature_names_for_sensors(sensors: list[str]) -> list[str]:
    names: list[str] = []
    for sensor in sensors:
        names.extend(feature_names_for_sensor(sensor))
    return names


# ---------------------------------------------------------------------------
# Cutting a raw collect.py session into a processed per-trial dataset
# ---------------------------------------------------------------------------
#
# collect.py streams continuously and only writes event markers + continuous
# per-sensor logs (imu.csv/uwb.csv/rfid.csv/mmwave.npz + events.csv). Nothing
# is cut into per-trial windows at collection time. `cut_session()` does that
# offline: it reads events.csv for each accepted trial's [start, end] window,
# slices every enabled sensor's continuous log to that window, and writes the
# *same* per-trial `trial_data.npz` + `trials.csv` manifest shape the old
# collector used to write directly -- so train.py/evaluate.py/
# combine_datasets.py don't need to change at all.
#
# "periodic" gestures (see gestures.py's GestureSpec.group) were recorded as
# one long continuous take; here their window gets split into fixed-length
# sub-windows (--segment-length/--segment-stride) that each become their own
# trial row sharing the same gesture label. "discrete" gestures pass through
# as a single segment (the trial's own recorded start/end, unsegmented).

INPUT_TYPE = "cosmos_multi_sensor_gesture"

MANIFEST_FIELDNAMES = [
    "dataset_name",
    "collector",
    "gesture",
    "gesture_group",
    "input_type",
    "trial_index",
    "attempt_index",
    "segment_index",
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


def _read_csv_rows(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _trial_windows(events: list[dict]) -> dict[str, tuple[float, float]]:
    starts: dict[str, float] = {}
    ends: dict[str, float] = {}
    for row in events:
        if row["event"] == "trial_start":
            starts[row["trial_id"]] = float(row["time_s"])
        elif row["event"] == "trial_end":
            ends[row["trial_id"]] = float(row["time_s"])
    return {trial_id: (starts[trial_id], ends[trial_id]) for trial_id in starts if trial_id in ends}


def _filter_rows(rows: list[dict], start: float, end: float) -> list[dict]:
    return [row for row in rows if start <= float(row["time_s"]) <= end]


def _segment_windows(
    start: float, end: float, segment_length: float, segment_stride: float
) -> list[tuple[float, float]]:
    if segment_length <= 0 or end - start <= segment_length:
        return [(start, end)]
    windows = []
    t = start
    while t + segment_length <= end:
        windows.append((t, t + segment_length))
        t += segment_stride
    return windows if windows else [(start, end)]


def _cut_one_trial(
    output_dir: Path,
    trial: dict,
    trial_id: str,
    segment_index: int,
    start: float,
    end: float,
    imu_rows: list[dict] | None,
    uwb_rows: list[dict] | None,
    rfid_rows: list[dict] | None,
    mmwave_data: dict[str, np.ndarray] | None,
) -> dict:
    trial_out_dir = output_dir / "sessions" / trial_id
    trial_out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = trial_out_dir / "trial_data.npz"

    gesture = trial["gesture"]
    collector = trial["collector"]
    sensors_enabled = [s for s in trial.get("sensors_enabled", "").split(",") if s]
    capture_duration_s = max(end - start, 1e-9)

    payload: dict[str, np.ndarray] = {
        "dataset_name": np.array(output_dir.name),
        "collector": np.array(collector),
        "gesture": np.array(gesture),
        "input_type": np.array(INPUT_TYPE),
        "duration_s": np.array(float(trial.get("planned_duration_s") or capture_duration_s)),
        "capture_duration_s": np.array(capture_duration_s),
        "sensors_enabled": np.array(",".join(sensors_enabled)),
    }

    mmwave_frame_count = 0
    mmwave_mean_fps = 0.0
    if "mmwave" in sensors_enabled and mmwave_data is not None:
        mask = (mmwave_data["mmwave_time_s"] >= start) & (mmwave_data["mmwave_time_s"] <= end)
        payload["mmwave_frame_number"] = mmwave_data["mmwave_frame_number"][mask]
        payload["mmwave_time_s"] = mmwave_data["mmwave_time_s"][mask] - start
        payload["mmwave_range_profile"] = mmwave_data["mmwave_range_profile"][mask]
        payload["mmwave_point_count"] = mmwave_data["mmwave_point_count"][mask]
        payload["mmwave_points_xyz"] = mmwave_data["mmwave_points_xyz"][mask]
        payload["mmwave_points_velocity"] = mmwave_data["mmwave_points_velocity"][mask]
        mmwave_frame_count = int(mask.sum())
        mmwave_mean_fps = mmwave_frame_count / capture_duration_s

    imu_line_count = 0
    if "imu" in sensors_enabled and imu_rows is not None:
        rows = _filter_rows(imu_rows, start, end)
        payload["imu_recv_time_s"] = np.array([float(row["time_s"]) - start for row in rows], dtype=float)
        payload["imu_raw_lines"] = np.array([row["raw_line"] for row in rows], dtype=object)
        imu_line_count = len(rows)

    uwb_sample_count = 0
    uwb_ok_count = 0
    if "uwb" in sensors_enabled and uwb_rows is not None:
        rows = _filter_rows(uwb_rows, start, end)
        payload["uwb_time_s"] = np.array([float(row["time_s"]) - start for row in rows], dtype=float)
        payload["uwb_sequence"] = np.array([int(row["sequence"]) for row in rows], dtype=np.int32)
        payload["uwb_mac_address"] = np.array([row["mac_address"] for row in rows], dtype=object)
        payload["uwb_status"] = np.array([row["status"] for row in rows], dtype=object)
        payload["uwb_distance_cm"] = np.array([float(row["distance_cm"]) for row in rows], dtype=float)
        uwb_sample_count = len(rows)
        uwb_ok_count = sum(1 for row in rows if row["status"] == "Ok")

    rfid_line_count = 0
    if "rfid" in sensors_enabled and rfid_rows is not None:
        rows = _filter_rows(rfid_rows, start, end)
        payload["rfid_recv_time_s"] = np.array([float(row["time_s"]) - start for row in rows], dtype=float)
        payload["rfid_raw_lines"] = np.array([row["raw_line"] for row in rows], dtype=object)
        rfid_line_count = len(rows)

    np.savez_compressed(npz_path, **payload)

    return {
        "dataset_name": output_dir.name,
        "collector": collector,
        "gesture": gesture,
        "gesture_group": trial.get("gesture_group", "discrete"),
        "input_type": INPUT_TYPE,
        "trial_index": trial.get("trial_index", ""),
        "attempt_index": trial.get("attempt_index", ""),
        "segment_index": segment_index,
        "duration_s": trial.get("planned_duration_s", ""),
        "capture_duration_s": capture_duration_s,
        "sensors_enabled": ",".join(sensors_enabled),
        "mmwave_frame_count": mmwave_frame_count,
        "mmwave_mean_frame_rate_hz": mmwave_mean_fps,
        "imu_line_count": imu_line_count,
        "uwb_sample_count": uwb_sample_count,
        "uwb_ok_sample_count": uwb_ok_count,
        "rfid_line_count": rfid_line_count,
        "npz_path": str(npz_path),
        "session_dir": str(trial_out_dir),
        "started_at": "",
        "finished_at": "",
    }


def cut_session(
    session_dir: Path,
    output_dir: Path,
    segment_length: float = 3.0,
    segment_stride: float | None = None,
) -> dict:
    """Cut a raw `collect.py` session into a processed per-trial dataset.

    Returns the same dict written to `<output_dir>/dataset_metadata.json`.
    """
    session_dir = Path(session_dir)
    output_dir = Path(output_dir)
    segment_stride = segment_length if segment_stride is None else segment_stride

    session_metadata = json.loads((session_dir / "session_metadata.json").read_text())
    events = _read_csv_rows(session_dir / "events.csv") or []
    trials = _read_csv_rows(session_dir / "trials.csv") or []
    windows = _trial_windows(events)

    imu_rows = _read_csv_rows(session_dir / "imu.csv")
    uwb_rows = _read_csv_rows(session_dir / "uwb.csv")
    rfid_rows = _read_csv_rows(session_dir / "rfid.csv")
    mmwave_path = session_dir / "mmwave.npz"
    mmwave_data = None
    if mmwave_path.exists():
        with np.load(mmwave_path) as npz:
            mmwave_data = {key: npz[key] for key in npz.files}

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[dict] = []
    skipped: list[dict] = []

    for trial in trials:
        trial_id = trial["trial_id"]
        if trial_id not in windows:
            skipped.append({"trial_id": trial_id, "reason": "missing trial_start/trial_end in events.csv"})
            continue
        start, end = windows[trial_id]
        gesture_group = trial.get("gesture_group", "discrete")
        segments = (
            _segment_windows(start, end, segment_length, segment_stride)
            if gesture_group == "periodic"
            else [(start, end)]
        )

        for segment_index, (seg_start, seg_end) in enumerate(segments):
            segment_trial_id = trial_id if len(segments) == 1 else f"{trial_id}_seg{segment_index:03d}"
            written.append(
                _cut_one_trial(
                    output_dir,
                    trial,
                    segment_trial_id,
                    segment_index,
                    seg_start,
                    seg_end,
                    imu_rows,
                    uwb_rows,
                    rfid_rows,
                    mmwave_data,
                )
            )

    manifest_path = output_dir / "trials.csv"
    with manifest_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDNAMES)
        writer.writeheader()
        writer.writerows(written)

    dataset_metadata = {
        "dataset_name": output_dir.name,
        "input_type": INPUT_TYPE,
        "source_session": str(session_dir),
        "collector": session_metadata.get("collector"),
        "segment_length_s": segment_length,
        "segment_stride_s": segment_stride,
        "row_count": len(written),
        "skipped": skipped,
        "created_at": timestamp(),
    }
    (output_dir / "dataset_metadata.json").write_text(json.dumps(dataset_metadata, indent=2, sort_keys=True) + "\n")
    return dataset_metadata


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cut a raw collect.py session into a processed per-trial dataset."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    cut_parser = subparsers.add_parser("cut", help="Cut a raw session into data/processed/<name>/.")
    cut_parser.add_argument("session_dir", type=Path, help="Raw session folder from collect.py.")
    cut_parser.add_argument("--output", type=Path, required=True, help="Output processed dataset folder.")
    cut_parser.add_argument(
        "--segment-length",
        type=float,
        default=3.0,
        help="Sub-window length (s) for periodic-gesture segmentation.",
    )
    cut_parser.add_argument(
        "--segment-stride",
        type=float,
        default=None,
        help="Sub-window stride (s); default = --segment-length (non-overlapping).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_cli_args()
    if args.command == "cut":
        metadata = cut_session(args.session_dir, args.output, args.segment_length, args.segment_stride)
        print(json.dumps(metadata, indent=2, sort_keys=True))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
