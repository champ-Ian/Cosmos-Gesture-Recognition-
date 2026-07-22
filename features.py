#!/usr/bin/env python3
"""
Per-sensor feature extraction from `trial_data.npz` files saved by
`collect_gesture_dataset.py`.

Modeled on `UWB_lab/uwb_lab_common.py`'s `extract_range_features()`: each
extractor takes the raw per-trial arrays for one sensor and returns a fixed-
length list of floats (or `None` if the trial doesn't have enough data), plus
a matching list of feature names. `train.py` and `eval_realtime.py` both use
`build_feature_vector()` to turn a trial's selected sensors into one combined
vector (early fusion) or a per-sensor dict (late fusion).

These are starter/baseline features, the same way UWB_lab ships baseline
range features and leaves a `_proposal` extractor as a TODO for students --
expect to replace or extend these once you've looked at your own data,
especially IMU/RFID (the JSON-line schema is whatever your group's firmware
actually prints; adjust `_parse_json_lines` field names to match).
"""
from __future__ import annotations

import json
import math

import numpy as np


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
# IMU / RFID: best-effort JSON-line parsing
# ---------------------------------------------------------------------------


def _parse_json_lines(raw_lines) -> list[dict]:
    records = []
    for line in raw_lines:
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _numeric_field(records: list[dict], field: str) -> np.ndarray:
    values = []
    for record in records:
        value = record.get(field)
        if isinstance(value, (int, float)) and math.isfinite(value):
            values.append(float(value))
    return np.array(values, dtype=float)


IMU_AXES = ("ax", "ay", "az", "gx", "gy", "gz")
IMU_FEATURE_NAMES = ["imu_line_count", "imu_record_count"]
for _axis in IMU_AXES:
    IMU_FEATURE_NAMES += [f"imu_{_axis}_mean", f"imu_{_axis}_std", f"imu_{_axis}_min", f"imu_{_axis}_max"]
IMU_FEATURE_NAMES += ["imu_accel_mag_mean", "imu_accel_mag_std"]


def extract_imu_features(npz) -> list[float] | None:
    """Assumes the recommended firmware JSON schema (see sensors/serial_json_stream.py).

    If your firmware prints something else, this will return all-zero axis
    stats (not crash) -- rewrite the field names below to match your actual
    JSON keys, or write your own extractor and register it in FEATURE_SPECS.
    """
    raw_lines = npz["imu_raw_lines"]
    line_count = len(raw_lines)
    if line_count == 0:
        return None

    records = _parse_json_lines(raw_lines)
    if len(records) < 2:
        return None

    features = [float(line_count), float(len(records))]
    axis_values = {}
    for axis in IMU_AXES:
        values = _numeric_field(records, axis)
        axis_values[axis] = values
        if len(values) == 0:
            features += [0.0, 0.0, 0.0, 0.0]
        else:
            features += [
                float(np.mean(values)),
                float(np.std(values)) if len(values) > 1 else 0.0,
                float(np.min(values)),
                float(np.max(values)),
            ]

    ax, ay, az = axis_values["ax"], axis_values["ay"], axis_values["az"]
    n = min(len(ax), len(ay), len(az))
    if n == 0:
        features += [0.0, 0.0]
    else:
        magnitude = np.sqrt(ax[:n] ** 2 + ay[:n] ** 2 + az[:n] ** 2)
        features += [float(np.mean(magnitude)), float(np.std(magnitude)) if n > 1 else 0.0]

    return features


RFID_FEATURE_NAMES = ["rfid_line_count", "rfid_record_count", "rfid_unique_tag_count", "rfid_rssi_mean", "rfid_rssi_std", "rfid_rssi_min", "rfid_rssi_max"]


def extract_rfid_features(npz) -> list[float] | None:
    """Assumes the recommended firmware JSON schema (see sensors/serial_json_stream.py)."""
    raw_lines = npz["rfid_raw_lines"]
    line_count = len(raw_lines)
    if line_count == 0:
        return None

    records = _parse_json_lines(raw_lines)
    if len(records) < 2:
        return None

    tag_ids = {record.get("tag_id") for record in records if record.get("tag_id")}
    rssi = _numeric_field(records, "rssi")

    features = [float(line_count), float(len(records)), float(len(tag_ids))]
    if len(rssi) == 0:
        features += [0.0, 0.0, 0.0, 0.0]
    else:
        features += [
            float(np.mean(rssi)),
            float(np.std(rssi)) if len(rssi) > 1 else 0.0,
            float(np.min(rssi)),
            float(np.max(rssi)),
        ]

    return features


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
