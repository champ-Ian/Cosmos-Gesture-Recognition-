#!/usr/bin/env python3
"""
Export extract_sensor_features() output to one flat CSV: a row per usable
trial/segment, a column per early-fusion feature, plus label/metadata columns.

Reuses train.py's read_manifests()/build_examples(), so a row appears here
if and only if `train.py` would have used it for training with the same
--sensors -- if a gesture you expect is missing, the printed skip-reason
breakdown (same one train.py prints) tells you why.

Run from the repo root:

    python src/export_features_csv.py data/processed/student1_minimodel \\
      --sensors mmwave,imu,uwb \\
      --output data/processed/student1_minimodel/features.csv

The output has columns: gesture, collector, source_dataset, session_dir,
then one column per feature, in the same order train.py's early-fusion
vector uses. `--feature-mode` controls what those feature columns are (see
extract_features.py's *_FEATURE_NAMES for what each one means):

    summary (default) -- mean/std/min/max per axis, one row-vector regardless
        of how many raw samples a trial had.
    raw -- each sensor's raw stream resampled onto a fixed 20-point sequence
        per axis (e.g. imu_raw_ax_0..19), preserving shape/timing instead of
        collapsing to a few statistics. Still fixed-length per trial (needed
        for sklearn), just far less lossy than summary stats.
    both -- summary columns AND raw columns together, per sensor, in one CSV.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from extract_features import (
    extract_sensor_features,
    extract_sensor_raw_sequence,
    feature_names_for_sensor,
    feature_names_for_sensors,
    raw_feature_names_for_sensor,
    raw_feature_names_for_sensors,
)
from train import build_examples, normalize_sensor_list, read_manifests


def combined_sensor_features(sensor: str, npz) -> list[float] | None:
    summary = extract_sensor_features(sensor, npz)
    raw = extract_sensor_raw_sequence(sensor, npz)
    if summary is None or raw is None:
        return None
    return summary + raw


def combined_feature_names(sensors: list[str]) -> list[str]:
    names: list[str] = []
    for sensor in sensors:
        names.extend(feature_names_for_sensor(sensor))
        names.extend(raw_feature_names_for_sensor(sensor))
    return names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", nargs="+", help="Processed dataset folders from extract_features.py cut.")
    parser.add_argument("--sensors", required=True, help="Comma-separated sensor subset, e.g. 'mmwave,imu,uwb'.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument(
        "--feature-mode",
        choices=["summary", "raw", "both"],
        default="summary",
        help=(
            "'summary' = mean/std/min/max per axis (default). 'raw' = fixed-length resampled raw "
            "sequence per axis. 'both' = summary and raw columns together per sensor."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sensors = normalize_sensor_list(args.sensors)
    dataset_dirs = [Path(item).expanduser().resolve() for item in args.datasets]
    rows, missing_datasets = read_manifests(dataset_dirs)
    if not rows:
        raise SystemExit("No trials found. Expected trials.csv or trial_metadata.json files.")
    if missing_datasets:
        print(f"Warning: no trials found in: {', '.join(missing_datasets)}")

    mode_extractors = {
        "summary": None,
        "raw": extract_sensor_raw_sequence,
        "both": combined_sensor_features,
    }
    extractor = mode_extractors[args.feature_mode]
    if extractor is not None:
        per_sensor_examples, labels, collectors, sources, session_dirs, skipped = build_examples(
            rows, sensors, feature_extractor=extractor
        )
    else:
        per_sensor_examples, labels, collectors, sources, session_dirs, skipped = build_examples(rows, sensors)
    if skipped:
        reason_counts: dict[str, int] = {}
        for entry in skipped:
            reason_counts[entry["reason"]] = reason_counts.get(entry["reason"], 0) + 1
        print(f"Skipped {len(skipped)}/{len(rows)} rows:")
        for reason, count in sorted(reason_counts.items(), key=lambda item: -item[1]):
            print(f"  {count:4d}  {reason}")
        skipped_labels = sorted({entry["gesture"] for entry in skipped} - set(labels))
        if skipped_labels:
            print(f"Gestures with zero usable trials: {', '.join(skipped_labels)}")

    if not labels:
        raise SystemExit("No usable trials for the requested --sensors.")

    if args.feature_mode == "raw":
        feature_names = raw_feature_names_for_sensors(sensors)
    elif args.feature_mode == "both":
        feature_names = combined_feature_names(sensors)
    else:
        feature_names = feature_names_for_sensors(sensors)
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["gesture", "collector", "source_dataset", "session_dir"] + feature_names
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for i in range(len(labels)):
            row = [labels[i], collectors[i], sources[i], session_dirs[i]]
            for sensor in sensors:
                row.extend(per_sensor_examples[sensor][i])
            writer.writerow(row)

    print(f"Wrote {len(labels)} rows x {len(feature_names)} feature columns to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
