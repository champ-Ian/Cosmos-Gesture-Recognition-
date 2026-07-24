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
then one column per feature (see extract_features.py's *_FEATURE_NAMES for
what each one means), in the same order train.py's early-fusion vector uses.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from extract_features import feature_names_for_sensors
from train import build_examples, normalize_sensor_list, read_manifests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", nargs="+", help="Processed dataset folders from extract_features.py cut.")
    parser.add_argument("--sensors", required=True, help="Comma-separated sensor subset, e.g. 'mmwave,imu,uwb'.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
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
