#!/usr/bin/env python3
"""
Combine gesture datasets collected by multiple group members into one.

Mirrors `mmwave_lab/combine_posture_datasets.py`: concatenates each source
dataset's `trials.csv` manifest (rewriting `session_dir`/`npz_path` to
absolute paths so the combined dataset can be moved), and writes a merged
`dataset_metadata.json` with per-gesture and per-collector counts.

Operates on *processed* datasets (the output of `extract_features.py cut`,
under `data/processed/`), not raw `collect.py` sessions -- cut each
collector's raw session first, then combine the processed datasets here.

Usage (run from the repo root):
    python src/combine_datasets.py \\
        data/processed/STUDENT_A \\
        data/processed/STUDENT_B \\
        --output data/processed/combined
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

from extract_features import INPUT_TYPE, MANIFEST_FIELDNAMES
from sensors.common import REPO_DIR, source_dataset_name, timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine multiple gesture dataset manifests.")
    parser.add_argument(
        "datasets",
        nargs="+",
        help="Processed dataset folders created by extract_features.py cut.",
    )
    parser.add_argument(
        "--output",
        default=str(REPO_DIR / "data" / "processed" / f"combined_{timestamp()}"),
        help="Output dataset folder. Default: data/processed/combined_<timestamp>",
    )
    parser.add_argument(
        "--collector",
        action="append",
        help="Keep only this collector. Can be repeated or comma separated.",
    )
    parser.add_argument(
        "--gesture",
        action="append",
        help="Keep only this gesture. Can be repeated or comma separated.",
    )
    parser.add_argument(
        "--allow-missing-session",
        action="store_true",
        help="Keep rows even if the referenced session_dir or npz_path does not exist.",
    )
    return parser.parse_args()


def read_rows(dataset_dir: Path) -> list[dict[str, str]]:
    manifest = dataset_dir / "trials.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"missing trials.csv: {manifest}")
    with manifest.open(newline="") as file:
        return list(csv.DictReader(file))


def normalize_filter(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    normalized: set[str] = set()
    for value in values:
        for part in value.split(","):
            label = part.strip()
            if label:
                normalized.add(label)
    return normalized


def resolve_path(dataset_dir: Path, value: str) -> Path:
    path = Path(value) if value else Path("")
    if path.is_absolute():
        return path
    return (dataset_dir / path).resolve()


def resolve_session_and_npz(dataset_dir: Path, row: dict[str, str]) -> tuple[Path, Path]:
    session_dir = resolve_path(dataset_dir, row.get("session_dir", ""))
    npz_value = row.get("npz_path", "")
    npz_path = resolve_path(dataset_dir, npz_value) if npz_value else session_dir / "trial_data.npz"
    return session_dir, npz_path


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_manifest = output_dir / "trials.csv"

    allowed_collectors = normalize_filter(args.collector)
    allowed_gestures = normalize_filter(args.gesture)
    source_dirs = [Path(item).expanduser().resolve() for item in args.datasets]
    combined: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    gesture_counts: dict[str, int] = {}
    collector_counts: dict[str, int] = {}
    sensor_combo_counts: dict[str, int] = {}

    for dataset_dir in source_dirs:
        try:
            rows = read_rows(dataset_dir)
        except FileNotFoundError as exc:
            skipped.append({"dataset": str(dataset_dir), "reason": str(exc)})
            continue

        source_name = source_dataset_name(dataset_dir)
        for row in rows:
            collector = row.get("collector", "")
            gesture = row.get("gesture", "")
            if allowed_collectors and collector not in allowed_collectors:
                continue
            if allowed_gestures and gesture not in allowed_gestures:
                continue

            session_dir, npz_path = resolve_session_and_npz(dataset_dir, row)
            if not args.allow_missing_session and (not session_dir.exists() or not npz_path.exists()):
                skipped.append(
                    {
                        "dataset": str(dataset_dir),
                        "session_dir": str(session_dir),
                        "npz_path": str(npz_path),
                        "reason": "missing session_dir or npz_path",
                    }
                )
                continue

            merged_row = {name: row.get(name, "") for name in MANIFEST_FIELDNAMES}
            merged_row["dataset_name"] = output_dir.name
            merged_row["collector"] = collector
            merged_row["gesture"] = gesture
            merged_row["input_type"] = row.get("input_type", INPUT_TYPE)
            merged_row["session_dir"] = str(session_dir)
            merged_row["npz_path"] = str(npz_path)
            combined.append(merged_row)

            gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1
            collector_counts[collector] = collector_counts.get(collector, 0) + 1
            sensors_enabled = row.get("sensors_enabled", "")
            sensor_combo_counts[sensors_enabled] = sensor_combo_counts.get(sensors_enabled, 0) + 1

        print(f"{source_name}: read {len(rows)} rows from {dataset_dir}")

    with output_manifest.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDNAMES)
        writer.writeheader()
        writer.writerows(combined)

    metadata = {
        "dataset_name": output_dir.name,
        "input_type": INPUT_TYPE,
        "combined_from": [str(path) for path in source_dirs],
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "row_count": len(combined),
        "gesture_counts": gesture_counts,
        "collector_counts": collector_counts,
        "sensor_combo_counts": sensor_combo_counts,
        "collector_filter": sorted(allowed_collectors) if allowed_collectors else None,
        "gesture_filter": sorted(allowed_gestures) if allowed_gestures else None,
        "skipped": skipped,
    }
    (output_dir / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    print(json.dumps(metadata, indent=2, sort_keys=True))
    if not combined:
        print("No rows were combined.", file=sys.stderr)
        return 1
    print(f"Combined manifest: {output_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
