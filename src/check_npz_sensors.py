#!/usr/bin/env python3
"""
Find trials whose trials.csv `sensors_enabled` column disagrees with what's
actually in that trial's trial_data.npz -- e.g. a row claims "mmwave" but the
npz has no mmwave_frame_number key at all (the raw session's mmwave.npz was
probably missing/empty when `extract_features.py cut` ran).

extract_features.py's extractors now return None (skip, with a reason)
instead of crashing on this, but that just silently drops the trial -- run
this to see exactly which raw session/trial is the actual root cause.

Run from the repo root:
    python src/check_npz_sensors.py data/processed/combined
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

MARKER_KEY = {
    "mmwave": "mmwave_frame_number",
    "imu": "imu_raw_lines",
    "uwb": "uwb_status",
    "rfid": "rfid_raw_lines",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path, help="Processed dataset folder (has trials.csv).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = args.dataset / "trials.csv"
    if not manifest.exists():
        raise SystemExit(f"No trials.csv under {args.dataset}")

    with manifest.open(newline="") as file:
        rows = list(csv.DictReader(file))

    problems = 0
    for row in rows:
        sensors_enabled = {s.strip() for s in row.get("sensors_enabled", "").split(",") if s.strip()}
        npz_path = Path(row.get("npz_path", ""))
        if not npz_path.exists():
            print(f"MISSING npz: {row.get('session_dir')} ({row.get('gesture')})")
            problems += 1
            continue
        with np.load(npz_path, allow_pickle=True) as npz:
            for sensor in sensors_enabled:
                marker = MARKER_KEY.get(sensor)
                if marker and marker not in npz.files:
                    print(
                        f"{row.get('session_dir')}: sensors_enabled claims '{sensor}' "
                        f"but npz has no '{marker}' key (gesture={row.get('gesture')}, "
                        f"trial_index={row.get('trial_index')}, source={row.get('dataset_name')})"
                    )
                    problems += 1

    print(f"\n{problems} problem trial(s) out of {len(rows)}." if problems else f"All {len(rows)} trials OK.")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
