#!/usr/bin/env python3
"""
Sanity-check raw collect.py session folders before cutting them.

extract_features.py cut() matches each data/raw/<session>/trials.csv row to a
trial_start/trial_end pair in that same folder's events.csv, by trial_id. If a
folder's trials.csv and events.csv don't come from the same recording run
(files got mixed up between sessions, a "part2" split grabbed the wrong
companion file, etc.), every trial in that folder silently gets skipped with
"missing trial_start/trial_end in events.csv" -- this script finds that before
you spend time cutting/combining/training on it.

Also flags session_metadata.json's "collector" field disagreeing with the
folder name, since that's the other easy-to-miss mistake (a copy-pasted
--collector flag at collection time). Not fatal on its own --
combine_datasets.py's `=name` override fixes it -- but worth knowing about
up front, and a session with BOTH problems at once is a strong sign its
files got shuffled with another student's.

Run from the repo root:
    python src/diagnose_sessions.py
    python src/diagnose_sessions.py data/raw/session_vibha_discrete   # one folder
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from sensors.common import DATA_RAW_DIR


def _read_trial_ids_from_trials_csv(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="") as file:
        return {row["trial_id"] for row in csv.DictReader(file) if row.get("trial_id")}


def _read_trial_ids_from_events_csv(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="") as file:
        return {
            row["trial_id"]
            for row in csv.DictReader(file)
            if row.get("event") in ("trial_start", "trial_end") and row.get("trial_id")
        }


def _guess_collector_from_folder_name(name: str) -> str | None:
    """`session_<collector>_<discrete|periodic>...` -- best-effort, not authoritative."""
    match = re.match(r"session_([^_]+)_(discrete|periodic)", name)
    return match.group(1) if match else None


def diagnose_session(session_dir: Path) -> dict:
    trial_ids = _read_trial_ids_from_trials_csv(session_dir / "trials.csv")
    event_trial_ids = _read_trial_ids_from_events_csv(session_dir / "events.csv")
    missing = trial_ids - event_trial_ids

    metadata_path = session_dir / "session_metadata.json"
    collector = json.loads(metadata_path.read_text()).get("collector") if metadata_path.exists() else None
    expected_collector = _guess_collector_from_folder_name(session_dir.name)

    return {
        "session": session_dir.name,
        "trial_count": len(trial_ids),
        "matched": len(trial_ids) - len(missing),
        "missing_trial_ids": sorted(missing),
        "collector": collector,
        "expected_collector": expected_collector,
        "collector_mismatch": bool(expected_collector and collector and collector != expected_collector),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "sessions", nargs="*", help="Specific session folders to check. Default: every data/raw/session_*."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    session_dirs = [Path(s) for s in args.sessions] if args.sessions else sorted(DATA_RAW_DIR.glob("session_*"))
    if not session_dirs:
        raise SystemExit(f"No session folders found under {DATA_RAW_DIR}.")

    any_problem = False
    for session_dir in session_dirs:
        if not session_dir.is_dir():
            print(f"{session_dir.name}: not a directory, skipping")
            continue
        result = diagnose_session(session_dir)
        missing_count = len(result["missing_trial_ids"])

        flags = []
        if result["trial_count"] == 0:
            flags.append("NO TRIALS in trials.csv")
        elif missing_count == result["trial_count"]:
            flags.append("ALL TRIALS MISMATCHED -- trials.csv/events.csv are from different runs")
        elif missing_count > 0:
            flags.append(f"{missing_count}/{result['trial_count']} trials mismatched")
        if result["collector_mismatch"]:
            flags.append(f"collector='{result['collector']}' but folder name implies '{result['expected_collector']}'")

        status = "OK" if not flags else " | ".join(flags)
        print(f"{result['session']:35s} trials={result['trial_count']:3d} matched={result['matched']:3d}  {status}")
        if 0 < missing_count < result["trial_count"]:
            preview = ", ".join(result["missing_trial_ids"][:10])
            more = " ..." if missing_count > 10 else ""
            print(f"    missing trial_ids: {preview}{more}")
        if flags:
            any_problem = True

    return 1 if any_problem else 0


if __name__ == "__main__":
    raise SystemExit(main())
