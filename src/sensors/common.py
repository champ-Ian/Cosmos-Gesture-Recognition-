#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_DIR = Path(__file__).resolve().parent.parent


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def safe_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return label.strip("_") or "unknown"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def read_manifest(dataset_dir: Path) -> list[dict[str, str]]:
    manifest = dataset_dir / "trials.csv"
    if not manifest.exists():
        return []
    with manifest.open(newline="") as file:
        return list(csv.DictReader(file))


def source_dataset_name(dataset_dir: Path) -> str:
    metadata_path = dataset_dir / "dataset_metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text())
            return str(metadata.get("dataset_name") or dataset_dir.name)
        except json.JSONDecodeError:
            pass
    return dataset_dir.name


def append_manifest(manifest_path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    exists = manifest_path.exists()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in fieldnames})
