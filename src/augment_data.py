#!/usr/bin/env python3
"""
Data augmentation: create synthetic extra training trials from real ones by
applying small, realistic perturbations directly to each sensor's raw
per-trial arrays -- the exact same `trial_data.npz` shape `extract_features.py
cut` produces, so every existing downstream script (export_features_csv.py,
train.py) needs zero changes to use the augmented data; it just looks like
more real trials with the same gesture/collector labels.

Two perturbations, applied together per synthetic copy:
  - Gaussian noise: added to each continuous channel, scaled to a fraction of
    THAT channel's own std within the trial (so a low-variance channel isn't
    drowned out and a high-variance one isn't barely nudged).
  - Time shift: circularly rolls each channel's *values* by a small random
    sample offset (timestamps stay fixed) -- simulates the same gesture
    happening slightly earlier/later within the recording window.

IMU/RFID are stored as raw firmware/reader log LINES, not parsed numbers, so
those get parsed, perturbed, then re-serialized back into the same log-line
text format extract_features.py's regex parsers expect -- to every
downstream step (including the outlier-removal/smoothing in
extract_features.py's clean_signal()), an augmented trial is indistinguishable
from a real, slightly-noisier recording.

Run from the repo root:
    python src/augment_data.py data/processed/combined --output data/processed/combined_augmented --num-augments 3
    python src/export_features_csv.py data/processed/combined_augmented --sensors mmwave,imu,uwb --output src/features_augmented.csv
    python src/export_features_csv.py data/processed/combined_augmented --sensors mmwave,imu,uwb --feature-mode raw --output data/processed/processed_signals_augmented.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np

from extract_features import MANIFEST_FIELDNAMES, parse_imu_line, parse_rfid_line
from sensors.common import timestamp

SENSOR_MARKER_KEY = {
    "mmwave": "mmwave_frame_number",
    "imu": "imu_raw_lines",
    "uwb": "uwb_status",
    "rfid": "rfid_raw_lines",
}


def _roll(values: np.ndarray, shift: int) -> np.ndarray:
    """Circular-shift values along the sample axis; a no-op for <2 samples or shift=0."""
    if len(values) < 2 or shift == 0:
        return values.copy()
    return np.roll(values, shift, axis=0)


def _add_noise(rng: np.random.Generator, values: np.ndarray, noise_std_fraction: float) -> np.ndarray:
    """Gaussian noise scaled to a fraction of this channel's own std (or its magnitude, if flat)."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values.copy()
    spread = float(np.std(values)) if values.size > 1 else 0.0
    scale = noise_std_fraction * spread
    if scale == 0:
        # Flat/near-constant channel -- still perturb a little rather than leaving it untouched.
        magnitude = float(np.mean(np.abs(values))) or 1.0
        scale = noise_std_fraction * magnitude
    return values + rng.normal(0.0, scale, size=values.shape)


def _random_shift(rng: np.random.Generator, n: int, max_shift_fraction: float) -> int:
    if n < 2:
        return 0
    max_shift = max(1, int(round(n * max_shift_fraction)))
    return int(rng.integers(-max_shift, max_shift + 1))


def augment_mmwave(rng: np.random.Generator, payload: dict, noise_std_fraction: float, max_shift_fraction: float) -> None:
    n = len(payload["mmwave_frame_number"])
    if n == 0:
        return
    shift = _random_shift(rng, n, max_shift_fraction)

    profile = payload["mmwave_range_profile"]
    if profile.size:
        payload["mmwave_range_profile"] = _roll(_add_noise(rng, profile, noise_std_fraction), shift)

    point_count = payload["mmwave_point_count"]
    if point_count.size:
        payload["mmwave_point_count"] = np.clip(_roll(_add_noise(rng, point_count, noise_std_fraction), shift), 0, None)

    velocity = payload["mmwave_points_velocity"]
    if velocity.size:
        payload["mmwave_points_velocity"] = _add_noise(rng, velocity, noise_std_fraction)


def augment_uwb(rng: np.random.Generator, payload: dict, noise_std_fraction: float, max_shift_fraction: float) -> None:
    n = len(payload["uwb_status"])
    if n == 0:
        return
    shift = _random_shift(rng, n, max_shift_fraction)
    noisy = _add_noise(rng, payload["uwb_distance_cm"], noise_std_fraction)
    payload["uwb_distance_cm"] = _roll(noisy, shift)
    payload["uwb_status"] = _roll(payload["uwb_status"], shift)


def augment_imu(rng: np.random.Generator, payload: dict, noise_std_fraction: float, max_shift_fraction: float) -> None:
    raw_lines = payload["imu_raw_lines"]
    if len(raw_lines) == 0:
        return
    parsed = [parse_imu_line(line) for line in raw_lines]
    valid_idx = [i for i, p in enumerate(parsed) if p is not None]
    if len(valid_idx) < 2:
        return

    samples = np.array([parsed[i] for i in valid_idx], dtype=float)  # (m, 6): ax,ay,az,gx,gy,gz
    for axis in range(6):
        samples[:, axis] = _add_noise(rng, samples[:, axis], noise_std_fraction)
    shift = _random_shift(rng, len(valid_idx), max_shift_fraction)
    samples = _roll(samples, shift)

    new_lines = list(raw_lines)
    for pos, i in enumerate(valid_idx):
        ax, ay, az, gx, gy, gz = samples[pos]
        new_lines[i] = (
            f"accel[g] x={ax:.4f} y={ay:.4f} z={az:.4f} | gyro[dps] x={gx:.4f} y={gy:.4f} z={gz:.4f}"
        )
    payload["imu_raw_lines"] = np.array(new_lines, dtype=object)


def augment_rfid(rng: np.random.Generator, payload: dict, noise_std_fraction: float, max_shift_fraction: float) -> None:
    raw_lines = payload["rfid_raw_lines"]
    if len(raw_lines) == 0:
        return
    parsed = [parse_rfid_line(line) for line in raw_lines]
    valid_idx = [i for i, p in enumerate(parsed) if p is not None]
    if len(valid_idx) < 2:
        return

    rssi = np.array([parsed[i][1] for i in valid_idx], dtype=float)
    rssi = _add_noise(rng, rssi, noise_std_fraction)
    shift = _random_shift(rng, len(valid_idx), max_shift_fraction)
    rssi = _roll(rssi, shift)

    new_lines = list(raw_lines)
    for pos, i in enumerate(valid_idx):
        epc, _, read_count = parsed[i]
        new_lines[i] = f"{epc} 2024-01-01 00:00:00.000 {int(round(rssi[pos]))} {read_count}"
    payload["rfid_raw_lines"] = np.array(new_lines, dtype=object)


AUGMENTERS = {
    "mmwave": augment_mmwave,
    "imu": augment_imu,
    "uwb": augment_uwb,
    "rfid": augment_rfid,
}


def augment_trial(
    rng: np.random.Generator, npz_path: Path, out_path: Path, noise_std_fraction: float, max_shift_fraction: float
) -> None:
    with np.load(npz_path, allow_pickle=True) as npz:
        payload = {key: npz[key] for key in npz.files}

    for sensor, marker in SENSOR_MARKER_KEY.items():
        if marker in payload:
            AUGMENTERS[sensor](rng, payload, noise_std_fraction, max_shift_fraction)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path, help="Processed dataset folder (has trials.csv) to augment.")
    parser.add_argument("--output", type=Path, required=True, help="Output folder: original trials + synthetic copies.")
    parser.add_argument("--num-augments", type=int, default=3, help="Synthetic copies to generate per real trial.")
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.05,
        help="Gaussian noise std, as a fraction of each channel's own std within the trial.",
    )
    parser.add_argument(
        "--max-shift-fraction",
        type=float,
        default=0.1,
        help="Max time-shift, as a fraction of the trial's sample count.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    manifest_path = args.dataset / "trials.csv"
    if not manifest_path.exists():
        raise SystemExit(f"No trials.csv under {args.dataset}")
    with manifest_path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise SystemExit(f"No trials found in {manifest_path}")

    args.output.mkdir(parents=True, exist_ok=True)
    out_rows: list[dict] = []

    for row in rows:
        src_npz = Path(row["npz_path"])
        if not src_npz.exists():
            continue
        trial_name = Path(row["session_dir"]).name

        # Keep the original trial byte-for-byte alongside the synthetic copies.
        dst_session_dir = args.output / "sessions" / trial_name
        dst_npz = dst_session_dir / "trial_data.npz"
        dst_session_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_npz, dst_npz)
        original_row = dict(row)
        original_row["npz_path"] = str(dst_npz.resolve())
        original_row["session_dir"] = str(dst_session_dir.resolve())
        out_rows.append(original_row)

        for aug_index in range(args.num_augments):
            aug_session_dir = args.output / "sessions" / f"{trial_name}_aug{aug_index:02d}"
            aug_npz = aug_session_dir / "trial_data.npz"
            augment_trial(rng, src_npz, aug_npz, args.noise_std, args.max_shift_fraction)

            aug_row = dict(row)
            aug_row["npz_path"] = str(aug_npz.resolve())
            aug_row["session_dir"] = str(aug_session_dir.resolve())
            out_rows.append(aug_row)

    manifest_out = args.output / "trials.csv"
    with manifest_out.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDNAMES)
        writer.writeheader()
        writer.writerows(out_rows)

    metadata = {
        "dataset_name": args.output.name,
        "source_dataset": str(args.dataset),
        "real_trial_count": len(rows),
        "synthetic_trial_count": len(rows) * args.num_augments,
        "row_count": len(out_rows),
        "num_augments": args.num_augments,
        "noise_std_fraction": args.noise_std,
        "max_shift_fraction": args.max_shift_fraction,
        "seed": args.seed,
        "created_at": timestamp(),
    }
    (args.output / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    print(
        f"Wrote {len(out_rows)} trial rows ({len(rows)} real + {len(rows) * args.num_augments} synthetic) "
        f"to {manifest_out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
