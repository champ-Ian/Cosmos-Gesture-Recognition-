#!/usr/bin/env python3
"""
Train the raw-sequence 1D-CNN (`gesture_models.TorchRawCNNClassifier`) from a
feature table that includes resampled raw per-channel sequences alongside the
usual hand-crafted summary stats -- e.g. `raw_features.csv`, with columns like
`imu_raw_ax_0..19`, `mmwave_raw_energy_0..19`, `uwb_raw_distance_0..19` (each
channel resampled to the same fixed length) plus the normal scalar columns
`mmwave_energy_mean`, `imu_ax_std`, etc.

Unlike `train.py --classifier cnn` (which treats one flat hand-crafted vector
as a single-channel signal -- there's no real time axis there), this convolves
over the actual resampled time axis of each raw channel, which is what
actually gets you the "CNN learns temporal patterns" benefit. The scalar
columns are still used, through a small side-branch that gets concatenated in
before the final classifier layer.

Not currently usable from `realtime_demo.py` -- that script's live windowing
only produces the flat scalar feature vector, not resampled raw sequences.

Run from the repo root as `python src/train_cnn_raw.py ...`:

    python src/train_cnn_raw.py src/raw_features.csv
    python src/train_cnn_raw.py src/raw_features.csv --test-collector student03
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

from gesture_models import TorchRawCNNClassifier, classifier_label
from sensors.common import MODELS_DIR, RESULTS_FIGURES_DIR, timestamp
from train import split_indices

RAW_COLUMN_RE = re.compile(r"^(mmwave|imu|uwb|rfid)_raw_([a-z]+)_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("features_csv", help="Feature table with raw_* resampled-sequence columns, e.g. raw_features.csv.")
    parser.add_argument("--cnn-epochs", type=int, default=200)
    parser.add_argument("--cnn-lr", type=float, default=1e-3)
    parser.add_argument("--cnn-hidden-channels", type=int, default=32)
    parser.add_argument("--cnn-dropout", type=float, default=0.3)
    parser.add_argument("--cnn-batch-size", type=int, default=16)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--test-collector",
        action="append",
        help="Hold out this collector as the test set (all others train). Repeatable/comma separated.",
    )
    parser.add_argument("--model-out", help="Output .joblib path. Default: <repo>/models/cnn_raw_<timestamp>.joblib")
    parser.add_argument("--confusion-out", help="Output confusion matrix PNG. Default: <repo>/results/figures/.")
    return parser.parse_args()


def read_raw_features_csv(path: Path):
    """Split a feature table's columns into scalar columns and raw per-channel
    sequence columns (grouped by `{sensor}_raw_{field}_{index}` name, regardless
    of where they fall in the header), and return per-row arrays for each."""
    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise SystemExit(f"{path} is empty.")
        meta_cols, feature_cols = header[:4], header[4:]
        if meta_cols != ["gesture", "collector", "source_dataset", "session_dir"]:
            raise SystemExit(
                f"{path} doesn't look like an export_features_csv.py-style output "
                f"(expected first columns gesture,collector,source_dataset,session_dir, got {meta_cols})."
            )

        scalar_col_indices: list[int] = []
        scalar_names: list[str] = []
        channel_col_indices: dict[str, list[tuple[int, int]]] = {}  # channel_key -> [(step_index, col_index), ...]

        for i, name in enumerate(feature_cols):
            match = RAW_COLUMN_RE.match(name)
            if match:
                sensor, field, step = match.group(1), match.group(2), int(match.group(3))
                channel_key = f"{sensor}_{field}"
                channel_col_indices.setdefault(channel_key, []).append((step, i))
            else:
                scalar_col_indices.append(i)
                scalar_names.append(name)

        if not channel_col_indices:
            raise SystemExit(
                f"{path} has no raw_* sequence columns (expected names like imu_raw_ax_0). "
                "Use train.py --features-csv for a plain summary-stat feature table instead."
            )

        channel_names = sorted(channel_col_indices.keys())
        raw_length = len(channel_col_indices[channel_names[0]])
        for channel_key in channel_names:
            steps = sorted(channel_col_indices[channel_key])
            if len(steps) != raw_length or [s for s, _ in steps] != list(range(raw_length)):
                raise SystemExit(
                    f"Channel '{channel_key}' in {path} doesn't have a clean 0..{raw_length - 1} step "
                    f"sequence (got steps {[s for s, _ in steps]}). Every raw channel must be resampled to "
                    "the same fixed length."
                )
            channel_col_indices[channel_key] = [i for _, i in steps]

        labels: list[str] = []
        collectors: list[str] = []
        sources: list[str] = []
        session_dirs: list[str] = []
        scalar_rows: list[list[float]] = []
        raw_rows: list[list[list[float]]] = []

        for row in reader:
            if not row:
                continue
            gesture, collector, source, session_dir = row[:4]
            values = row[4:]
            scalar_rows.append([float(values[i]) for i in scalar_col_indices])
            raw_rows.append([[float(values[i]) for i in channel_col_indices[channel_key]] for channel_key in channel_names])
            labels.append(gesture)
            collectors.append(collector)
            sources.append(source)
            session_dirs.append(session_dir)

    if not labels:
        raise SystemExit(f"No rows found in {path}.")

    return {
        "scalar_names": scalar_names,
        "channel_names": channel_names,
        "raw_length": raw_length,
        "X_scalar": np.asarray(scalar_rows, dtype=float),
        "X_raw": np.asarray(raw_rows, dtype=float),  # (N, n_channels, raw_length)
        "labels": labels,
        "collectors": collectors,
        "sources": sources,
        "session_dirs": session_dirs,
    }


def main() -> int:
    args = parse_args()
    features_csv_path = Path(args.features_csv).expanduser().resolve()
    data = read_raw_features_csv(features_csv_path)

    y = np.asarray(data["labels"])
    collectors_array = np.asarray(data["collectors"])
    if len(set(data["labels"])) < 2:
        raise SystemExit("Need at least two gesture classes to train.")
    if len(data["labels"]) < 4:
        raise SystemExit("Need at least four usable rows to train.")
    class_counts = {label: int((y == label).sum()) for label in sorted(set(data["labels"]))}

    train_idx, test_idx, split_method = split_indices(y, collectors_array, args)
    y_train, y_test = y[train_idx], y[test_idx]
    train_only_check = set(y_test) - set(y_train)
    if train_only_check:
        raise SystemExit("The test set contains gesture labels not present in training: " + ", ".join(sorted(train_only_check)))
    if len(set(y_train)) < 2:
        raise SystemExit("Training split needs at least two gesture classes.")
    train_collectors = sorted(set(collectors_array[train_idx]))
    test_collectors_used = sorted(set(collectors_array[test_idx]))

    X_raw_train, X_raw_test = data["X_raw"][train_idx], data["X_raw"][test_idx]
    X_scalar_train, X_scalar_test = data["X_scalar"][train_idx], data["X_scalar"][test_idx]

    try:
        import joblib
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
    except ImportError as exc:
        raise SystemExit(f"Training dependencies missing: {exc}")

    requested_batch_size = max(1, int(args.cnn_batch_size))
    actual_batch_size = min(requested_batch_size, len(X_raw_train))
    if actual_batch_size < requested_batch_size:
        print(f"Reducing CNN batch size from {requested_batch_size} to {actual_batch_size} because the training split has {len(X_raw_train)} examples.")

    classifier_params = {
        "epochs": args.cnn_epochs,
        "lr": args.cnn_lr,
        "hidden_channels": args.cnn_hidden_channels,
        "dropout": args.cnn_dropout,
        "batch_size": actual_batch_size,
        "random_state": args.random_state,
    }
    model = TorchRawCNNClassifier(**classifier_params)
    model.fit(X_raw_train, X_scalar_train, y_train)
    predictions = model.predict(X_raw_test, X_scalar_test)

    accuracy = float(accuracy_score(y_test, predictions))
    labels_order = sorted(set(y))
    matrix = confusion_matrix(y_test, predictions, labels=labels_order)

    model_out = Path(args.model_out) if args.model_out else MODELS_DIR / f"cnn_raw_{timestamp()}.joblib"
    model_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "input_type": "cosmos_multi_sensor_gesture_raw",
        "classifier": "cnn_raw",
        "classifier_label": classifier_label("cnn_raw"),
        "classifier_params": classifier_params,
        "scalar_names": data["scalar_names"],
        "channel_names": data["channel_names"],
        "raw_length": data["raw_length"],
        "labels": labels_order,
        "training_datasets": [str(features_csv_path)],
        "split_method": split_method,
        "test_collectors": test_collectors_used,
    }
    joblib.dump(payload, model_out)

    confusion_out = Path(args.confusion_out) if args.confusion_out else RESULTS_FIGURES_DIR / (model_out.stem + "_confusion_matrix.png")
    confusion_out.parent.mkdir(parents=True, exist_ok=True)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels_order)
    fig, ax = plt.subplots(figsize=(6, 5))
    display.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
    ax.set_title(f"{classifier_label('cnn_raw')} (raw+scalar)\nAccuracy: {accuracy:.3f}")
    fig.tight_layout()
    fig.savefig(confusion_out, dpi=180)
    plt.close(fig)

    per_class_recall = {
        label: float(matrix[i, i] / matrix[i].sum()) if matrix[i].sum() else None
        for i, label in enumerate(labels_order)
    }
    summary = {
        "features_csv": str(features_csv_path),
        "channel_names": data["channel_names"],
        "raw_length": data["raw_length"],
        "n_scalar_features": len(data["scalar_names"]),
        "classifier": "cnn_raw",
        "classifier_label": classifier_label("cnn_raw"),
        "classifier_params": classifier_params,
        "model": str(model_out),
        "confusion_matrix": str(confusion_out),
        "accuracy": accuracy,
        "per_class_recall": per_class_recall,
        "class_counts": class_counts,
        "n_examples": int(len(data["labels"])),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "split_method": split_method,
        "train_collectors": train_collectors,
        "test_collectors": test_collectors_used,
        "classification_report": classification_report(y_test, predictions, labels=labels_order, zero_division=0, output_dict=True),
    }
    summary_path = model_out.with_name(model_out.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
