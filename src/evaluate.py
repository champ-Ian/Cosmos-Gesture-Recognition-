#!/usr/bin/env python3
"""
Evaluate an already-trained model (from `train.py`) against a dataset,
without retraining.

Useful for:
    - Testing on data collected AFTER training (e.g. a new collector's
      session, to check generalization on a genuinely held-out person).
    - Regenerating a confusion matrix / classification report without
      rerunning `train.py`.

Uses the model's own `sensors`/`fusion` config (stored in the `.joblib`
payload) to know which sensors to extract features for and how to combine
them -- you don't need to re-specify `--sensors`/`--fusion`.

Usage:

    python evaluate.py --model models/knn_early_mmwave-imu_....joblib \\
        data/processed/combined

    # Held-out-person check: only evaluate one collector's trials
    python evaluate.py --model models/knn_early_mmwave-imu_....joblib \\
        data/processed/combined --test-collector student04
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from train import build_examples, normalize_filter, read_manifests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", nargs="+", help="Processed dataset folders to evaluate against.")
    parser.add_argument("--model", required=True, help="Trained model .joblib from train.py.")
    parser.add_argument(
        "--test-collector",
        action="append",
        help="Only evaluate trials from this collector (e.g. a held-out person). Default: evaluate on all trials.",
    )
    parser.add_argument("--confusion-out", help="Output confusion matrix PNG. Default: next to --model.")
    parser.add_argument("--summary-out", help="Output summary JSON. Default: next to --model.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        import joblib
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
    except ImportError as exc:
        raise SystemExit(f"Evaluation dependencies missing: {exc}")

    model_path = Path(args.model)
    payload = joblib.load(model_path)
    model = payload["model"]
    sensors = payload["sensors"]
    fusion = payload.get("fusion", "early")
    classifier = payload.get("classifier", "unknown")
    classifier_label = payload.get("classifier_label", classifier)
    trained_labels = payload.get("labels", [])

    dataset_dirs = [Path(item).expanduser().resolve() for item in args.datasets]
    rows, missing_datasets = read_manifests(dataset_dirs)
    if not rows:
        raise SystemExit("No trials found. Expected trials.csv or trial_metadata.json files.")

    per_sensor_examples, labels, collectors, sources, session_dirs, skipped = build_examples(rows, sensors)
    if not labels:
        raise SystemExit(f"No usable trials for this model's sensors ({', '.join(sensors)}).")

    test_collectors = normalize_filter(args.test_collector)
    if test_collectors:
        mask = np.array([collector in test_collectors for collector in collectors], dtype=bool)
        if not mask.any():
            raise SystemExit(
                "No trials matched --test-collector. Available collectors: " + ", ".join(sorted(set(collectors)))
            )
    else:
        mask = np.ones(len(labels), dtype=bool)

    y = np.asarray(labels)[mask]
    collectors_used = sorted(set(np.asarray(collectors)[mask]))
    per_sensor_X = {sensor: np.asarray(vectors, dtype=float)[mask] for sensor, vectors in per_sensor_examples.items()}

    unseen_labels = sorted(set(y) - set(trained_labels)) if trained_labels else []
    if unseen_labels:
        print(
            f"Warning: evaluation data has gesture labels the model never saw during training: "
            f"{', '.join(unseen_labels)}. Predictions for these will always be wrong."
        )

    if fusion == "early":
        X = np.hstack([per_sensor_X[sensor] for sensor in sensors])
        predictions = model.predict(X)
    else:
        predictions = model.predict(per_sensor_X)

    accuracy = float(accuracy_score(y, predictions))
    labels_order = sorted(set(y) | set(trained_labels))
    matrix = confusion_matrix(y, predictions, labels=labels_order)
    per_class_recall = {
        label: float(matrix[i, i] / matrix[i].sum()) if matrix[i].sum() else None
        for i, label in enumerate(labels_order)
    }

    confusion_out = Path(args.confusion_out) if args.confusion_out else model_path.with_name(
        model_path.stem + "_eval_confusion_matrix.png"
    )
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels_order)
    fig, ax = plt.subplots(figsize=(6, 5))
    display.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
    ax.set_title(f"{classifier_label} ({fusion} fusion: {'+'.join(sensors)})\nEval accuracy: {accuracy:.3f}")
    fig.tight_layout()
    fig.savefig(confusion_out, dpi=180)
    plt.close(fig)

    summary = {
        "model": str(model_path),
        "datasets": [str(path) for path in dataset_dirs],
        "sensors": sensors,
        "fusion": fusion,
        "classifier": classifier,
        "classifier_label": classifier_label,
        "accuracy": accuracy,
        "per_class_recall": per_class_recall,
        "n_examples": int(len(y)),
        "test_collectors": collectors_used if test_collectors else "all",
        "unseen_labels": unseen_labels,
        "confusion_matrix": str(confusion_out),
        "source_datasets": sorted(set(sources)),
        "session_dirs": session_dirs,
        "missing_datasets": missing_datasets,
        "skipped": skipped,
        "classification_report": classification_report(
            y, predictions, labels=labels_order, zero_division=0, output_dict=True
        ),
    }
    summary_out = Path(args.summary_out) if args.summary_out else model_path.with_name(
        model_path.stem + "_eval_summary.json"
    )
    summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
