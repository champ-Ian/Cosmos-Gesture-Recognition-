#!/usr/bin/env python3
"""
Leave-one-subject(collector)-out sweep: hold out each collector in turn, train
on the other nine, predict on the held-out one, and pool every held-out
prediction across all ten collectors into one confusion matrix / accuracy
number.

`train.py` only supports a single train/test split (or a single
`--test-collector` holdout) -- it does not itself rotate through every
collector and pool the results. This script reuses `train.py`'s own
`read_features_csv` and `gesture_models.py`'s `build_classifier` /
`LateFusionClassifier` unchanged, so a given fold's model is exactly what
`train.py` would build; the only new part is the "loop over every collector as
the held-out fold" aggregation.

This is the methodology behind the paper's Model Comparison and Modality
Ablation tables (leave-one-subject-out across all ten collectors on
features_augmented.csv). `--ablation` reproduces every row of the Modality
Ablation table; `--model-comparison` reproduces the Random Forest / kNN / CNN
numbers in Model Comparison.

Usage (run from the repo root):
    python src/loso_sweep.py --sensors mmwave,imu,uwb --fusion early --classifier random_forest
    python src/loso_sweep.py --sensors mmwave --fusion early --classifier random_forest
    python src/loso_sweep.py --ablation
    python src/loso_sweep.py --model-comparison
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from train import read_features_csv
from gesture_models import build_classifier, LateFusionClassifier

DEFAULT_FEATURES_CSV = Path(__file__).parent / "features_augmented.csv"
ALL_SENSORS = ["mmwave", "imu", "uwb"]
RANDOM_STATE = 42

# Every row of the paper's Modality Ablation table (sensors, fusion, classifier).
# Single-sensor rows use "early" since early/late fusion are equivalent with one sensor.
ABLATION_ROWS = [
    (["mmwave"], "early"),
    (["uwb"], "early"),
    (["imu"], "early"),
    (["mmwave", "uwb"], "late"),
    (["mmwave", "uwb"], "early"),
    (["mmwave", "imu"], "late"),
    (["mmwave", "imu"], "early"),
    (["imu", "uwb"], "late"),
    (["imu", "uwb"], "early"),
    (ALL_SENSORS, "late"),
    (ALL_SENSORS, "early"),
]

# The paper's Model Comparison numbers: RF early/late (isolates fusion strategy),
# kNN and the feature-vector CNN under early fusion.
MODEL_COMPARISON_ROWS = [
    ("random_forest", "early"),
    ("random_forest", "late"),
    ("knn", "early"),
    ("cnn", "early"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features-csv", default=str(DEFAULT_FEATURES_CSV), help="Flat feature table from export_features_csv.py.")
    parser.add_argument("--sensors", help="Comma-separated sensor subset, e.g. 'mmwave' or 'mmwave,imu,uwb'.")
    parser.add_argument("--fusion", choices=["early", "late"], default="early")
    parser.add_argument("--classifier", choices=["random_forest", "knn", "cnn"], default="random_forest")
    parser.add_argument("--ablation", action="store_true", help="Run every row of the Modality Ablation table.")
    parser.add_argument("--model-comparison", action="store_true", help="Run the Model Comparison table (RF/kNN/CNN).")
    parser.add_argument("--output", help="Optional path to write results as JSON.")
    return parser.parse_args()


def build_model(classifier: str, train_count: int):
    return build_classifier(
        classifier, train_count, RANDOM_STATE,
        1.0,        # svm_c (unused for random_forest/knn/cnn)
        5, "distance",  # knn_neighbors, knn_weights
        200, 1e-3, 32, 0.3, 16,  # cnn_epochs, cnn_lr, cnn_hidden_channels, cnn_dropout, cnn_batch_size
        200, None,  # rf_n_estimators, rf_max_depth
    )


def run_loso(sensors: list[str], fusion: str, classifier: str, per_sensor_X: dict, y: np.ndarray, collectors: np.ndarray, labels_order: list[str]):
    """One full LOSO sweep for one (sensors, fusion, classifier) config.

    Holds out each unique collector in turn, trains on the rest, predicts on the
    held-out collector, and pools every prediction into one confusion matrix over
    all examples (each predicted exactly once, since every collector is held out
    exactly once) -- this pooled aggregate is what the paper reports as accuracy
    "under leave-one-subject-out cross-validation across all ten collectors."
    """
    unique_collectors = sorted(set(collectors.tolist()))
    all_true: list[str] = []
    all_pred: list[str] = []

    for held_out in unique_collectors:
        test_mask = collectors == held_out
        train_mask = ~test_mask
        y_train, y_test = y[train_mask], y[test_mask]

        if fusion == "early":
            X_train = np.hstack([per_sensor_X[s][train_mask] for s in sensors])
            X_test = np.hstack([per_sensor_X[s][test_mask] for s in sensors])
            model, _ = build_model(classifier, len(X_train))
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
        else:
            sensor_models = {}
            for sensor in sensors:
                sub_model, _ = build_model(classifier, int(train_mask.sum()))
                sub_model.fit(per_sensor_X[sensor][train_mask], y_train)
                sensor_models[sensor] = sub_model
            model = LateFusionClassifier(sensor_models, sensors)
            preds = model.predict({s: per_sensor_X[s][test_mask] for s in sensors})

        all_true.extend(y_test.tolist())
        all_pred.extend(preds.tolist())

    from sklearn.metrics import accuracy_score, confusion_matrix

    accuracy = float(accuracy_score(all_true, all_pred))
    matrix = confusion_matrix(all_true, all_pred, labels=labels_order)
    return accuracy, matrix, len(all_true)


def load_data(features_csv: Path):
    per_sensor_examples, labels, collectors, sources, session_dirs = read_features_csv(
        features_csv, ALL_SENSORS, "summary"
    )
    per_sensor_X = {s: np.asarray(v, dtype=float) for s, v in per_sensor_examples.items()}
    y = np.asarray(labels)
    collectors_arr = np.asarray(collectors)
    labels_order = sorted(set(labels))
    return per_sensor_X, y, collectors_arr, labels_order


def main() -> int:
    args = parse_args()
    features_csv = Path(args.features_csv).expanduser().resolve()
    per_sensor_X, y, collectors_arr, labels_order = load_data(features_csv)
    print(f"Loaded {len(y)} rows, {len(labels_order)} classes, {len(set(collectors_arr))} collectors from {features_csv}")

    results: dict[str, dict] = {}

    if args.ablation:
        print("\n=== Modality Ablation ===")
        for sensors, fusion in ABLATION_ROWS:
            acc, matrix, n = run_loso(sensors, fusion, "random_forest", per_sensor_X, y, collectors_arr, labels_order)
            key = f"{'+'.join(sensors)}_{fusion}"
            print(f"{'+'.join(sensors):20s} {fusion:6s} accuracy = {acc*100:.2f}%  n={n}")
            results[key] = {"sensors": sensors, "fusion": fusion, "classifier": "random_forest", "accuracy": acc, "n": n, "matrix": matrix.tolist()}

    if args.model_comparison:
        print("\n=== Model Comparison ===")
        for classifier, fusion in MODEL_COMPARISON_ROWS:
            acc, matrix, n = run_loso(ALL_SENSORS, fusion, classifier, per_sensor_X, y, collectors_arr, labels_order)
            key = f"{classifier}_{fusion}"
            print(f"{classifier:15s} {fusion:6s} accuracy = {acc*100:.2f}%  n={n}")
            results[key] = {"sensors": ALL_SENSORS, "fusion": fusion, "classifier": classifier, "accuracy": acc, "n": n, "matrix": matrix.tolist()}

    if not args.ablation and not args.model_comparison:
        if not args.sensors:
            raise SystemExit("Pass --sensors (or use --ablation / --model-comparison).")
        sensors = [s.strip().lower() for s in args.sensors.split(",") if s.strip()]
        acc, matrix, n = run_loso(sensors, args.fusion, args.classifier, per_sensor_X, y, collectors_arr, labels_order)
        print(f"\n{'+'.join(sensors)}  {args.fusion}  {args.classifier}  accuracy = {acc*100:.2f}%  n={n}")
        results["result"] = {"sensors": sensors, "fusion": args.fusion, "classifier": args.classifier, "accuracy": acc, "n": n, "matrix": matrix.tolist()}

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps({"labels_order": labels_order, "results": results}, indent=2))
        print(f"\nWrote results to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
