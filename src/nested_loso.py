#!/usr/bin/env python3
"""
Nested leave-one-subject-out cross-validation for Random Forest and Linear SVM.

For each of the 7 outer folds (one collector held out as the untouched test
set), hyperparameters are chosen by an inner LOSO sweep over only that fold's
6 training collectors -- the outer test collector is never touched during
tuning. The winning config (by mean inner-fold accuracy) is refit on all 6
training collectors and evaluated once on the outer test collector.

Does not modify gesture_models.py / loso_sweep.py -- reuses their data loader
and mirrors their fixed classifier settings (class_weight='balanced',
random_state=42, RF n_estimators=200) but builds RF/SVM directly since
build_classifier() doesn't expose min_samples_leaf.

Usage:
    python src/nested_loso.py --features-csv src/features_clean7_noaug.csv --classifier random_forest
    python src/nested_loso.py --features-csv src/features_clean7_noaug.csv --classifier svm_linear
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from loso_sweep import ALL_SENSORS, load_data

RANDOM_STATE = 42

RF_GRID = [
    {"max_depth": md, "min_samples_leaf": msl}
    for md in [None, 5, 10, 20]
    for msl in [1, 3]
]
SVM_GRID = [{"C": c} for c in [0.01, 0.1, 1, 10, 100]]


def build_rf(params: dict) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )


def build_svm(params: dict):
    return make_pipeline(
        StandardScaler(),
        SVC(kernel="linear", C=params["C"], class_weight="balanced", probability=False, random_state=RANDOM_STATE),
    )


def inner_loso_score(build_fn, params: dict, X: np.ndarray, y: np.ndarray, collectors: np.ndarray, inner_collectors: list[str]) -> float:
    """Mean accuracy across |inner_collectors| inner LOSO folds (each inner collector held out once, trained on the other inner collectors)."""
    accs = []
    for held in inner_collectors:
        test_mask = collectors == held
        train_mask = np.isin(collectors, inner_collectors) & ~test_mask
        model = build_fn(params)
        model.fit(X[train_mask], y[train_mask])
        preds = model.predict(X[test_mask])
        accs.append(accuracy_score(y[test_mask], preds))
    return float(np.mean(accs))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features-csv", required=True)
    parser.add_argument("--classifier", choices=["random_forest", "svm_linear"], required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    per_sensor_X, y, collectors_arr, labels_order = load_data(Path(args.features_csv))
    X = np.hstack([per_sensor_X[s] for s in ALL_SENSORS])
    unique_collectors = sorted(set(collectors_arr.tolist()))
    print(f"Loaded {len(y)} rows, {len(labels_order)} classes, {len(unique_collectors)} collectors from {args.features_csv}")

    build_fn = build_rf if args.classifier == "random_forest" else build_svm
    grid = RF_GRID if args.classifier == "random_forest" else SVM_GRID

    fold_results = []
    all_true: list[str] = []
    all_pred: list[str] = []

    for outer_held in unique_collectors:
        outer_test_mask = collectors_arr == outer_held
        inner_collectors = [c for c in unique_collectors if c != outer_held]

        best_params, best_score = None, -1.0
        grid_scores = {}
        for params in grid:
            score = inner_loso_score(build_fn, params, X, y, collectors_arr, inner_collectors)
            grid_scores[json.dumps(params)] = score
            if score > best_score:
                best_score, best_params = score, params

        outer_train_mask = np.isin(collectors_arr, inner_collectors)
        final_model = build_fn(best_params)
        final_model.fit(X[outer_train_mask], y[outer_train_mask])
        preds = final_model.predict(X[outer_test_mask])
        y_true = y[outer_test_mask]
        fold_acc = float(accuracy_score(y_true, preds))

        all_true.extend(y_true.tolist())
        all_pred.extend(preds.tolist())

        fold_results.append({
            "outer_test_collector": outer_held,
            "best_params": best_params,
            "inner_cv_score": best_score,
            "outer_fold_accuracy": fold_acc,
            "n_test": int(outer_test_mask.sum()),
            "grid_scores": grid_scores,
        })
        print(f"outer={outer_held:10s} best_params={best_params} inner_cv={best_score*100:.2f}%  outer_fold_acc={fold_acc*100:.2f}%  n_test={int(outer_test_mask.sum())}")

    pooled_acc = float(accuracy_score(all_true, all_pred))
    fold_accs = [f["outer_fold_accuracy"] for f in fold_results]
    mean_acc = float(np.mean(fold_accs))
    std_acc = float(np.std(fold_accs))

    print(f"\nPooled accuracy (all {len(all_true)} held-out predictions): {pooled_acc*100:.2f}%")
    print(f"Mean of per-fold accuracy: {mean_acc*100:.2f}% +/- {std_acc*100:.2f}% (std across {len(fold_accs)} folds)")

    if args.output:
        Path(args.output).write_text(json.dumps({
            "classifier": args.classifier,
            "features_csv": args.features_csv,
            "n_rows": len(y),
            "n_collectors": len(unique_collectors),
            "pooled_accuracy": pooled_acc,
            "mean_fold_accuracy": mean_acc,
            "std_fold_accuracy": std_acc,
            "fold_results": fold_results,
        }, indent=2))
        print(f"Wrote {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
