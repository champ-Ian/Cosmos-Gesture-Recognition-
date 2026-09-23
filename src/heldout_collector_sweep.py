#!/usr/bin/env python3
"""
Held-out-collector generalization test: unlike loso_sweep.py's pooled LOSO
(every collector held out exactly once, all predictions pooled into one
accuracy number), this repeatedly carves the collector pool into a train
group and a *never-trained-on* test group, trains fresh each time, and
reports the mean/std of per-split accuracy -- a second, independent read on
how well the model generalizes to unseen people.

The paper's original version of this test described "30 random splits of the
ten collectors into 7 (train) and 3 (test)". That was run against
features_augmented.csv, which (see loso_sweep.py's module docstring) still
contained three collectors (ian/joanna/kaiwei) whose rows are byte-identical
duplicates of a single surviving recording from a collector-label collision
during data collection -- not independent people. Once those are excluded
(features_clean7.csv, 7 fully-independent collectors), "10 collectors into
7/3" no longer applies. With 7 collectors, this script instead does an
EXHAUSTIVE sweep over every 5-train/2-test split (C(7,2) = 21 unique splits)
rather than a random sample of a made-up count -- exhaustive is both more
rigorous and better-defined than "N random" once the full space is only 21
splits.

Usage (run from the repo root):
    python src/heldout_collector_sweep.py
    python src/heldout_collector_sweep.py --features-csv src/features_clean7.csv --test-size 2 --output results/figures/heldout_collector_clean7.json
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from loso_sweep import ALL_SENSORS, DEFAULT_FEATURES_CSV, build_model, load_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features-csv", default=str(DEFAULT_FEATURES_CSV))
    parser.add_argument("--test-size", type=int, default=2, help="Number of collectors held out per split (default 2, i.e. 5-train/2-test for 7 collectors).")
    parser.add_argument("--classifier", choices=["random_forest", "knn", "svm_linear", "cnn"], default="random_forest")
    parser.add_argument("--output", help="Optional path to write results as JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    features_csv = Path(args.features_csv).expanduser().resolve()
    per_sensor_X, y, collectors_arr, labels_order = load_data(features_csv)
    unique_collectors = sorted(set(collectors_arr.tolist()))
    n = len(unique_collectors)
    print(f"Loaded {len(y)} rows, {len(labels_order)} classes, {n} collectors from {features_csv}")

    if args.test_size >= n:
        raise SystemExit(f"--test-size ({args.test_size}) must be smaller than the collector count ({n}).")

    splits = list(itertools.combinations(unique_collectors, args.test_size))
    print(f"Exhaustive {n - args.test_size}-train/{args.test_size}-test sweep: {len(splits)} unique splits\n")

    accuracies = []
    per_split = []
    for test_collectors in splits:
        test_mask = np.isin(collectors_arr, test_collectors)
        train_mask = ~test_mask

        X_train = np.hstack([per_sensor_X[s][train_mask] for s in ALL_SENSORS])
        X_test = np.hstack([per_sensor_X[s][test_mask] for s in ALL_SENSORS])
        y_train, y_test = y[train_mask], y[test_mask]

        model, _ = build_model(args.classifier, len(X_train))
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = float((preds == y_test).mean())
        accuracies.append(acc)
        per_split.append({"test_collectors": list(test_collectors), "accuracy": acc, "n_test": int(test_mask.sum())})
        print(f"test={','.join(test_collectors):30s} accuracy = {acc*100:.2f}%  n_test={int(test_mask.sum())}")

    mean_acc = float(np.mean(accuracies))
    std_acc = float(np.std(accuracies))
    print(f"\nMean accuracy across {len(splits)} splits: {mean_acc*100:.2f}% +/- {std_acc*100:.2f}%")

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps({
            "features_csv": str(features_csv),
            "classifier": args.classifier,
            "n_collectors": n,
            "test_size": args.test_size,
            "n_splits": len(splits),
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "splits": per_split,
        }, indent=2))
        print(f"Wrote results to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
