#!/usr/bin/env python3
"""Two things not covered by existing scripts:

1. Per-fold accuracy for kNN and CNN (early, non-augmented, standard LOSO) --
   loso_sweep.py's run_loso() only returns pooled accuracy.
2. The 8 augmented-condition (early/late x 4 models) configs, re-evaluated so
   the held-out collector's TEST rows are restricted to her real (non-
   synthetic) samples only -- synthetic rows still train normally, only the
   test-time evaluation changes. New script; does not touch loso_sweep.py,
   augment_data.py, or any feature/preprocessing code.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score
from loso_sweep import ALL_SENSORS, build_model, load_data
from gesture_models import LateFusionClassifier

OUT = Path("/private/tmp/claude-501/-Users-vibhamandayam-OpenCS-Cosmos-Gesture-Recognition-/4acb9c94-82ab-4917-9400-cdc7f7119f79/scratchpad")


def per_fold(classifier: str, features_csv: str):
    per_sensor_X, y, collectors_arr, labels_order = load_data(Path(features_csv))
    X = np.hstack([per_sensor_X[s] for s in ALL_SENSORS])
    unique_collectors = sorted(set(collectors_arr.tolist()))
    fold_accs = {}
    for held in unique_collectors:
        test_mask = collectors_arr == held
        train_mask = ~test_mask
        model, _ = build_model(classifier, int(train_mask.sum()))
        model.fit(X[train_mask], y[train_mask])
        preds = model.predict(X[test_mask])
        fold_accs[held] = float(accuracy_score(y[test_mask], preds))
    accs = list(fold_accs.values())
    print(f"{classifier}: mean={np.mean(accs)*100:.2f}% std={np.std(accs)*100:.2f}%  per-fold={fold_accs}")
    return {"classifier": classifier, "mean": float(np.mean(accs)), "std": float(np.std(accs)), "fold_accuracies": fold_accs}


def main():
    import csv
    # Load features_clean7.csv directly to get session_dir (load_data() doesn't expose it)
    with open("src/features_clean7.csv") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    sensors = ALL_SENSORS
    from extract_features import feature_names_for_sensor
    lengths = {s: len(feature_names_for_sensor(s)) for s in sensors}
    per_sensor_X = {s: [] for s in sensors}
    y, collectors, session_dirs = [], [], []
    for row in rows:
        gesture, collector, source, session_dir = row[:4]
        values = [float(v) for v in row[4:]]
        offset = 0
        for s in sensors:
            per_sensor_X[s].append(values[offset:offset+lengths[s]])
            offset += lengths[s]
        y.append(gesture); collectors.append(collector); session_dirs.append(session_dir)
    per_sensor_X = {s: np.asarray(v, dtype=float) for s, v in per_sensor_X.items()}
    X = np.hstack([per_sensor_X[s] for s in sensors])
    y = np.asarray(y); collectors_arr = np.asarray(collectors)
    is_real = np.array(["_aug" not in Path(s).name for s in session_dirs])
    unique_collectors = sorted(set(collectors))
    labels_order = sorted(set(y))

    results = {}
    for classifier in ["random_forest", "knn", "svm_linear", "cnn"]:
        for fusion in ["early", "late"]:
            all_true, all_pred = [], []
            for held in unique_collectors:
                held_mask = collectors_arr == held
                train_mask = ~held_mask  # augmented: all rows (real+synth) of other 6 collectors
                test_mask = held_mask & is_real  # REAL rows only of held-out collector
                if fusion == "early":
                    model, _ = build_model(classifier, int(train_mask.sum()))
                    model.fit(X[train_mask], y[train_mask])
                    preds = model.predict(X[test_mask])
                else:
                    sensor_models = {}
                    for s in sensors:
                        m, _ = build_model(classifier, int(train_mask.sum()))
                        m.fit(per_sensor_X[s][train_mask], y[train_mask])
                        sensor_models[s] = m
                    model = LateFusionClassifier(sensor_models, sensors)
                    preds = model.predict({s: per_sensor_X[s][test_mask] for s in sensors})
                all_true.extend(y[test_mask].tolist()); all_pred.extend(preds.tolist())
            acc = float(accuracy_score(all_true, all_pred))
            key = f"{classifier}_{fusion}"
            results[key] = acc
            print(f"{key:30s} real-only-test accuracy = {acc*100:.2f}%  n={len(all_true)}")

    OUT.joinpath("real_only_test.json").write_text(json.dumps(results, indent=2))

    print("\n=== kNN/CNN per-fold (early, non-augmented) ===")
    knn_r = per_fold("knn", "src/features_clean7_noaug.csv")
    cnn_r = per_fold("cnn", "src/features_clean7_noaug.csv")
    OUT.joinpath("knn_cnn_perfold.json").write_text(json.dumps({"knn": knn_r, "cnn": cnn_r}, indent=2))


if __name__ == "__main__":
    main()
