#!/usr/bin/env python3
"""Per-fold accuracy (not just pooled) for every Modality Ablation row, so mean+-std across the 7 LOSO folds can be reported alongside the pooled number. New script, does not modify loso_sweep.py."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score
from loso_sweep import ABLATION_ROWS, ALL_SENSORS, build_model, load_data

def main():
    features_csv = Path("src/features_clean7_noaug.csv")
    per_sensor_X, y, collectors_arr, labels_order = load_data(features_csv)
    unique_collectors = sorted(set(collectors_arr.tolist()))

    out = {}
    for sensors, fusion in ABLATION_ROWS:
        fold_accs = []
        all_true, all_pred = [], []
        for held in unique_collectors:
            test_mask = collectors_arr == held
            train_mask = ~test_mask
            y_train, y_test = y[train_mask], y[test_mask]
            if fusion == "early":
                X_train = np.hstack([per_sensor_X[s][train_mask] for s in sensors])
                X_test = np.hstack([per_sensor_X[s][test_mask] for s in sensors])
                model, _ = build_model("random_forest", len(X_train))
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
            else:
                from gesture_models import LateFusionClassifier
                sensor_models = {}
                for s in sensors:
                    m, _ = build_model("random_forest", int(train_mask.sum()))
                    m.fit(per_sensor_X[s][train_mask], y_train)
                    sensor_models[s] = m
                model = LateFusionClassifier(sensor_models, sensors)
                preds = model.predict({s: per_sensor_X[s][test_mask] for s in sensors})
            fold_acc = float(accuracy_score(y_test, preds))
            fold_accs.append(fold_acc)
            all_true.extend(y_test.tolist()); all_pred.extend(preds.tolist())
        pooled = float(accuracy_score(all_true, all_pred))
        key = f"{'+'.join(sensors)}_{fusion}"
        out[key] = {"sensors": sensors, "fusion": fusion, "pooled_accuracy": pooled,
                     "mean_fold_accuracy": float(np.mean(fold_accs)), "std_fold_accuracy": float(np.std(fold_accs)),
                     "fold_accuracies": dict(zip(unique_collectors, fold_accs))}
        print(f"{key:30s} pooled={pooled*100:.2f}%  mean={np.mean(fold_accs)*100:.2f}%  std={np.std(fold_accs)*100:.2f}%")

    Path("/private/tmp/claude-501/-Users-vibhamandayam-OpenCS-Cosmos-Gesture-Recognition-/4acb9c94-82ab-4917-9400-cdc7f7119f79/scratchpad/ablation_per_fold.json").write_text(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
