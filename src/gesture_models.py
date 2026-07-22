#!/usr/bin/env python3
"""
Shared classifier building blocks for `train.py` / `evaluate.py` / `realtime_demo.py`.

`build_classifier` mirrors `UWB_lab/train.py`'s starter KNN/linear-SVM
choices so the two classifiers you're asked to compare (per the final
project's modeling requirements) work the same way here.

`LateFusionClassifier` implements the "late fusion" option from the project
spec: one classifier per sensor, combined at prediction time by averaging
each sensor's predicted class probabilities. It's picklable (plain functions/
classes at module level) so `joblib.dump`/`joblib.load` can save and load it
like any other model.
"""
from __future__ import annotations

import numpy as np


def classifier_label(classifier: str) -> str:
    return {
        "knn": "KNN",
        "svm_linear": "Linear SVM",
    }[classifier]


def build_classifier(
    classifier: str,
    train_count: int,
    random_state: int = 42,
    svm_c: float = 1.0,
    knn_neighbors: int = 5,
    knn_weights: str = "distance",
):
    """Build an (untrained) scikit-learn pipeline. Returns (pipeline, params)."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    if classifier == "svm_linear":
        params = {
            "kernel": "linear",
            "C": svm_c,
            "class_weight": "balanced",
            "probability": True,
            "random_state": random_state,
        }
        return make_pipeline(StandardScaler(), SVC(**params)), params

    if classifier == "knn":
        requested_neighbors = max(1, int(knn_neighbors))
        actual_neighbors = min(requested_neighbors, int(train_count))
        if actual_neighbors < requested_neighbors:
            print(
                f"Reducing KNN neighbors from {requested_neighbors} to {actual_neighbors} "
                f"because the training split has {train_count} examples."
            )
        params = {"n_neighbors": actual_neighbors, "weights": knn_weights}
        return make_pipeline(StandardScaler(), KNeighborsClassifier(**params)), params

    raise SystemExit(f"Unsupported classifier: {classifier}")


class LateFusionClassifier:
    """Averages per-sensor predicted probabilities (late fusion).

    `sensor_models[sensor]` must be a fitted scikit-learn-style estimator
    (has `.predict_proba` and `.classes_`) trained on that sensor's feature
    matrix alone, all sharing the same label set. Predict with a dict mapping
    sensor -> feature matrix (2D, one row per example) or a single 1D feature
    vector (treated as one example).
    """

    def __init__(self, sensor_models: dict[str, object], sensors: list[str]):
        self.sensor_models = sensor_models
        self.sensors = list(sensors)
        reference = self.sensor_models[self.sensors[0]]
        self.classes_ = list(reference.classes_)
        for sensor in self.sensors[1:]:
            other_classes = list(self.sensor_models[sensor].classes_)
            if other_classes != self.classes_:
                raise ValueError(
                    f"Per-sensor models disagree on class order/labels: "
                    f"{self.sensors[0]}={self.classes_} vs {sensor}={other_classes}. "
                    "Late fusion requires every sub-model to be trained on the same label set."
                )

    def predict_proba(self, X: dict[str, np.ndarray]) -> np.ndarray:
        probas = []
        for sensor in self.sensors:
            arr = np.atleast_2d(np.asarray(X[sensor], dtype=float))
            probas.append(self.sensor_models[sensor].predict_proba(arr))
        return np.mean(probas, axis=0)

    def predict(self, X: dict[str, np.ndarray]) -> np.ndarray:
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return np.array(self.classes_)[indices]
