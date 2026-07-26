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
        "cnn": "1D-CNN",
    }[classifier]


def build_classifier(
    classifier: str,
    train_count: int,
    random_state: int = 42,
    svm_c: float = 1.0,
    knn_neighbors: int = 5,
    knn_weights: str = "distance",
    cnn_epochs: int = 200,
    cnn_lr: float = 1e-3,
    cnn_hidden_channels: int = 32,
    cnn_dropout: float = 0.3,
    cnn_batch_size: int = 16,
):
    """Build an (untrained) classifier. Returns (model, params).

    `knn`/`svm_linear` are scikit-learn pipelines; `cnn` is a `TorchCNNClassifier`
    below. All three share the same `.fit(X, y)` / `.predict(X)` /
    `.predict_proba(X)` / `.classes_` surface, which is all `train.py`,
    `evaluate.py`, and `realtime_demo.py` rely on.
    """
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

    if classifier == "cnn":
        requested_batch_size = max(1, int(cnn_batch_size))
        actual_batch_size = min(requested_batch_size, int(train_count))
        if actual_batch_size < requested_batch_size:
            print(
                f"Reducing CNN batch size from {requested_batch_size} to {actual_batch_size} "
                f"because the training split has {train_count} examples."
            )
        params = {
            "epochs": int(cnn_epochs),
            "lr": float(cnn_lr),
            "hidden_channels": int(cnn_hidden_channels),
            "dropout": float(cnn_dropout),
            "batch_size": actual_batch_size,
            "random_state": random_state,
        }
        return TorchCNNClassifier(**params), params

    raise SystemExit(f"Unsupported classifier: {classifier}")


class _CNN1D:
    """`torch.nn.Module` isn't importable at module load time if torch isn't
    installed; this builds it lazily inside `TorchCNNClassifier.fit`."""

    @staticmethod
    def build(n_features: int, n_classes: int, hidden_channels: int, dropout: float):
        import torch.nn as nn

        return nn.Sequential(
            nn.Conv1d(1, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, n_classes),
        )


class TorchCNNClassifier:
    """A small 1D-CNN wrapped to look like a scikit-learn classifier.

    Treats each fixed-length feature vector (from `extract_features.py`) as a
    single-channel 1D signal of length `n_features` and runs it through a couple
    of Conv1d + ReLU blocks, global-average-pooled into a per-class logit. This
    lets `--classifier cnn` slot into `train.py`, `evaluate.py`, and
    `realtime_demo.py` exactly like the `knn`/`svm_linear` pipelines -- same
    `.fit(X, y)` / `.predict(X)` / `.predict_proba(X)` / `.classes_` surface,
    same joblib save/load path.

    Feature scaling is done internally (mean/std computed in `fit`) since raw
    feature vectors mix wildly different sensor units/ranges, the same reason
    `build_classifier` wraps knn/svm in a `StandardScaler`.
    """

    def __init__(
        self,
        epochs: int = 200,
        lr: float = 1e-3,
        hidden_channels: int = 32,
        dropout: float = 0.3,
        batch_size: int = 16,
        random_state: int = 42,
    ):
        self.epochs = epochs
        self.lr = lr
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        self.batch_size = batch_size
        self.random_state = random_state
        self.classes_: list = []
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._model = None

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return (X - self._mean) / self._std

    def fit(self, X, y) -> "TorchCNNClassifier":
        import torch
        import torch.nn as nn

        torch.manual_seed(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.classes_ = sorted(set(y.tolist()))
        class_to_idx = {label: idx for idx, label in enumerate(self.classes_)}
        y_idx = np.array([class_to_idx[label] for label in y], dtype=np.int64)

        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0)
        self._std[self._std < 1e-8] = 1.0
        X = self._standardize(X)

        n_features = X.shape[1]
        self._model = _CNN1D.build(n_features, len(self.classes_), self.hidden_channels, self.dropout)

        X_tensor = torch.from_numpy(X).unsqueeze(1)  # (N, 1, n_features)
        y_tensor = torch.from_numpy(y_idx)
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()

        self._model.train()
        for _ in range(self.epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                loss = loss_fn(self._model(batch_X), batch_y)
                loss.backward()
                optimizer.step()

        return self

    def predict_proba(self, X) -> np.ndarray:
        import torch

        X = np.atleast_2d(np.asarray(X, dtype=np.float32))
        X = self._standardize(X)
        X_tensor = torch.from_numpy(X.astype(np.float32)).unsqueeze(1)
        self._model.eval()
        with torch.no_grad():
            logits = self._model(X_tensor)
            proba = torch.softmax(logits, dim=1).numpy()
        return proba

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return np.array(self.classes_)[indices]


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
