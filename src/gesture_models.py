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
        "cnn_raw": "1D-CNN (raw sequences)",
        "random_forest": "Random Forest",
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
    rf_n_estimators: int = 200,
    rf_max_depth: int | None = None,
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

    if classifier == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        params = {
            "n_estimators": int(rf_n_estimators),
            "max_depth": rf_max_depth,
            "class_weight": "balanced",
            "random_state": random_state,
        }
        # No StandardScaler: tree splits are scale-invariant, unlike KNN/SVM's distance
        # or margin-based decision rules.
        return RandomForestClassifier(**params), params

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
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels * 2),
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


class _HybridRawCNN:
    """Builds the raw-sequence CNN from plain `nn.Sequential`/`nn.ModuleDict`
    pieces -- a real temporal Conv1d stack over raw per-channel sequences
    (e.g. IMU ax/ay/az/gx/gy/gz, mmWave energy/points, UWB distance, each
    resampled to a fixed length), fused with a small dense branch over the
    existing hand-crafted summary-stat features.

    Deliberately NOT a custom `nn.Module` subclass: a class defined inside a
    function can't be `joblib.dump`/pickle'd (pickle needs a real importable
    qualified name), which is exactly the same reason `TorchCNNClassifier`'s
    `_CNN1D.build` only ever returns an `nn.Sequential` of stock layers.
    `nn.ModuleDict` is itself a stock, picklable container, so wrapping the
    three (all-stock-layer) pieces in one lets `.parameters()`/`.train()`/
    `.eval()` still work as if this were one module -- `TorchRawCNNClassifier`
    just calls each piece explicitly instead of relying on a `forward()`.
    """

    @staticmethod
    def build(raw_channels: int, scalar_dim: int, n_classes: int, hidden_channels: int, dropout: float):
        import torch.nn as nn

        conv = nn.Sequential(
            nn.Conv1d(raw_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels * 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        scalar_hidden = max(8, hidden_channels // 2) if scalar_dim > 0 else 0
        parts = {"conv": conv}
        if scalar_dim > 0:
            parts["scalar"] = nn.Sequential(nn.Linear(scalar_dim, scalar_hidden), nn.ReLU())
        combined_dim = hidden_channels * 2 + scalar_hidden
        parts["head"] = nn.Sequential(nn.Dropout(dropout), nn.Linear(combined_dim, n_classes))
        return nn.ModuleDict(parts)


class TorchRawCNNClassifier:
    """1D-CNN over raw per-channel sequences (+ an optional scalar side-branch).

    Unlike `TorchCNNClassifier` (which treats one flat hand-crafted feature
    vector as a length-N single-channel signal -- there's no real temporal
    locality between adjacent feature columns there), this convolves over an
    actual time axis: `X_raw` is `(n_examples, n_channels, n_timesteps)`, e.g.
    9 channels (mmwave energy/points, imu ax/ay/az/gx/gy/gz, uwb distance)
    each resampled to a fixed length. This is what actually gets the
    "CNN learns temporal patterns" advantage.

    Not currently wired into `train.py`'s generic --classifier flag or
    `realtime_demo.py` -- both assume a single flat feature vector per
    example. Use directly (see `train_cnn_raw.py`) until/unless the live
    windowing path is extended to produce matching raw resampled sequences.
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
        self._raw_mean: np.ndarray | None = None  # (n_channels, 1)
        self._raw_std: np.ndarray | None = None
        self._scalar_mean: np.ndarray | None = None
        self._scalar_std: np.ndarray | None = None
        self._model = None

    def _standardize(self, X_raw: np.ndarray, X_scalar: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X_raw = (X_raw - self._raw_mean) / self._raw_std
        if X_scalar.shape[1] > 0:
            X_scalar = (X_scalar - self._scalar_mean) / self._scalar_std
        return X_raw, X_scalar

    def _forward(self, x_raw, x_scalar):
        import torch

        raw_embed = self._model["conv"](x_raw)
        if "scalar" in self._model:
            combined = torch.cat([raw_embed, self._model["scalar"](x_scalar)], dim=1)
        else:
            combined = raw_embed
        return self._model["head"](combined)

    def fit(self, X_raw, X_scalar, y) -> "TorchRawCNNClassifier":
        import torch
        import torch.nn as nn

        torch.manual_seed(self.random_state)

        X_raw = np.asarray(X_raw, dtype=np.float32)  # (N, channels, timesteps)
        X_scalar = np.atleast_2d(np.asarray(X_scalar, dtype=np.float32))
        if X_scalar.shape[0] != X_raw.shape[0]:
            X_scalar = X_scalar.reshape(X_raw.shape[0], -1)
        y = np.asarray(y)
        self.classes_ = sorted(set(y.tolist()))
        class_to_idx = {label: idx for idx, label in enumerate(self.classes_)}
        y_idx = np.array([class_to_idx[label] for label in y], dtype=np.int64)

        # Per-channel mean/std over all examples and timesteps combined.
        self._raw_mean = X_raw.mean(axis=(0, 2), keepdims=True)
        self._raw_std = X_raw.std(axis=(0, 2), keepdims=True)
        self._raw_std[self._raw_std < 1e-8] = 1.0

        scalar_dim = X_scalar.shape[1]
        if scalar_dim > 0:
            self._scalar_mean = X_scalar.mean(axis=0)
            self._scalar_std = X_scalar.std(axis=0)
            self._scalar_std[self._scalar_std < 1e-8] = 1.0

        X_raw, X_scalar = self._standardize(X_raw, X_scalar)

        n_channels = X_raw.shape[1]
        self._model = _HybridRawCNN.build(n_channels, scalar_dim, len(self.classes_), self.hidden_channels, self.dropout)

        raw_tensor = torch.from_numpy(X_raw)
        scalar_tensor = torch.from_numpy(X_scalar)
        y_tensor = torch.from_numpy(y_idx)
        dataset = torch.utils.data.TensorDataset(raw_tensor, scalar_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()

        self._model.train()
        for _ in range(self.epochs):
            for batch_raw, batch_scalar, batch_y in loader:
                optimizer.zero_grad()
                loss = loss_fn(self._forward(batch_raw, batch_scalar), batch_y)
                loss.backward()
                optimizer.step()

        return self

    def predict_proba(self, X_raw, X_scalar) -> np.ndarray:
        import torch

        X_raw = np.asarray(X_raw, dtype=np.float32)
        if X_raw.ndim == 2:
            X_raw = X_raw[np.newaxis, ...]
        X_scalar = np.atleast_2d(np.asarray(X_scalar, dtype=np.float32))
        if X_scalar.shape[0] != X_raw.shape[0]:
            X_scalar = X_scalar.reshape(X_raw.shape[0], -1)
        X_raw, X_scalar = self._standardize(X_raw, X_scalar)

        self._model.eval()
        with torch.no_grad():
            logits = self._forward(torch.from_numpy(X_raw), torch.from_numpy(X_scalar))
            proba = torch.softmax(logits, dim=1).numpy()
        return proba

    def predict(self, X_raw, X_scalar) -> np.ndarray:
        proba = self.predict_proba(X_raw, X_scalar)
        indices = np.argmax(proba, axis=1)
        return np.array(self.classes_)[indices]


class RawSingleArrayWrapper:
    """Wraps a `TorchRawCNNClassifier` behind a single-flat-array `.fit(X, y)` /
    `.predict(X)` / `.predict_proba(X)` interface, splitting `X` back into
    `(X_raw, X_scalar)` by the stored channel/scalar layout.

    `LateFusionClassifier` calls `sensor_models[sensor].predict_proba(arr)` --
    one array, like every sklearn estimator -- but `TorchRawCNNClassifier`
    takes two. Rather than special-case `LateFusionClassifier` for raw models,
    this makes a per-sensor raw CNN look like any other sensor_model, so late
    fusion for `cnn_raw` reuses the exact same averaging code `knn`/`svm_linear`/
    `cnn` already use instead of a separate implementation.
    """

    def __init__(self, channel_names: list[str], scalar_names: list[str], raw_length: int, **cnn_kwargs):
        self.channel_names = channel_names
        self.scalar_names = scalar_names
        self.raw_length = raw_length
        self.model = TorchRawCNNClassifier(**cnn_kwargs)
        self.classes_: list = []

    def _split(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.atleast_2d(np.asarray(X, dtype=np.float32))
        n_raw = len(self.channel_names) * self.raw_length
        X_raw = X[:, :n_raw].reshape(-1, len(self.channel_names), self.raw_length)
        X_scalar = X[:, n_raw:]
        return X_raw, X_scalar

    def fit(self, X, y) -> "RawSingleArrayWrapper":
        X_raw, X_scalar = self._split(X)
        self.model.fit(X_raw, X_scalar, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X) -> np.ndarray:
        X_raw, X_scalar = self._split(X)
        return self.model.predict(X_raw, X_scalar)

    def predict_proba(self, X) -> np.ndarray:
        X_raw, X_scalar = self._split(X)
        return self.model.predict_proba(X_raw, X_scalar)


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
