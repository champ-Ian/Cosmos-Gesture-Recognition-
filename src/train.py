#!/usr/bin/env python3
"""
Train a gesture classifier from `extract_features.py cut` trials.

Modeled on `UWB_lab/train.py`: pick a sensor subset with `--sensors` (a
single sensor gives you the "single-sensor baseline" the project rubric
asks for; multiple sensors gives you the "fused model" to compare against
it), a fusion strategy (`--fusion early|late`, see the project spec's
Fusion type table), and a classifier to compare against another run
(`--classifier knn|svm_linear`).

Only trials where `sensors_enabled` (in trials.csv) includes every sensor in
`--sensors`, and where every one of those sensors' features can actually be
extracted (see extract_features.py), are used -- this keeps early/late
fusion runs and different `--sensors` choices comparable on the same
underlying trials where possible, and reports what got skipped and why.

Run from the repo root as `python src/train.py ...` (paths below assume that):

    # Single-sensor mmWave baseline
    python src/train.py data/processed/combined --sensors mmwave

    # Early-fusion mmWave+IMU model
    python src/train.py data/processed/combined --sensors mmwave,imu --fusion early

    # Late-fusion mmWave+IMU model (one classifier per sensor, averaged)
    python src/train.py data/processed/combined --sensors mmwave,imu --fusion late

    # Held-out-person evaluation
    python src/train.py data/processed/combined --sensors mmwave,imu --test-collector student03
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from extract_features import feature_names_for_sensor, feature_names_for_sensors, extract_sensor_features
from gesture_models import LateFusionClassifier, build_classifier, classifier_label
from sensors.common import MODELS_DIR, RESULTS_FIGURES_DIR, timestamp

ALL_SENSORS = ("mmwave", "imu", "uwb", "rfid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", nargs="+", help="Processed dataset folders from extract_features.py cut / combine_datasets.py.")
    parser.add_argument(
        "--sensors",
        required=True,
        help="Comma-separated sensor subset to use, e.g. 'mmwave' or 'mmwave,imu,uwb'.",
    )
    parser.add_argument("--fusion", choices=["early", "late"], default="early", help="How to combine multiple sensors.")
    parser.add_argument("--classifier", choices=["knn", "svm_linear"], default="knn")
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--knn-neighbors", type=int, default=5)
    parser.add_argument("--knn-weights", choices=["uniform", "distance"], default="distance")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--test-collector",
        action="append",
        help="Hold out this collector as the test set (all others train). Repeatable/comma separated.",
    )
    parser.add_argument("--model-out", help="Output .joblib path. Default: <repo>/models/<classifier>_<fusion>_<sensors>_<timestamp>.joblib")
    parser.add_argument("--confusion-out", help="Output confusion matrix PNG. Default: <repo>/results/figures/.")
    return parser.parse_args()


def normalize_sensor_list(value: str) -> list[str]:
    sensors = [part.strip().lower() for part in value.split(",") if part.strip()]
    for sensor in sensors:
        if sensor not in ALL_SENSORS:
            raise SystemExit(f"Unknown sensor '{sensor}'. Valid sensors: {', '.join(ALL_SENSORS)}")
    if not sensors:
        raise SystemExit("--sensors must name at least one sensor.")
    return sensors


def normalize_filter(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    normalized: set[str] = set()
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                normalized.add(part)
    return normalized


def read_manifest(dataset_dir: Path) -> list[dict]:
    manifest = dataset_dir / "trials.csv"
    if manifest.exists():
        with open(manifest, newline="") as f:
            return list(csv.DictReader(f))
    rows = []
    for metadata_path in sorted(dataset_dir.rglob("trial_metadata.json")):
        try:
            rows.append(json.loads(metadata_path.read_text()))
        except json.JSONDecodeError:
            continue
    return rows


def source_dataset_name(dataset_dir: Path) -> str:
    metadata_path = dataset_dir / "dataset_metadata.json"
    if metadata_path.exists():
        try:
            return json.loads(metadata_path.read_text()).get("dataset_name") or dataset_dir.name
        except json.JSONDecodeError:
            pass
    return dataset_dir.name


def read_manifests(dataset_dirs: list[Path]) -> tuple[list[dict], list[str]]:
    rows = []
    missing = []
    for dataset_dir in dataset_dirs:
        dataset_rows = read_manifest(dataset_dir)
        if not dataset_rows:
            missing.append(str(dataset_dir))
            continue
        source_name = source_dataset_name(dataset_dir)
        for row in dataset_rows:
            item = dict(row)
            item["_dataset_dir"] = str(dataset_dir)
            item["_source_dataset"] = source_name
            rows.append(item)
    return rows, missing


def resolve_path(dataset_dir: Path, value: str, default: Path) -> Path:
    if not value:
        return default
    path = Path(value)
    return path if path.is_absolute() else (dataset_dir / path).resolve()


def build_examples(rows: list[dict], sensors: list[str]) -> tuple[dict, list, list, list, list, list]:
    """Returns (per_sensor_examples, labels, collectors, sources, session_dirs, skipped)."""
    per_sensor_examples: dict[str, list] = {sensor: [] for sensor in sensors}
    labels: list[str] = []
    collectors: list[str] = []
    sources: list[str] = []
    session_dirs: list[str] = []
    skipped: list[dict] = []

    for row in rows:
        gesture = row.get("gesture")
        dataset_dir = Path(row.get("_dataset_dir", "."))
        session_dir = resolve_path(dataset_dir, row.get("session_dir", ""), dataset_dir)
        npz_path = resolve_path(dataset_dir, row.get("npz_path", ""), session_dir / "trial_data.npz")

        sensors_enabled = {s.strip() for s in row.get("sensors_enabled", "").split(",") if s.strip()}
        missing_sensors = [s for s in sensors if s not in sensors_enabled]
        if missing_sensors:
            skipped.append({"session_dir": str(session_dir), "gesture": gesture, "reason": f"sensors not enabled: {missing_sensors}"})
            continue
        if not npz_path.exists():
            skipped.append({"session_dir": str(session_dir), "gesture": gesture, "reason": "missing trial_data.npz"})
            continue

        with np.load(npz_path, allow_pickle=True) as npz:
            per_sensor_vectors = {}
            ok = True
            for sensor in sensors:
                vector = extract_sensor_features(sensor, npz)
                if vector is None:
                    ok = False
                    break
                per_sensor_vectors[sensor] = vector
            if not ok:
                skipped.append({"session_dir": str(session_dir), "gesture": gesture, "reason": "not enough sensor data to extract features"})
                continue

        for sensor in sensors:
            per_sensor_examples[sensor].append(per_sensor_vectors[sensor])
        labels.append(gesture)
        collectors.append(row.get("collector", ""))
        sources.append(row.get("_source_dataset", ""))
        session_dirs.append(str(session_dir))

    return per_sensor_examples, labels, collectors, sources, session_dirs, skipped


def split_indices(y: np.ndarray, collectors: np.ndarray, args: argparse.Namespace):
    test_collectors = normalize_filter(args.test_collector)
    if test_collectors:
        test_mask = np.array([c in test_collectors for c in collectors], dtype=bool)
        train_mask = ~test_mask
        if not test_mask.any():
            raise SystemExit(
                "No examples matched --test-collector. Available collectors: " + ", ".join(sorted(set(collectors)))
            )
        if not train_mask.any():
            raise SystemExit("No training examples remain after applying --test-collector.")
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        return train_idx, test_idx, "collector_holdout"

    from sklearn.model_selection import train_test_split

    class_counts = {label: int((y == label).sum()) for label in sorted(set(y))}
    stratify = y if min(class_counts.values()) >= 2 else None
    n_samples = len(y)
    split_test_size = args.test_size
    if stratify is not None:
        n_classes = len(class_counts)
        requested = math.ceil(args.test_size * n_samples) if args.test_size < 1 else int(args.test_size)
        split_test_size = min(max(requested, n_classes), n_samples - n_classes)

    indices = np.arange(n_samples)
    train_idx, test_idx = train_test_split(
        indices, test_size=split_test_size, random_state=args.random_state, stratify=stratify
    )
    return train_idx, test_idx, "random"


def default_model_path(classifier: str, fusion: str, sensors: list[str]) -> Path:
    tag = "-".join(sensors)
    name = f"{classifier}_{fusion}_{tag}_{timestamp()}.joblib"
    return MODELS_DIR / name


def main() -> int:
    args = parse_args()
    sensors = normalize_sensor_list(args.sensors)
    dataset_dirs = [Path(item).expanduser().resolve() for item in args.datasets]
    rows, missing_datasets = read_manifests(dataset_dirs)
    if not rows:
        raise SystemExit("No trials found. Expected trials.csv or trial_metadata.json files.")

    per_sensor_examples, labels, collectors, sources, session_dirs, skipped = build_examples(rows, sensors)
    if len(set(labels)) < 2:
        raise SystemExit("Need at least two gesture classes to train.")
    if len(labels) < 4:
        raise SystemExit("Need at least four usable trials to train.")

    per_sensor_X = {sensor: np.asarray(vectors, dtype=float) for sensor, vectors in per_sensor_examples.items()}
    y = np.asarray(labels)
    collectors_array = np.asarray(collectors)
    class_counts = {label: int((y == label).sum()) for label in sorted(set(labels))}

    train_idx, test_idx, split_method = split_indices(y, collectors_array, args)
    y_train, y_test = y[train_idx], y[test_idx]
    train_only_check = set(y_test) - set(y_train)
    if train_only_check:
        raise SystemExit("The test set contains gesture labels not present in training: " + ", ".join(sorted(train_only_check)))
    if len(set(y_train)) < 2:
        raise SystemExit("Training split needs at least two gesture classes.")
    train_collectors = sorted(set(collectors_array[train_idx]))
    test_collectors_used = sorted(set(collectors_array[test_idx]))

    per_sensor_X_train = {s: X[train_idx] for s, X in per_sensor_X.items()}
    per_sensor_X_test = {s: X[test_idx] for s, X in per_sensor_X.items()}

    try:
        import joblib
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
    except ImportError as exc:
        raise SystemExit(f"Training dependencies missing: {exc}")

    classifier_params: dict
    feature_names: list[str] | dict[str, list[str]]

    if args.fusion == "early":
        X_train = np.hstack([per_sensor_X_train[s] for s in sensors])
        X_test = np.hstack([per_sensor_X_test[s] for s in sensors])
        model, classifier_params = build_classifier(
            args.classifier, len(X_train), args.random_state, args.svm_c, args.knn_neighbors, args.knn_weights
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        feature_names = feature_names_for_sensors(sensors)
    else:
        sensor_models = {}
        classifier_params = {}
        for sensor in sensors:
            sub_model, params = build_classifier(
                args.classifier, len(per_sensor_X_train[sensor]), args.random_state, args.svm_c, args.knn_neighbors, args.knn_weights
            )
            sub_model.fit(per_sensor_X_train[sensor], y_train)
            sensor_models[sensor] = sub_model
            classifier_params[sensor] = params
        model = LateFusionClassifier(sensor_models, sensors)
        predictions = model.predict(per_sensor_X_test)
        feature_names = {sensor: feature_names_for_sensor(sensor) for sensor in sensors}

    accuracy = float(accuracy_score(y_test, predictions))
    labels_order = sorted(set(y))
    matrix = confusion_matrix(y_test, predictions, labels=labels_order)

    model_out = Path(args.model_out) if args.model_out else default_model_path(args.classifier, args.fusion, sensors)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "sensors": sensors,
        "fusion": args.fusion,
        "feature_names": feature_names,
        "input_type": "cosmos_multi_sensor_gesture",
        "classifier": args.classifier,
        "classifier_label": classifier_label(args.classifier),
        "classifier_params": classifier_params,
        "labels": labels_order,
        "training_datasets": [str(path) for path in dataset_dirs],
        "split_method": split_method,
        "test_collectors": test_collectors_used,
    }
    joblib.dump(payload, model_out)

    confusion_out = (
        Path(args.confusion_out) if args.confusion_out else RESULTS_FIGURES_DIR / (model_out.stem + "_confusion_matrix.png")
    )
    confusion_out.parent.mkdir(parents=True, exist_ok=True)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels_order)
    fig, ax = plt.subplots(figsize=(6, 5))
    display.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
    ax.set_title(
        f"{classifier_label(args.classifier)} ({args.fusion} fusion: {'+'.join(sensors)})\nAccuracy: {accuracy:.3f}"
    )
    fig.tight_layout()
    fig.savefig(confusion_out, dpi=180)
    plt.close(fig)

    per_class_recall = {
        label: float(matrix[i, i] / matrix[i].sum()) if matrix[i].sum() else None
        for i, label in enumerate(labels_order)
    }
    summary = {
        "datasets": [str(path) for path in dataset_dirs],
        "sensors": sensors,
        "fusion": args.fusion,
        "classifier": args.classifier,
        "classifier_label": classifier_label(args.classifier),
        "classifier_params": classifier_params,
        "model": str(model_out),
        "confusion_matrix": str(confusion_out),
        "accuracy": accuracy,
        "per_class_recall": per_class_recall,
        "class_counts": class_counts,
        "n_examples": int(len(labels)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "split_method": split_method,
        "train_collectors": train_collectors,
        "test_collectors": test_collectors_used,
        "source_datasets": sorted(set(sources)),
        "session_dirs": session_dirs,
        "missing_datasets": missing_datasets,
        "skipped": skipped,
        "classification_report": classification_report(y_test, predictions, labels=labels_order, zero_division=0, output_dict=True),
    }
    summary_path = model_out.with_name(model_out.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
