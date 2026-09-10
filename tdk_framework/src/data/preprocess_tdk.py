import argparse
from pathlib import Path

import numpy as np
import torch

from .base_dataset import CognitiveLoadDataset


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_DATA = PROJECT_ROOT / "data" / "processed" / "processed_dataset_calibrated.npz"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def binarize_labels(subjects: np.ndarray, y: np.ndarray) -> np.ndarray:
    y_binary = np.zeros_like(y, dtype=float)

    for subject in np.unique(subjects):
        subject_mask = subjects == subject
        median = np.median(y[subject_mask])
        y_binary[subject_mask] = (y[subject_mask] >= median).astype(float)

    return y_binary


def extract_basic_features(X: np.ndarray) -> np.ndarray:
    if X.ndim != 3:
        raise ValueError(f"X_dynamic must be a 3D array with shape (N, T, C), got {X.shape}")

    means = np.mean(X, axis=1)
    stds = np.std(X, axis=1)

    features = np.concatenate([means, stds], axis=1)
    return features


def process_dataset(raw_data_path: Path, output_dir: Path) -> Path:
    if not raw_data_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {raw_data_path}")

    data = np.load(raw_data_path, allow_pickle=True)

    required_keys = {"X_dynamic", "X_static", "y", "subjects"}
    missing_keys = required_keys - set(data.keys())
    if missing_keys:
        raise KeyError(f"Missing keys in raw data: {missing_keys}")

    X_dynamic = data["X_dynamic"]
    X_static = data["X_static"]
    y_raw = data["y"]
    subjects = data["subjects"]

    print("Extracting basic features (mean and std per channel)...")
    features = extract_basic_features(X_dynamic)

    print("Binarizing labels using subject-level median split...")
    y_binary = binarize_labels(subjects, y_raw)

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = CognitiveLoadDataset(
        dynamic_data=features,
        static_data=X_static,
        labels=y_binary,
        subject_ids=subjects,
    )

    output_path = output_dir / "cognitive_load_dataset.pt"
    torch.save(dataset, output_path)

    print(f"Dataset cached to: {output_path}")
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Dynamic feature dim: {dataset.num_dynamic_features}")
    print(f"Static feature dim: {dataset.num_static_features}")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process TDK raw dataset and cache CognitiveLoadDataset."
    )
    parser.add_argument(
        "--raw-data",
        type=Path,
        default=DEFAULT_RAW_DATA,
        help="Path to raw .npz file containing X_dynamic, X_static, y, subjects.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the cached dataset will be saved.",
    )

    args = parser.parse_args()
    process_dataset(args.raw_data, args.output_dir)


if __name__ == "__main__":
    main()
