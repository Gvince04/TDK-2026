import argparse
import pickle
from pathlib import Path

import numpy as np
import torch

from tdk_framework.src.data.base_dataset import CognitiveLoadDataset


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_DATA_DIR = PROJECT_ROOT / "tdk_framework" / "data" / "raw" / "wesad"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

CHEST_SAMPLING_RATE = 700
WINDOW_SECONDS = 60
STRIDE_SECONDS = 30
NUM_STATIC_FEATURES = 5


def _window_starts(num_samples: int, window_size: int, stride: int) -> range:
    if num_samples < window_size:
        return range(0)
    return range(0, num_samples - window_size + 1, stride)


def _extract_window_features(signal: np.ndarray) -> np.ndarray:
    means = np.mean(signal, axis=1)
    stds = np.std(signal, axis=1)
    mins = np.min(signal, axis=1)
    maxs = np.max(signal, axis=1)
    return np.stack([means, stds, mins, maxs], axis=1)


def _subject_seed(subject_id: str) -> int:
    digits = "".join(ch for ch in subject_id if ch.isdigit())
    return int(digits) if digits else 0


def _static_vector(subject_id: str) -> np.ndarray:
    rng = np.random.default_rng(_subject_seed(subject_id))
    return rng.random(NUM_STATIC_FEATURES).astype(np.float32)


def _load_subject(subject_dir: Path) -> dict:
    pkl_files = sorted(subject_dir.glob("*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl file found in {subject_dir}")

    with open(pkl_files[0], "rb") as f:
        return pickle.load(f, encoding="latin1")


def _process_subject(subject_id: str, subject_dir: Path):
    data = _load_subject(subject_dir)

    chest = data["signal"]["chest"]
    ecg = np.asarray(chest["ECG"]).reshape(-1)
    eda = np.asarray(chest["EDA"]).reshape(-1)
    labels = np.asarray(data["label"]).reshape(-1)

    num_samples = min(len(ecg), len(eda), len(labels))
    ecg = ecg[:num_samples]
    eda = eda[:num_samples]
    labels = labels[:num_samples]

    window_size = WINDOW_SECONDS * CHEST_SAMPLING_RATE
    stride = STRIDE_SECONDS * CHEST_SAMPLING_RATE

    dynamic_rows = []
    window_labels = []

    for start in _window_starts(num_samples, window_size, stride):
        end = start + window_size
        window_labels_arr = labels[start:end]

        valid = np.isin(window_labels_arr, (1, 2))
        if not np.all(valid):
            continue

        ecg_window = ecg[start:end].reshape(1, -1)
        eda_window = eda[start:end].reshape(1, -1)

        ecg_features = _extract_window_features(ecg_window).reshape(-1)
        eda_features = _extract_window_features(eda_window).reshape(-1)

        dynamic_rows.append(np.concatenate([ecg_features, eda_features]))
        window_labels.append(0 if window_labels_arr[0] == 1 else 1)

    if not dynamic_rows:
        return None

    dynamic = np.asarray(dynamic_rows, dtype=np.float32)
    y = np.asarray(window_labels, dtype=float)
    static = np.tile(_static_vector(subject_id), (len(dynamic), 1))
    subjects = np.array([subject_id] * len(dynamic))

    return dynamic, static, y, subjects


def process_dataset(raw_data_dir: Path, output_dir: Path) -> Path:
    if not raw_data_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_data_dir}")

    subject_dirs = sorted(
        [p for p in raw_data_dir.iterdir() if p.is_dir() and p.name.startswith("S")],
        key=lambda p: _subject_seed(p.name),
    )
    if not subject_dirs:
        raise FileNotFoundError(f"No subject folders found in {raw_data_dir}")

    dynamic_parts = []
    static_parts = []
    label_parts = []
    subject_parts = []

    for subject_dir in subject_dirs:
        subject_id = subject_dir.name
        print(f"Processing subject {subject_id}...")
        result = _process_subject(subject_id, subject_dir)
        if result is None:
            print(f"Skipping subject {subject_id}: no valid windows")
            continue

        dynamic, static, y, subjects = result
        dynamic_parts.append(dynamic)
        static_parts.append(static)
        label_parts.append(y)
        subject_parts.append(subjects)

    if not dynamic_parts:
        raise ValueError("No valid windows extracted from any subject")

    dynamic_data = np.concatenate(dynamic_parts, axis=0)
    static_data = np.concatenate(static_parts, axis=0)
    labels = np.concatenate(label_parts, axis=0)
    subject_ids = np.concatenate(subject_parts, axis=0)

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = CognitiveLoadDataset(
        dynamic_data=dynamic_data,
        static_data=static_data,
        labels=labels,
        subject_ids=subject_ids,
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
        description="Process WESAD dataset and cache CognitiveLoadDataset."
    )
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=DEFAULT_RAW_DATA_DIR,
        help="Directory containing WESAD subject folders (S2, S3, ...).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the cached dataset will be saved.",
    )

    args = parser.parse_args()
    process_dataset(args.raw_data_dir, args.output_dir)


if __name__ == "__main__":
    main()
