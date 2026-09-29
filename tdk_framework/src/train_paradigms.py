import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Ensure the src directory is importable when the script is executed directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.gated_fusion import GatedFusionModel


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def flatten_dynamic(dynamic_arr, target_len: int) -> torch.Tensor:
    flat = torch.tensor(dynamic_arr, dtype=torch.float32).flatten()
    if flat.numel() < target_len:
        padding = torch.zeros(target_len - flat.numel())
        flat = torch.cat([flat, padding])
    else:
        flat = flat[:target_len]
    return flat


def build_tensors(samples, dynamic_dim: int, static_dim: int):
    dyns = []
    stats = []
    labels = []

    for sample in samples:
        dyn = flatten_dynamic(sample["dynamic"], dynamic_dim)
        stat = torch.tensor(sample["static"], dtype=torch.float32)
        label = torch.tensor(float(sample["label"]), dtype=torch.float32)

        dyns.append(dyn)
        stats.append(stat)
        labels.append(label)

    return torch.stack(dyns), torch.stack(stats), torch.stack(labels)


def make_loader(samples, dynamic_dim: int, static_dim: int, batch_size: int = 16, shuffle: bool = True):
    if len(samples) == 0:
        raise ValueError("Cannot create DataLoader from empty sample list")

    dyns, stats, labels = build_tensors(samples, dynamic_dim, static_dim)

    # BatchNorm1d requires at least 2 samples in a batch during training.
    # If we have a single training sample, duplicate it to avoid runtime errors.
    if len(dyns) < 2:
        dyns = torch.cat([dyns, dyns], dim=0)
        stats = torch.cat([stats, stats], dim=0)
        labels = torch.cat([labels, labels], dim=0)

    dataset = TensorDataset(dyns, stats, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_model(model: torch.nn.Module, train_loader: DataLoader, epochs: int = 20, lr: float = 1e-3, device: torch.device = torch.device("cpu")):
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        for dyn, stat, y in train_loader:
            dyn = dyn.to(device)
            stat = stat.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(dyn, stat)

            if logits.ndim > 1:
                logits = logits.squeeze(1)

            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

    model.to("cpu")
    return model


def predict(model: torch.nn.Module, test_loader: DataLoader, device: torch.device = torch.device("cpu")):
    model.to(device)
    model.eval()
    probs = []
    targets = []

    with torch.no_grad():
        for dyn, stat, y in test_loader:
            dyn = dyn.to(device)
            stat = stat.to(device)
            y = y.to(device)

            logits = model(dyn, stat)
            if logits.ndim > 1:
                logits = logits.squeeze(1)

            prob = torch.sigmoid(logits)
            probs.extend(prob.cpu().numpy().tolist())
            targets.extend(y.cpu().numpy().tolist())

    model.to("cpu")
    return np.array(probs), np.array(targets)


def safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")


def split_stratified(samples, test_size, random_state=42):
    indices = np.arange(len(samples))
    labels = np.array([sample["label"] for sample in samples])
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=labels,
        random_state=random_state,
    )
    train_samples = [samples[i] for i in train_idx]
    test_samples = [samples[i] for i in test_idx]
    return train_samples, test_samples


def main():
    parser = argparse.ArgumentParser(description="Run training paradigm comparison for Gated Fusion models")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to a processed .pt dataset")
    args = parser.parse_args()

    dataset_path = args.dataset
    dataset = torch.load(dataset_path, weights_only=False)

    if not isinstance(dataset, list):
        raise ValueError("Processed dataset must be a list of sample dictionaries")

    if len(dataset) == 0:
        raise ValueError("Dataset is empty")

    all_labels_for_dist = [int(sample["label"]) for sample in dataset]
    unique_labels, counts = np.unique(all_labels_for_dist, return_counts=True)
    print("Global class distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"  Class {label}: {count} samples")

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine feature dimensions from the data.
    max_rows = max(len(sample["dynamic"]) for sample in dataset)
    num_dynamic_feats = dataset[0]["dynamic"].shape[1]
    dynamic_dim = max_rows * num_dynamic_feats
    static_dim = len(dataset[0]["static"])

    # Organize samples by subject.
    samples_by_subject = {}
    for sample in dataset:
        subj = sample["subject"]
        samples_by_subject.setdefault(subj, []).append(sample)

    subjects = list(samples_by_subject.keys())
    if len(subjects) == 0:
        raise ValueError("No subjects found in dataset")

    # -----------------------------------------------------------------
    # Paradigm A: Zero-shot LOSO
    # -----------------------------------------------------------------
    zero_shot_y_true_all = []
    zero_shot_y_score_all = []
    zero_shot_models = {}

    for test_subj in subjects:
        train_samples = []
        for subj in subjects:
            if subj != test_subj:
                train_samples.extend(samples_by_subject[subj])

        test_samples = samples_by_subject[test_subj]

        if not train_samples or not test_samples:
            continue

        train_loader = make_loader(train_samples, dynamic_dim, static_dim, shuffle=True)
        test_loader = make_loader(test_samples, dynamic_dim, static_dim, shuffle=False)

        model = GatedFusionModel(num_dynamic_features=dynamic_dim, num_static_features=static_dim)
        model = train_model(model, train_loader, device=device)

        # Store this base model for the corresponding few-shot fold.
        zero_shot_models[test_subj] = {k: v.clone().cpu() for k, v in model.state_dict().items()}

        probs, targets = predict(model, test_loader, device=device)
        zero_shot_y_true_all.extend(targets.tolist())
        zero_shot_y_score_all.extend(probs.tolist())

    # -----------------------------------------------------------------
    # Paradigm B: Intra-subject (Personalized)
    # -----------------------------------------------------------------
    intra_y_true_all = []
    intra_y_score_all = []

    for subj in subjects:
        subject_samples = samples_by_subject[subj]
        n = len(subject_samples)

        if n < 2:
            continue

        subject_labels = np.array([sample["label"] for sample in subject_samples])
        if len(np.unique(subject_labels)) < 2:
            continue

        try:
            train_samples, test_samples = split_stratified(
                subject_samples,
                test_size=0.2,
                random_state=42,
            )
        except ValueError:
            continue

        if not train_samples or not test_samples:
            continue

        train_loader = make_loader(train_samples, dynamic_dim, static_dim, shuffle=True)
        test_loader = make_loader(test_samples, dynamic_dim, static_dim, shuffle=False)

        model = GatedFusionModel(num_dynamic_features=dynamic_dim, num_static_features=static_dim)
        model = train_model(model, train_loader, device=device)

        probs, targets = predict(model, test_loader, device=device)
        intra_y_true_all.extend(targets.tolist())
        intra_y_score_all.extend(probs.tolist())

    # -----------------------------------------------------------------
    # Paradigm C: Few-shot (Fine-tuning)
    # -----------------------------------------------------------------
    few_shot_y_true_all = []
    few_shot_y_score_all = []

    for subj in subjects:
        base_state = zero_shot_models.get(subj)
        if base_state is None:
            continue

        subject_samples = samples_by_subject[subj]
        n = len(subject_samples)

        if n < 2:
            continue

        subject_labels = np.array([sample["label"] for sample in subject_samples])
        if len(np.unique(subject_labels)) < 2:
            continue

        try:
            calibration_samples, test_samples = split_stratified(
                subject_samples,
                test_size=0.8,
                random_state=42,
            )
        except ValueError:
            continue

        if not calibration_samples or not test_samples:
            continue

        calib_loader = make_loader(calibration_samples, dynamic_dim, static_dim, shuffle=True)
        test_loader = make_loader(test_samples, dynamic_dim, static_dim, shuffle=False)

        model = GatedFusionModel(num_dynamic_features=dynamic_dim, num_static_features=static_dim)
        model.load_state_dict(base_state)
        model = train_model(model, calib_loader, epochs=10, device=device)

        probs, targets = predict(model, test_loader, device=device)
        few_shot_y_true_all.extend(targets.tolist())
        few_shot_y_score_all.extend(probs.tolist())

    # Convert global predictions to numpy arrays for evaluation.
    zero_shot_y_true_all = np.array(zero_shot_y_true_all)
    zero_shot_y_score_all = np.array(zero_shot_y_score_all)
    intra_y_true_all = np.array(intra_y_true_all)
    intra_y_score_all = np.array(intra_y_score_all)
    few_shot_y_true_all = np.array(few_shot_y_true_all)
    few_shot_y_score_all = np.array(few_shot_y_score_all)

    # Compute final global AUROC for each paradigm.
    avg_zero_shot = safe_auroc(zero_shot_y_true_all, zero_shot_y_score_all)
    avg_intra = safe_auroc(intra_y_true_all, intra_y_score_all)
    avg_few_shot = safe_auroc(few_shot_y_true_all, few_shot_y_score_all)

    # -----------------------------------------------------------------
    # Save and print summary
    # -----------------------------------------------------------------
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    summary = {
        "dataset": str(dataset_path),
        "average_auroc": {
            "zero_shot": avg_zero_shot,
            "intra_subject": avg_intra,
            "few_shot": avg_few_shot,
        },
    }

    output_json = results_dir / "paradigms_summary.json"
    with open(output_json, "w") as f:
        json.dump(summary, f, indent=4)

    print("\nParadigm AUROC Summary")
    print("-" * 40)
    print(f"{'Paradigm':<20}{'Global AUROC':<20}")
    print(f"{'Zero-shot (LOSO)':<20}{avg_zero_shot:<20.4f}")
    print(f"{'Intra-subject':<20}{avg_intra:<20.4f}")
    print(f"{'Few-shot':<20}{avg_few_shot:<20.4f}")
    print(f"\nSummary saved to {output_json}")


if __name__ == "__main__":
    main()
