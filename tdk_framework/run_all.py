import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

project_root = Path(__file__).resolve().parent.parent
tdk_framework_dir = Path(__file__).resolve().parent

for p in (str(project_root), str(tdk_framework_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.data.preprocess_tdk import process_dataset as process_tdk_dataset
from src.data.preprocess_wesad import process_dataset as process_wesad_dataset
from src.training.core_trainer import train_model
from src.experiments.zero_shot import run_zero_shot_experiment
from src.experiments.intra_subject import run_intra_subject_experiment
from src.experiments.few_shot import run_few_shot_experiment
from tdk_framework.src.models.baseline_model import BaselineModel
from tdk_framework.src.models.gated_fusion import GatedFusionModel


MODEL_REGISTRY = {
    "baseline": BaselineModel,
    "gated_fusion": GatedFusionModel,
}


def build_model_kwargs(model_name: str, dataset) -> dict:
    if model_name == "gated_fusion":
        return {
            "num_dynamic_features": dataset.num_dynamic_features,
            "num_static_features": dataset.num_static_features,
        }
    return {"num_dynamic_features": dataset.num_dynamic_features}


def build_model(model_name: str, dataset) -> torch.nn.Module:
    model_class = MODEL_REGISTRY[model_name]
    return model_class(**build_model_kwargs(model_name, dataset))


def find_tdk_raw_data_path(tdk_framework_dir: Path, project_root: Path) -> Path:
    raw_candidate = tdk_framework_dir / "data" / "raw" / "processed_dataset_calibrated.npz"
    if raw_candidate.exists() and raw_candidate.is_file():
        return raw_candidate

    default_candidate = project_root / "data" / "processed" / "processed_dataset_calibrated.npz"
    if default_candidate.exists() and default_candidate.is_file():
        return default_candidate

    raw_dir = tdk_framework_dir / "data" / "raw"
    if raw_dir.exists() and raw_dir.is_dir():
        npz_files = [p for p in raw_dir.glob("*.npz") if p.is_file()]
        if npz_files:
            return npz_files[0]

    raise FileNotFoundError(
        "No raw .npz dataset found. Tried "
        f"{raw_candidate}, {default_candidate}, and any .npz in {raw_dir}"
    )


def find_wesad_raw_data_path(tdk_framework_dir: Path, project_root: Path) -> Path:
    raw_candidate = tdk_framework_dir / "data" / "raw" / "wesad"
    if raw_candidate.exists() and raw_candidate.is_dir():
        return raw_candidate

    default_candidate = project_root / "data" / "raw" / "wesad"
    if default_candidate.exists() and default_candidate.is_dir():
        return default_candidate

    raise FileNotFoundError(
        "No WESAD raw data directory found. Tried "
        f"{raw_candidate} and {default_candidate}"
    )


def resolve_dataset(dataset_name: str, tdk_framework_dir: Path, project_root: Path):
    if dataset_name == "tdk":
        return find_tdk_raw_data_path(tdk_framework_dir, project_root), process_tdk_dataset
    if dataset_name == "wesad":
        return find_wesad_raw_data_path(tdk_framework_dir, project_root), process_wesad_dataset
    raise ValueError(f"Unknown dataset: {dataset_name}")


def train_base_model(dataset, model, trainer_kwargs):
    device = trainer_kwargs["device"]
    model = model.to(device)

    full_loader = DataLoader(
        dataset,
        batch_size=trainer_kwargs["batch_size"],
        shuffle=True,
    )

    optimizer_cls = trainer_kwargs["optimizer_cls"]
    optimizer_kwargs = trainer_kwargs.get("optimizer_kwargs", {"lr": 0.001})
    optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)

    best_state_dict, _ = train_model(
        model=model,
        train_loader=full_loader,
        val_loader=full_loader,
        criterion=trainer_kwargs["criterion"],
        optimizer=optimizer,
        epochs=trainer_kwargs["epochs"],
        device=device,
        metric_to_monitor=trainer_kwargs.get("metric_to_monitor", "loss"),
        early_stopping_patience=trainer_kwargs.get("early_stopping_patience"),
        verbose=trainer_kwargs.get("verbose", False),
    )

    model.load_state_dict(best_state_dict)
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Run zero-shot, intra-subject and few-shot experiments."
    )
    parser.add_argument(
        "--dataset",
        choices=["tdk", "wesad"],
        default="tdk",
        help="Which dataset to preprocess and evaluate.",
    )
    parser.add_argument(
        "--model",
        choices=["baseline", "gated_fusion"],
        default="baseline",
        help="Which model architecture to evaluate.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer_kwargs = {
        "device": device,
        "batch_size": 32,
        "epochs": 20,
        "optimizer_cls": torch.optim.Adam,
        "optimizer_kwargs": {"lr": 0.001},
        "criterion": torch.nn.BCEWithLogitsLoss(),
        "metric_to_monitor": "loss",
        "early_stopping_patience": None,
        "verbose": False,
    }

    raw_data_path, process_dataset = resolve_dataset(
        args.dataset, tdk_framework_dir, project_root
    )
    output_dir = tdk_framework_dir / "data" / "processed"
    dataset_path = process_dataset(raw_data_path, output_dir)
    dataset = torch.load(dataset_path, weights_only=False)

    model_class = MODEL_REGISTRY[args.model]
    model_kwargs = build_model_kwargs(args.model, dataset)

    zero_shot_result = run_zero_shot_experiment(
        dataset=dataset,
        model_class=model_class,
        trainer_kwargs=trainer_kwargs,
        model_kwargs=model_kwargs,
    )

    base_model = train_base_model(
        dataset=dataset,
        model=build_model(args.model, dataset),
        trainer_kwargs=trainer_kwargs,
    )

    intra_subject_result = run_intra_subject_experiment(
        dataset=dataset,
        model_class=model_class,
        trainer_kwargs=trainer_kwargs,
        model_kwargs=model_kwargs,
    )

    few_shot_result = run_few_shot_experiment(
        dataset=dataset,
        base_model=base_model,
        trainer_kwargs=trainer_kwargs,
    )

    results = {
        "zero_shot": zero_shot_result,
        "intra_subject": intra_subject_result,
        "few_shot": few_shot_result,
    }

    results_dir = tdk_framework_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"experiment_results_{args.dataset}_{args.model}.json"

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
