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

from src.data.preprocess_tdk import process_dataset
from src.training.core_trainer import train_model
from src.experiments.zero_shot import run_zero_shot_experiment
from src.experiments.intra_subject import run_intra_subject_experiment
from src.experiments.few_shot import run_few_shot_experiment
from models.baseline_model import BaselineModel


def train_base_model(dataset, model_class, trainer_kwargs):
    device = trainer_kwargs["device"]
    model = model_class().to(device)

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

    dataset_path = process_dataset()
    dataset = torch.load(dataset_path, weights_only=False)

    zero_shot_result = run_zero_shot_experiment(
        dataset=dataset,
        model_class=BaselineModel,
        trainer_kwargs=trainer_kwargs,
    )

    base_model = train_base_model(
        dataset=dataset,
        model_class=BaselineModel,
        trainer_kwargs=trainer_kwargs,
    )

    intra_subject_result = run_intra_subject_experiment(
        dataset=dataset,
        model_class=BaselineModel,
        trainer_kwargs=trainer_kwargs,
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
    results_file = results_dir / "experiment_results.json"

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
