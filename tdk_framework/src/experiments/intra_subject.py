from typing import Any, Dict, List, Optional, Type

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ..data.base_dataset import CognitiveLoadDataset
from ..training.core_trainer import train_model


def _get_last_metrics(history: Dict[str, List[Any]]) -> Dict[str, Optional[float]]:
    return {
        "balanced_acc": history["val_balanced_acc"][-1] if history["val_balanced_acc"] else None,
        "macro_f1": history["val_f1"][-1] if history["val_f1"] else None,
        "auroc": history["val_auroc"][-1] if history["val_auroc"] else None,
    }


def run_intra_subject_experiment(
    dataset: CognitiveLoadDataset,
    model_class: Type[torch.nn.Module],
    trainer_kwargs: Dict[str, Any],
    train_ratio: float = 0.8,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if model_kwargs is None:
        model_kwargs = {}
    subjects = sorted(set(dataset.subject_id))
    fold_results: List[Dict[str, Any]] = []
    device = trainer_kwargs.get("device", torch.device("cpu"))

    for subject in subjects:
        subject_indices = [i for i, sid in enumerate(dataset.subject_id) if sid == subject]

        split = int(len(subject_indices) * train_ratio)
        if split == 0:
            split = 1

        train_indices = subject_indices[:split]
        val_indices = subject_indices[split:]

        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)

        batch_size = trainer_kwargs.get("batch_size", 32)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        model = model_class(**model_kwargs).to(device)
        optimizer_cls = trainer_kwargs.get("optimizer_cls", torch.optim.Adam)
        optimizer_kwargs = trainer_kwargs.get("optimizer_kwargs", {"lr": 0.001})
        optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)

        criterion = trainer_kwargs.get("criterion", torch.nn.BCEWithLogitsLoss())

        _, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            epochs=trainer_kwargs.get("epochs", 20),
            device=device,
            metric_to_monitor=trainer_kwargs.get("metric_to_monitor", "loss"),
            early_stopping_patience=trainer_kwargs.get("early_stopping_patience"),
            verbose=trainer_kwargs.get("verbose", False),
        )

        metrics = _get_last_metrics(history)
        fold_results.append({
            "subject": subject,
            **metrics,
        })

    bal_accs = [r["balanced_acc"] for r in fold_results if r["balanced_acc"] is not None]
    f1s = [r["macro_f1"] for r in fold_results if r["macro_f1"] is not None]
    aurocs = [r["auroc"] for r in fold_results if r["auroc"] is not None]

    aggregates = {
        "bal_acc_mean": float(np.mean(bal_accs)) if bal_accs else None,
        "bal_acc_std": float(np.std(bal_accs)) if bal_accs else None,
        "f1_mean": float(np.mean(f1s)) if f1s else None,
        "f1_std": float(np.std(f1s)) if f1s else None,
        "auroc_mean": float(np.mean(aurocs)) if aurocs else None,
        "auroc_std": float(np.std(aurocs)) if aurocs else None,
    }

    return {
        "paradigm": "intra_subject",
        "folds": fold_results,
        "aggregates": aggregates,
    }
