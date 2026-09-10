import copy
from typing import Any, Dict, List, Optional

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


def run_few_shot_experiment(
    dataset: CognitiveLoadDataset,
    base_model: torch.nn.Module,
    trainer_kwargs: Dict[str, Any],
    calib_ratio: float = 0.1,
) -> Dict[str, Any]:
    subjects = sorted(set(dataset.subject_id))
    fold_results: List[Dict[str, Any]] = []
    device = trainer_kwargs.get("device", torch.device("cpu"))

    for subject in subjects:
        subject_indices = [i for i, sid in enumerate(dataset.subject_id) if sid == subject]

        calib_num = int(len(subject_indices) * calib_ratio)
        if calib_num == 0:
            calib_num = 1

        calib_indices = subject_indices[:calib_num]
        test_indices = subject_indices[calib_num:]

        calib_subset = Subset(dataset, calib_indices)
        test_subset = Subset(dataset, test_indices)

        batch_size = trainer_kwargs.get("batch_size", 32)
        calib_loader = DataLoader(calib_subset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

        model = copy.deepcopy(base_model).to(device)

        optimizer_cls = trainer_kwargs.get("optimizer_cls", torch.optim.Adam)
        optimizer_kwargs = trainer_kwargs.get("optimizer_kwargs", {"lr": 0.001})
        optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)

        criterion = trainer_kwargs.get("criterion", torch.nn.BCEWithLogitsLoss())

        _, history = train_model(
            model=model,
            train_loader=calib_loader,
            val_loader=test_loader,
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
            "test_subject": subject,
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
        "paradigm": "few_shot",
        "folds": fold_results,
        "aggregates": aggregates,
    }
