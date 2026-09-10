import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score


def _prepare_batch(batch: Tuple[torch.Tensor, ...], device: torch.device):
    if len(batch) == 4:
        x_dyn, x_stat, y, _ = batch
    elif len(batch) == 3:
        x_dyn, x_stat, y = batch
    elif len(batch) == 2:
        x_dyn, y = batch
        x_stat = None
    else:
        raise ValueError(f"Unexpected batch size: {len(batch)}")

    x_dyn = x_dyn.to(device, non_blocking=True)
    if x_stat is not None:
        x_stat = x_stat.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)

    return x_dyn, x_stat, y


def _forward_pass(model: nn.Module, x_dyn: torch.Tensor, x_stat: Optional[torch.Tensor]):
    if x_stat is not None:
        return model(x_dyn, x_stat)
    return model(x_dyn)


def _compute_binary_metrics(probs: np.ndarray, targets: np.ndarray) -> Dict[str, Optional[float]]:
    preds = (probs >= 0.5).astype(int)

    if len(np.unique(targets)) < 2:
        auroc = None
        balanced_acc = balanced_accuracy_score(targets, preds)
        f1 = f1_score(targets, preds, average="macro", zero_division=0)
        return {"auroc": auroc, "balanced_acc": balanced_acc, "f1": f1}

    try:
        auroc = roc_auc_score(targets, probs)
    except ValueError:
        auroc = None

    balanced_acc = balanced_accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average="macro", zero_division=0)

    return {"auroc": auroc, "balanced_acc": balanced_acc, "f1": f1}


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
    metric_to_monitor: str = "loss",
    early_stopping_patience: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[Any]]]:
    """
    Paradigm-agnostic training loop.

    Returns:
        best_state_dict: state dict of the best validation checkpoint.
        history: per-epoch metrics (train_loss, val_loss, val_balanced_acc,
                 val_f1, val_auroc).
    """

    best_state_dict = copy.deepcopy(model.state_dict())
    best_metric = float("inf") if metric_to_monitor == "loss" else -float("inf")
    epochs_no_improve = 0

    train_losses: List[float] = []
    val_losses: List[float] = []
    val_balanced_accs: List[Optional[float]] = []
    val_f1s: List[Optional[float]] = []
    val_aurocs: List[Optional[float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        n_train = 0

        for batch in train_loader:
            optimizer.zero_grad()
            x_dyn, x_stat, y = _prepare_batch(batch, device)
            outputs = _forward_pass(model, x_dyn, x_stat)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * len(y)
            n_train += len(y)

        avg_train_loss = total_train_loss / n_train if n_train > 0 else 0.0

        model.eval()
        total_val_loss = 0.0
        n_val = 0
        all_val_probs: List[float] = []
        all_val_targets: List[int] = []

        with torch.no_grad():
            for batch in val_loader:
                x_dyn, x_stat, y = _prepare_batch(batch, device)
                outputs = _forward_pass(model, x_dyn, x_stat)
                loss = criterion(outputs, y)

                total_val_loss += loss.item() * len(y)
                n_val += len(y)

                probs = torch.sigmoid(outputs).cpu().numpy().reshape(-1)
                all_val_probs.extend(probs.tolist())
                all_val_targets.extend(y.cpu().numpy().tolist())

        avg_val_loss = total_val_loss / n_val if n_val > 0 else 0.0

        if len(all_val_targets) > 0:
            val_metrics = _compute_binary_metrics(
                np.array(all_val_probs), np.array(all_val_targets)
            )
        else:
            val_metrics = {"auroc": None, "balanced_acc": None, "f1": None}

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_balanced_accs.append(val_metrics["balanced_acc"])
        val_f1s.append(val_metrics["f1"])
        val_aurocs.append(val_metrics["auroc"])

        if metric_to_monitor == "loss":
            current_metric = avg_val_loss
            is_better = current_metric < best_metric
        elif metric_to_monitor == "auroc":
            current_metric = val_metrics["auroc"] if val_metrics["auroc"] is not None else -float("inf")
            is_better = current_metric > best_metric
        elif metric_to_monitor == "balanced_acc":
            current_metric = val_metrics["balanced_acc"] if val_metrics["balanced_acc"] is not None else -float("inf")
            is_better = current_metric > best_metric
        elif metric_to_monitor == "f1":
            current_metric = val_metrics["f1"] if val_metrics["f1"] is not None else -float("inf")
            is_better = current_metric > best_metric
        else:
            raise ValueError(
                f"metric_to_monitor must be one of 'loss', 'auroc', 'balanced_acc', 'f1'. Got {metric_to_monitor}"
            )

        if is_better:
            best_metric = current_metric
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose:
            print(
                f"Epoch {epoch}/{epochs} | train_loss: {avg_train_loss:.4f} | "
                f"val_loss: {avg_val_loss:.4f} | val_bal_acc: {val_metrics['balanced_acc']} | "
                f"val_f1: {val_metrics['f1']} | val_auroc: {val_metrics['auroc']}"
            )

        if early_stopping_patience is not None and epochs_no_improve >= early_stopping_patience:
            if verbose:
                print(f"Early stopping activated after {epochs_no_improve} epochs without improvement.")
            break

    history = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "val_balanced_acc": val_balanced_accs,
        "val_f1": val_f1s,
        "val_auroc": val_aurocs,
    }

    return best_state_dict, history
