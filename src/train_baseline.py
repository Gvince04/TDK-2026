import sys
import os
import argparse
import json
import shutil

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from models.baseline_model import BaselineModel

class CLAREDataset(Dataset):
    def __init__(self, dynamic_data, static_data, labels):
        self.dynamic = torch.FloatTensor(dynamic_data)
        self.static = torch.LongTensor(static_data)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.dynamic[idx], self.static[idx], self.labels[idx]

def extract_features(X):
    pupil_left_mean = np.mean(X[:, :, 0], axis=1)
    pupil_left_std = np.std(X[:, :, 0], axis=1)
    pupil_right_mean = np.mean(X[:, :, 1], axis=1)
    pupil_right_std = np.std(X[:, :, 1], axis=1)

    gaze_x = X[:, :, 2]
    gaze_y = X[:, :, 3]

    gaze_dispersion_x = np.std(gaze_x, axis=1)
    gaze_dispersion_y = np.std(gaze_y, axis=1)

    diff_x = np.diff(gaze_x, axis=1)
    diff_y = np.diff(gaze_y, axis=1)
    velocity = np.sqrt(diff_x ** 2 + diff_y ** 2)

    gaze_velocity_mean = np.mean(velocity, axis=1)
    gaze_velocity_std = np.std(velocity, axis=1)

    features = np.stack([
        pupil_left_mean,
        pupil_left_std,
        pupil_right_mean,
        pupil_right_std,
        gaze_dispersion_x,
        gaze_dispersion_y,
        gaze_velocity_mean,
        gaze_velocity_std
    ], axis=1)

    return features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fast', action='store_true')
    args = parser.parse_args()
    max_subjects = 3 if args.fast else None

    print("--- Adatok betöltése ---")
    data_path = os.path.join(project_root, 'data', 'processed', 'processed_dataset_calibrated.npz')
    data = np.load(data_path, allow_pickle=True)
    
    X_dynamic = data['X_dynamic']
    X_static = data['X_static']
    y_raw = data['y']
    subjects = data['subjects']

    y_binary = np.zeros_like(y_raw, dtype=int)
    for subj in np.unique(subjects):
        subj_mask = (subjects == subj)
        subj_median = np.median(y_raw[subj_mask])
        y_binary[subj_mask] = (y_raw[subj_mask] >= subj_median).astype(int)

    unique_subjects = np.unique(subjects)
    
    print(f"Alany-szintű binarizálás kész! 0-s osztály: {sum(y_binary==0)}, 1-es osztály: {sum(y_binary==1)}")
    print(f"Adathalmaz: {len(X_dynamic)} ablak, {len(unique_subjects)} alany.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Használt eszköz: {device}")

    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    existing_runs = [d for d in os.listdir(results_dir)
                     if os.path.isdir(os.path.join(results_dir, d)) and d.startswith('run_')]
    run_numbers = []
    for d in existing_runs:
        try:
            run_numbers.append(int(d.split('_')[1]))
        except (IndexError, ValueError):
            pass
    next_num = max(run_numbers) + 1 if run_numbers else 1
    run_dir = os.path.join(results_dir, f'run_{next_num:03d}')
    os.makedirs(run_dir, exist_ok=True)
    print(f"Eredmények mappája: {run_dir}")

    config = {
        'model': 'BaselineModel',
        'fast': args.fast,
        'max_subjects': max_subjects,
        'batch_size': 32,
        'epochs': 25,
        'learning_rate': 0.0005,
        'weight_decay': 1e-3,
        'num_dynamic_features': 8,
        'num_static_levels': None
    }

    unique_subjects_loop = unique_subjects if max_subjects is None else unique_subjects[:max_subjects]
    if args.fast:
        print(f"Fast mode aktiv: csak {len(unique_subjects_loop)} alany feldolgozása")

    all_bal_acc, all_f1, all_auroc = [], [], []
    fold_metrics = []
    best_score = -float('inf')
    best_fold_model_path = None

    for test_subject in unique_subjects_loop:
        train_mask = subjects != test_subject
        test_mask = subjects == test_subject

        X_dyn_train_raw = X_dynamic[train_mask].copy()
        X_dyn_test_raw = X_dynamic[test_mask].copy()

        train_features = extract_features(X_dyn_train_raw)
        test_features = extract_features(X_dyn_test_raw)

        mean = train_features.mean(axis=0, keepdims=True)
        std = train_features.std(axis=0, keepdims=True) + 1e-8

        X_dyn_train = (train_features - mean) / std
        X_dyn_test = (test_features - mean) / std

        train_dataset = CLAREDataset(X_dyn_train, X_static[train_mask], y_binary[train_mask])
        test_dataset = CLAREDataset(X_dyn_test, X_static[test_mask], y_binary[test_mask])

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        model = BaselineModel(num_dynamic_features=8).to(device)
        
        pos_weight_val = sum(y_binary[train_mask] == 0) / sum(y_binary[train_mask] == 1)
        pos_weight_tensor = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-3)

        epochs = 25
        model.train()
        for epoch in range(epochs):
            for dyn_x, stat_x, targets in train_loader:
                dyn_x, stat_x, targets = dyn_x.to(device), stat_x.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(dyn_x, stat_x)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

        fold_model_path = os.path.join(run_dir, f'model_fold_{test_subject}.pth')
        torch.save(model.state_dict(), fold_model_path)

        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for dyn_x, stat_x, targets in test_loader:
                dyn_x, stat_x, targets = dyn_x.to(device), stat_x.to(device), targets.to(device)
                
                outputs = model(dyn_x, stat_x)
                probs = torch.sigmoid(outputs)
                
                all_preds.extend(probs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        pred_labels = (all_preds >= 0.5).astype(int)
        
        if len(np.unique(all_targets)) > 1:
            auroc = roc_auc_score(all_targets, all_preds)
            bal_acc = balanced_accuracy_score(all_targets, pred_labels)
            f1 = f1_score(all_targets, pred_labels, average='macro', zero_division=0)
        else:
            auroc = np.nan
            bal_acc = balanced_accuracy_score(all_targets, pred_labels)
            f1 = f1_score(all_targets, pred_labels, average='macro', zero_division=0)

        score = auroc if not np.isnan(auroc) else bal_acc
        if score > best_score:
            best_score = score
            best_fold_model_path = fold_model_path

        fold_metrics.append({
            'subject': str(test_subject),
            'balanced_accuracy': float(bal_acc),
            'macro_f1': float(f1),
            'auroc': float(auroc) if not np.isnan(auroc) else None
        })

        print(f"[{test_subject}] -> Bal. Acc: {bal_acc:.4f} | Macro-F1: {f1:.4f} | AUROC: {auroc if np.isnan(auroc) else f'{auroc:.4f}'}")

        all_bal_acc.append(bal_acc)
        all_f1.append(f1)
        if not np.isnan(auroc):
            all_auroc.append(auroc)

    print("\n=======================================================")
    print("VÉGLEGES LOSO EREDMÉNYEK (NORMALIZÁLT ADATOKKAL)")
    print("=======================================================")
    print(f"Átlagos Balanced Accuracy: {np.mean(all_bal_acc):.4f} ± {np.std(all_bal_acc):.4f}")
    print(f"Átlagos Macro-F1 Score:    {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")
    print(f"Átlagos AUROC Score:       {np.mean(all_auroc):.4f} ± {np.std(all_auroc):.4f}")
    print("=======================================================")

    agg_bal_acc_mean = float(np.mean(all_bal_acc))
    agg_bal_acc_std = float(np.std(all_bal_acc))
    agg_f1_mean = float(np.mean(all_f1))
    agg_f1_std = float(np.std(all_f1))
    agg_auroc_mean = float(np.mean(all_auroc)) if all_auroc else None
    agg_auroc_std = float(np.std(all_auroc)) if all_auroc else None

    final_metrics = {
        'config': config,
        'fold_metrics': fold_metrics,
        'aggregates': {
            'balanced_accuracy_mean': agg_bal_acc_mean,
            'balanced_accuracy_std': agg_bal_acc_std,
            'macro_f1_mean': agg_f1_mean,
            'macro_f1_std': agg_f1_std,
            'auroc_mean': agg_auroc_mean,
            'auroc_std': agg_auroc_std
        }
    }

    metrics_path = os.path.join(run_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=4)

    if best_fold_model_path is not None:
        best_model_path = os.path.join(run_dir, 'best_model.pth')
        shutil.copy2(best_fold_model_path, best_model_path)
        print(f"Legjobb fold modell elmentve: {best_model_path}")

if __name__ == '__main__':
    main()
