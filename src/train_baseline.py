import sys
import os

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

def main():
    print("--- Adatok betöltése ---")
    data_path = os.path.join(project_root, 'data', 'processed', 'processed_dataset_calibrated.npz')
    data = np.load(data_path, allow_pickle=True)
    
    X_dynamic = data['X_dynamic']  
    X_static = data['X_static']
    y_raw = data['y']
    subjects = data['subjects']

    # --- CÍMKE TRANSZFORMÁCIÓ ---
    threshold = np.median(y_raw)
    y_binary = (y_raw >= threshold).astype(int)
    
    unique_subjects = np.unique(subjects)
    print(f"Adathalmaz: {len(X_dynamic)} ablak, {len(unique_subjects)} alany.")
    print(f"Címke küszöb: {threshold} -> 0-s osztály: {sum(y_binary==0)}, 1-es osztály: {sum(y_binary==1)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Használt eszköz: {device}")

    all_bal_acc, all_f1, all_auroc = [], [], []

    # ---------------------------------------------------------
    # LOSO KERESZTVVALIDÁCIÓS CIKLUS
    # ---------------------------------------------------------
    for test_subject in unique_subjects:
        train_mask = subjects != test_subject
        test_mask = subjects == test_subject

        X_dyn_train = X_dynamic[train_mask].copy()
        X_dyn_test = X_dynamic[test_mask].copy()

        mean = X_dyn_train.mean(axis=(0, 1), keepdims=True)
        std = X_dyn_train.std(axis=(0, 1), keepdims=True) + 1e-8 # 1e-8 a 0-val való osztás ellen

        X_dyn_train = (X_dyn_train - mean) / std
        X_dyn_test = (X_dyn_test - mean) / std

        # Transzponálás: (Batch, Idő, Csatorna) -> (Batch, Csatorna, Idő) a Conv1d-hez
        X_dyn_train = np.transpose(X_dyn_train, (0, 2, 1))
        X_dyn_test = np.transpose(X_dyn_test, (0, 2, 1))

        train_dataset = CLAREDataset(X_dyn_train, X_static[train_mask], y_binary[train_mask])
        test_dataset = CLAREDataset(X_dyn_test, X_static[test_mask], y_binary[test_mask])

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Modell inicializálása
        model = BaselineModel(num_dynamic_features=4).to(device)
        
        # Súlyozott veszteségfüggvény az esetleges imbalansz ellen
        pos_weight_val = sum(y_binary[train_mask] == 0) / sum(y_binary[train_mask] == 1)
        pos_weight_tensor = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)

        # --- TANÍTÁS ---
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

        # --- TESZTELÉS (Zero-shot) ---
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
        
        # Döntési küszöb
        pred_labels = (all_preds >= 0.5).astype(int)
        
        # Metrikák
        if len(np.unique(all_targets)) > 1:
            auroc = roc_auc_score(all_targets, all_preds)
            bal_acc = balanced_accuracy_score(all_targets, pred_labels)
            f1 = f1_score(all_targets, pred_labels, average='macro', zero_division=0)
        else:
            auroc = np.nan
            bal_acc = balanced_accuracy_score(all_targets, pred_labels)
            f1 = f1_score(all_targets, pred_labels, average='macro', zero_division=0)

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

if __name__ == '__main__':
    main()