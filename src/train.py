import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from model import GatedFusionModel


class CLAREDataset(Dataset):
    def __init__(self, dynamic_data, static_data, labels):
        self.dynamic = torch.FloatTensor(dynamic_data)
        self.dynamic = self.dynamic.permute(0, 2, 1) 
        
        self.static = torch.LongTensor(static_data)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.dynamic[idx], self.static[idx], self.labels[idx]

def main():
    print("--- Adatok betöltése ---")
    data = np.load('processed_dataset.npz', allow_pickle=True)
    X_dynamic = data['X_dynamic']
    X_static = data['X_static']
    y_raw = data['y']
    subjects = data['subjects']

    threshold = np.median(y_raw)
    y_binary = (y_raw >= threshold).astype(int)
    
    print(f"Eredeti címkék eloszlása (min/max/medián): {y_raw.min()} / {y_raw.max()} / {threshold}")
    print(f"Bináris eloszlás: {sum(y_binary==0)} Alacsony (0), {sum(y_binary==1)} Magas (1)")

    unique_subjects = np.unique(subjects)
    print(f"\nÖsszesen {len(unique_subjects)} résztvevő a LOSO validációhoz.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Használt eszköz: {device}")

    all_bal_acc, all_f1, all_auroc = [], [], []

    for test_subject in unique_subjects:
        print(f"\n[ LOSO Iteráció: Tesztelés a {test_subject} résztvevőn ]")

        train_mask = subjects != test_subject
        test_mask = subjects == test_subject

        train_dataset = CLAREDataset(X_dynamic[train_mask], X_static[train_mask], y_binary[train_mask])
        test_dataset = CLAREDataset(X_dynamic[test_mask], X_static[test_mask], y_binary[test_mask])

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        model = GatedFusionModel(num_dynamic_features=4, num_static_levels=4).to(device)
        
        criterion = nn.BCEWithLogitsLoss() 
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        epochs = 15 
        model.train()
        for epoch in range(epochs):
            for dyn_x, stat_x, targets in train_loader:
                dyn_x, stat_x, targets = dyn_x.to(device), stat_x.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(dyn_x, stat_x)
                loss = criterion(outputs, targets)
                loss.backward() 
                optimizer.step()

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
        else:
            auroc = np.nan

        bal_acc = balanced_accuracy_score(all_targets, pred_labels)
        f1 = f1_score(all_targets, pred_labels, average='macro')

        print(f" -> Eredmények: Bal. Acc: {bal_acc:.4f} | Macro-F1: {f1:.4f} | AUROC: {auroc:.4f}")

        all_bal_acc.append(bal_acc)
        all_f1.append(f1)
        if not np.isnan(auroc):
            all_auroc.append(auroc)

    print("\n=======================================================")
    print("VÉGLEGES LOSO EREDMÉNYEK (Zero-shot generalizáció)")
    print("=======================================================")
    print(f"Átlagos Balanced Accuracy: {np.mean(all_bal_acc):.4f} ± {np.std(all_bal_acc):.4f}")
    print(f"Átlagos Macro-F1 Score:    {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")
    print(f"Átlagos AUROC Score:       {np.mean(all_auroc):.4f} ± {np.std(all_auroc):.4f}")
    print("=======================================================")

if __name__ == '__main__':
    main()