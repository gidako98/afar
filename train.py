# train.py
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

N = 16

class CSVDataset(Dataset):
    def __init__(self, path, indices=None):
        df = pd.read_csv(path, header=None)
        data = df.values.astype(np.float32)
        if indices is not None:
            data = data[indices]
        X = data[:, :48]      # amp16 | sin16 | cos16
        y = data[:, 48:64]    # mask16 (1=fail)
        amp  = X[:, 0:16]
        sinp = X[:, 16:32]
        cosp = X[:, 32:48]
        self.X = torch.tensor(np.stack([amp, sinp, cosp], axis=-1), dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]

class FaultNet(nn.Module):
    def __init__(self, f_in=3, hidden=64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(f_in, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Conv1d(hidden, 1, 1)
    
    def forward(self, x):
        x = x.transpose(1, 2)  # (B,16,3) -> (B,3,16)
        h = self.backbone(x)
        return self.head(h).squeeze(1)  # (B,16)

def metrics_from_logits(logits, y_true, thr):
    prob = torch.sigmoid(logits)
    pred = (prob > thr).float()
    tp = ((pred == 1) & (y_true == 1)).sum().item()
    fp = ((pred == 1) & (y_true == 0)).sum().item()
    fn = ((pred == 0) & (y_true == 1)).sum().item()
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    exact = (pred.eq(y_true).all(dim=1)).float().mean().item()
    return precision, recall, f1, exact

def pick_best_threshold(model, loader, device):
    model.eval()
    all_logits, all_y = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            all_logits.append(logits.cpu())
            all_y.append(y.cpu())
    logits = torch.cat(all_logits, dim=0)
    y_true = torch.cat(all_y, dim=0)
    best_thr, best_f1 = 0.5, -1
    for thr in np.linspace(0.05, 0.95, 19):
        _, _, f1, _ = metrics_from_logits(logits, y_true, thr=float(thr))
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr, best_f1

def main():
    np.random.seed(42)
    torch.manual_seed(42)
    path = "dataset_correct.csv"  # ← ИСПОЛЬЗУЕМ КОРРЕКТНЫЙ ДАТАСЕТ
    df = pd.read_csv(path, header=None)
    n = len(df)
    idx = np.random.permutation(n)
    n_train = int(0.80 * n)
    n_val = int(0.10 * n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train+n_val]
    test_idx = idx[n_train+n_val:]
    
    train_ds = CSVDataset(path, train_idx)
    val_ds = CSVDataset(path, val_idx)
    test_ds = CSVDataset(path, test_idx)
    
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FaultNet(f_in=3, hidden=64).to(device)
    
    # Взвешенная функция потерь для борьбы с дисбалансом
    y_train = train_ds.y.numpy()
    pos = y_train.sum()
    neg = y_train.size - pos
    pos_weight = torch.tensor([neg / (pos + 1e-9)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    
    best_val_f1, best_state, best_thr = -1, None, 0.5
    
    for epoch in range(1, 101):  # 100 эпох для сходимости
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * X.size(0)
        
        # Валидация и подбор порога
        thr, _ = pick_best_threshold(model, val_loader, device)
        model.eval()
        with torch.no_grad():
            all_logits, all_y = [], []
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                all_logits.append(model(X).cpu())
                all_y.append(y.cpu())
            v_logits = torch.cat(all_logits, dim=0)
            v_y = torch.cat(all_y, dim=0)
            p, r, f1, ex = metrics_from_logits(v_logits, v_y, thr)
        
        print(f"Epoch {epoch:02d} | train_loss={total_loss/len(train_ds):.4f} | "
              f"val_thr={thr:.2f} | val_F1={f1:.3f} | P={p:.3f} R={r:.3f} | exact={ex:.3f}")
        
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_thr = thr
    
    # Финальная оценка на тесте
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        all_logits, all_y = [], []
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            all_logits.append(model(X).cpu())
            all_y.append(y.cpu())
        t_logits = torch.cat(all_logits, dim=0)
        t_y = torch.cat(all_y, dim=0)
        p, r, f1, ex = metrics_from_logits(t_logits, t_y, best_thr)
    
    print("\nТЕСТОВЫЕ РЕЗУЛЬТАТЫ (корректный датасет):")
    print(f"threshold={best_thr:.2f} | F1={f1:.3f} | Precision={p:.3f} | "
          f"Recall={r:.3f} | ExactMatch={ex:.3f}")
    
    # Сохранение
    torch.save(model.state_dict(), "faultnet.pt")
    with open("threshold.json", "w", encoding="utf-8") as f:
        json.dump({"threshold": best_thr}, f, ensure_ascii=False, indent=2)
    print("\n✅ Модель сохранена: faultnet.pt, threshold.json")

if __name__ == "__main__":
    main()