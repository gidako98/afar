import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

N = 16

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

def load_one_from_csv(path, row=0):
    df = pd.read_csv(path, header=None)
    data = df.values.astype(np.float32)[row]
    X = data[:48]  # ← ТОЛЬКО ПРИЗНАКИ, МЕТКИ ИГНОРИРУЮТСЯ
    amp = X[0:16]
    sinp = X[16:32]
    cosp = X[32:48]
    x = np.stack([amp, sinp, cosp], axis=-1)  # (16,3)
    return torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # (1,16,3)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FaultNet(f_in=3, hidden=64).to(device)
    model.load_state_dict(torch.load("faultnet.pt", map_location=device))  # ← без пробела!
    model.eval()
    
    with open("threshold.json", "r", encoding="utf-8") as f:  # ← без пробела!
        thr = float(json.load(f)["threshold"])  # ← без пробела!
    
    # Пример: строка 100 — отказ элементов 5 и 6
    x = load_one_from_csv("dataset_correct.csv", row=100).to(device)
    
    with torch.no_grad():
        logits = model(x)  # (1,16)
        prob = torch.sigmoid(logits).cpu().numpy().reshape(-1)
    
    failed = [i+1 for i, p in enumerate(prob) if p > thr]  # 1-based индексы
    
    # Загрузка реальных меток для демонстрации точности
    df = pd.read_csv("dataset_correct.csv", header=None)
    y_true = df.values[100, 48:64].astype(int)
    true_failed = [i+1 for i, v in enumerate(y_true) if v == 1]
    
    print(f"Порог бинаризации: {thr:.2f}")
    print("Вероятности отказа по элементам (1..16):")
    for i, p in enumerate(prob, 1):
        marker = " ← ОТКАЗ" if i in failed else ""
        print(f"  Элемент {i:2d}: {p:.3f}{marker}")
    print(f"\nРЕАЛЬНЫЕ отказы:    {true_failed}")
    print(f"ПРЕДСКАЗАННЫЕ:      {failed}")
    print(f"Совпадение: {'✓ ТОЧНО' if set(failed) == set(true_failed) else '✗ Частично'}")

if __name__ == "__main__":
    main()
