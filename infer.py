# infer.py
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
        x = x.transpose(1, 2)            # (B,3,16)
        h = self.backbone(x)
        return self.head(h).squeeze(1)   # (B,16)

def load_one_from_csv(path, row=0):
    df = pd.read_csv(path, header=None)
    data = df.values.astype(np.float32)[row]

    X = data[:48]
    amp  = X[0:16]
    sinp = X[16:32]
    cosp = X[32:48]

    x = np.stack([amp, sinp, cosp], axis=-1)  # (16,3)
    return torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # (1,16,3)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = FaultNet(f_in=3, hidden=64).to(device)
    model.load_state_dict(torch.load("faultnet.pt", map_location=device))
    model.eval()

    with open("threshold.json", "r", encoding="utf-8") as f:
        thr = float(json.load(f)["threshold"])

    # пример: берём первую строку из dataset.csv
    x = load_one_from_csv("dataset.csv", row=0).to(device)

    with torch.no_grad():
        logits = model(x)  # (1,16)
        prob = torch.sigmoid(logits).cpu().numpy().reshape(-1)

    failed = [i+1 for i, p in enumerate(prob) if p > thr]

    print(f"threshold = {thr:.2f}")
    print("Probabilities per PPM (1..16):")
    print(np.array2string(prob, precision=3, suppress_small=True))
    print("FAILED PPM indices:", failed)

if __name__ == "__main__":
    main()
