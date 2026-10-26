"""Training loop for the deep learning models."""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class MinMax:
    """Min-max normalisation of the CCS target to [0, 1]."""

    def fit(self, y):
        self.lo, self.hi = float(np.min(y)), float(np.max(y))
        return self

    def transform(self, y):
        return (np.asarray(y) - self.lo) / (self.hi - self.lo)

    def inverse(self, y):
        return np.asarray(y) * (self.hi - self.lo) + self.lo


def loader(seq, charge, mods, y=None, batch_size=256, shuffle=False):
    tensors = [torch.as_tensor(a, dtype=torch.float32) for a in (seq, charge, mods)]
    if y is not None:
        tensors.append(torch.as_tensor(y, dtype=torch.float32))
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def predict(model, data: DataLoader) -> np.ndarray:
    model.eval()
    return np.concatenate([model(*batch[:3]).numpy() for batch in data])


def fit(model, train: DataLoader, val: DataLoader, *, kind: str, epochs: int, lr: float = 1e-3,
        patience: int = 5, warmup_epochs: int = 2, log=print) -> dict:
    """LSTM: Adam, lr 1e-3, MSE. CNN: AdamW (wd 1e-4), linear warmup, ReduceLROnPlateau, early stopping."""
    loss_fn = nn.MSELoss()
    if kind == "cnn":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        plateau = None

    history = {"train_loss": [], "val_loss": []}
    best, best_loss, bad = None, float("inf"), 0
    for epoch in range(epochs):
        if kind == "cnn" and epoch < warmup_epochs:
            for g in opt.param_groups:
                g["lr"] = lr * (epoch + 1) / warmup_epochs
        model.train()
        total, n = 0.0, 0
        for seq, charge, mods, y in train:
            opt.zero_grad()
            loss = loss_fn(model(seq, charge, mods), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            n += len(y)
        model.eval()
        with torch.no_grad():
            val_loss = float(np.mean([loss_fn(model(s, c, m), y).item() for s, c, m, y in val]))
        history["train_loss"].append(total / n)
        history["val_loss"].append(val_loss)
        log(f"epoch {epoch + 1:3d}  train {total / n:.5f}  val {val_loss:.5f}")
        if plateau is not None and epoch >= warmup_epochs:
            plateau.step(val_loss)
        if val_loss < best_loss:
            best, best_loss, bad = copy.deepcopy(model.state_dict()), val_loss, 0
        else:
            bad += 1
            if kind == "cnn" and bad >= patience:
                log("early stopping")
                break
    model.load_state_dict(best)
    return history
