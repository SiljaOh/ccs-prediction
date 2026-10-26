"""Deep learning models (PyTorch): LSTM and residual 1D-CNN with charge / modification branches."""
from __future__ import annotations

import torch
from torch import nn

N_AA, N_CHARGE, N_MODS = 20, 4, 3


class AuxBranches(nn.Module):
    """Small dense branches for the charge state and the modification flags."""

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.charge = nn.Sequential(nn.Linear(N_CHARGE, hidden), nn.ReLU())
        self.mods = nn.Sequential(nn.Linear(N_MODS, hidden), nn.ReLU())
        self.out_dim = 2 * hidden

    def forward(self, charge, mods):
        return torch.cat([self.charge(charge), self.mods(mods)], dim=1)


class LSTMRegressor(nn.Module):
    """LSTM (64 hidden units) over the one-hot sequence; final hidden state + aux features -> CCS."""

    def __init__(self, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(N_AA, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.aux = AuxBranches()
        self.head = nn.Sequential(nn.Linear(hidden + self.aux.out_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, seq, charge, mods):
        lengths = seq.sum(dim=(1, 2)).clamp(min=1).long().cpu()  # one-hot rows that are not padding
        packed = nn.utils.rnn.pack_padded_sequence(seq, lengths, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        z = torch.cat([self.dropout(h[-1]), self.aux(charge, mods)], dim=1)
        return self.head(z).squeeze(1)


class ResBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.2):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1), nn.BatchNorm1d(channels), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 3, padding=1), nn.BatchNorm1d(channels),
        )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x + self.body(x))


class CNNRegressor(nn.Module):
    """Conv(64, k=5) + BN + ReLU, two residual blocks (k=3), adaptive average pooling, dense head."""

    def __init__(self, channels: int = 64):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(N_AA, channels, 5, padding=2), nn.BatchNorm1d(channels), nn.ReLU())
        self.blocks = nn.Sequential(ResBlock(channels), ResBlock(channels))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.aux = AuxBranches()
        self.head = nn.Sequential(nn.Linear(channels + self.aux.out_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, seq, charge, mods):
        x = self.blocks(self.stem(seq.transpose(1, 2)))  # (batch, channels, length)
        z = torch.cat([self.pool(x).squeeze(-1), self.aux(charge, mods)], dim=1)
        return self.head(z).squeeze(1)
