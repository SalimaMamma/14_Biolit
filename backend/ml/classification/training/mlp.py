"""
mlp.py — MLP par niveau taxonomique de la méthode v2 (référence de non-régression uniquement)
==============================================================================================
Le modèle livré n'utilise pas de MLP ; cette architecture sert uniquement à ré-entraîner
la v2 à l'identique (baseline_v2.py), référence de non-régression de train.py.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from ..config import SEED


class LevelMLP(nn.Module):
    """Linear(input, hidden) → BatchNorm → ReLU → Dropout → Linear(hidden, n_classes)."""

    def __init__(self, input_dim: int, n_classes: int, dropout: float = 0.3):
        super().__init__()
        hidden = max(256, n_classes * 8)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, x):
        return self.net(x)


def train_mlp(features: np.ndarray, labels: np.ndarray, n_classes: int, epochs: int = 40,
              seed: int = SEED) -> LevelMLP:
    """AdamW + cosine, échantillonnage équilibré par classe (comme la v2 d'origine)."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    counts = np.bincount(labels, minlength=n_classes)
    weights = 1.0 / np.maximum(counts[labels], 1)
    sampler = WeightedRandomSampler(torch.tensor(weights, dtype=torch.double),
                                    num_samples=len(labels), replacement=True, generator=gen)
    ds = TensorDataset(torch.tensor(features, dtype=torch.float32),
                       torch.tensor(labels, dtype=torch.long))
    loader = DataLoader(ds, batch_size=64, sampler=sampler, drop_last=len(ds) > 64)
    mlp = LevelMLP(features.shape[1], n_classes)
    opt = AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = CosineAnnealingLR(opt, T_max=epochs)
    mlp.train()
    for _ in range(epochs):
        for xb, yb in loader:
            loss = F.cross_entropy(mlp(xb), yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
            opt.step()
        sch.step()
    return mlp.eval()


def mlp_logits(mlp: LevelMLP, features: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return mlp(torch.tensor(features, dtype=torch.float32)).numpy()
