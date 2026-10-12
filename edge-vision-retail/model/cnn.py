"""
A small convolutional network and its training loop, as a stand-in for an edge vision model.

The data is scikit-learn's bundled handwritten-digits set (1,797 images, 8x8 pixels, ten
classes). It ships with scikit-learn, so everything here runs offline in seconds. The model is
tiny on purpose: the experiments in this folder are about what quantization and per-store
calibration do to a network, and those mechanisms are the same at any scale. Accuracy numbers
from it are not retail-detection numbers.

The network uses batch normalisation (BN) after each convolution, as production vision
backbones do. That matters twice:
  * at deployment BN is folded into the convolution weights, which gives each output channel
    a very different weight range - the reason per-channel quantization exists
  * BN's running statistics describe the input distribution, so re-estimating them on a new
    store's images is a cheap, label-free way to adapt one model to many stores
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x).flatten(1))


def load_data(seed: int = 0):
    d = load_digits()
    x = (d.images / 16.0).astype(np.float32)[:, None]          # [N, 1, 8, 8] in [0, 1]
    y = d.target.astype(np.int64)
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.3, random_state=seed, stratify=y)
    return torch.from_numpy(xtr), torch.from_numpy(ytr), torch.from_numpy(xte), torch.from_numpy(yte)


def train(seed: int = 0, epochs: int = 40) -> tuple[SmallCNN, tuple]:
    torch.manual_seed(seed)
    data = load_data(seed)
    xtr, ytr, _, _ = data
    model = SmallCNN()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(xtr))
        for i in range(0, len(xtr), 64):
            idx = perm[i:i + 64]
            loss = nn.functional.cross_entropy(model(xtr[idx]), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.eval(), data


@torch.no_grad()
def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    return float((model(x).argmax(1) == y).float().mean())
