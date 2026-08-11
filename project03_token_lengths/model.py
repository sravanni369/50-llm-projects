"""Two small PyTorch heads over the same 20-number token-length histogram.

Deliberately small. The input is a probability vector of length 20 and there are a few
hundred training samples, so anything wider would memorise the corpus rather than learn
the shape of the distribution. If a 20-hidden-unit MLP cannot do it, the feature does
not carry the signal and a bigger model would only hide that.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from fingerprint import MAX_LEN


class LanguageClassifier(nn.Module):
    """Histogram -> which language."""

    def __init__(self, n_classes: int, n_features: int = MAX_LEN, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CostRegressor(nn.Module):
    """Histogram -> characters per token, the number that sets the API bill.

    Softplus on the output because chars-per-token is strictly positive; letting the
    model predict negatives wastes capacity learning a constraint we already know.
    """

    def __init__(self, n_features: int = MAX_LEN, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.softplus(self.net(x)).squeeze(-1)


def train(model: nn.Module, X: torch.Tensor, y: torch.Tensor, *, loss_fn,
          epochs: int = 400, lr: float = 0.01, seed: int = 0) -> list[float]:
    """Full-batch training. Returns the loss curve so it can be plotted and inspected."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses
