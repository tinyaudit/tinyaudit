"""Shared torch fixtures for the uncertainty-estimator tests.

The three estimators (MC Dropout, deep ensemble, early-exit ensemble) all need
a tiny trained torch MLP. Building one in fixtures keeps each test file focused
on the estimator's contract rather than on model construction. torch is an
optional dependency, so every fixture is gated behind ``importorskip``; the
whole module is skipped cleanly when torch is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not installed")
nn = torch.nn


class DropoutMLP(nn.Module):
    """A 2-hidden-layer MLP with dropout, for MC Dropout and ensembles."""

    def __init__(self, n_features: int, n_classes: int = 2, p: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 16),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(16, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PlainMLP(nn.Module):
    """The same MLP without any dropout layer (MC Dropout must reject it)."""

    def __init__(self, n_features: int, n_classes: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 16),
            nn.ReLU(),
            nn.Linear(16, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EarlyExitMLP(nn.Module):
    """An MLP that advertises one intermediate early-exit head.

    ``forward`` returns only the final logits (so the ordinary predict path is
    unchanged) but also exposes a callable ``early_exit_logits`` that returns
    the branch logits, matching the contract the TorchModel adapter probes.
    """

    def __init__(self, n_features: int, n_classes: int = 2) -> None:
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(n_features, 16), nn.ReLU())
        self.exit_head = nn.Linear(16, n_classes)
        self.final_head = nn.Linear(16, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return self.final_head(h)

    def early_exit_logits(self, x: torch.Tensor) -> list[torch.Tensor]:
        h = self.backbone(x)
        return [self.exit_head(h)]


def _train(module: nn.Module, X: np.ndarray, y: np.ndarray, *, epochs: int = 80) -> None:
    """Tiny training loop so the modules are not random at test time."""
    torch.manual_seed(0)
    xt = torch.as_tensor(X.astype(np.float32))
    yt = torch.as_tensor(y.astype(np.int64))
    opt = torch.optim.Adam(module.parameters(), lr=0.05)
    loss_fn = nn.CrossEntropyLoss()
    module.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(module(xt), yt)
        loss.backward()
        opt.step()
    module.eval()


@pytest.fixture
def xy(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """A small, learnable binary task: two informative features."""
    n = 200
    y = rng.integers(0, 2, size=n)
    x0 = y + rng.normal(0, 0.4, size=n)
    x1 = y + rng.normal(0, 0.6, size=n)
    X = np.column_stack([x0, x1]).astype(np.float64)
    return X, y.astype(np.int64)


@pytest.fixture
def dropout_model(xy: tuple[np.ndarray, np.ndarray]):
    """A fitted TorchModel wrapping a dropout MLP."""
    from tinyaudit.models.torch import TorchModel

    X, y = xy
    module = DropoutMLP(n_features=X.shape[1])
    _train(module, X, y)
    return TorchModel(module)


@pytest.fixture
def plain_model(xy: tuple[np.ndarray, np.ndarray]):
    """A fitted TorchModel with no dropout layer."""
    from tinyaudit.models.torch import TorchModel

    X, y = xy
    module = PlainMLP(n_features=X.shape[1])
    _train(module, X, y)
    return TorchModel(module)


@pytest.fixture
def early_exit_model(xy: tuple[np.ndarray, np.ndarray]):
    """A fitted TorchModel exposing one early-exit head."""
    from tinyaudit.models.torch import TorchModel

    X, y = xy
    module = EarlyExitMLP(n_features=X.shape[1])
    _train(module, X, y)
    return TorchModel(module)
