"""Deep-learning uncertainty baseline — deep ensembles (Lakshminarayanan et al.,
2017) and MC-dropout, the comparison methods from Ovadia et al. (2019), the
paper this project's central question is grounded in but that the classical
baselines never actually implemented.

Runs on CPU by default. On the lab's AMD Instinct MI300X, install the
ROCm build of PyTorch (see https://pytorch.org/get-started/locally/, select
ROCm) instead of the default CUDA wheel from requirements.txt — once that's
installed, `torch.cuda.is_available()` and the `.cuda()` calls below work
unchanged, since PyTorch's ROCm backend is exposed through the same
torch.cuda API.
"""

import numpy as np
import torch
from torch import nn


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TabularMLP(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # logits


def train_single_net(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = 42,
    epochs: int = 20,
    batch_size: int = 2048,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> TabularMLP:
    device = device or get_device()
    torch.manual_seed(seed)

    model = TabularMLP(n_features=X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
    return model


def predict_proba(model: TabularMLP, X: np.ndarray, device: torch.device | None = None) -> np.ndarray:
    device = device or get_device()
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32).to(device))
        return torch.sigmoid(logits).cpu().numpy()


def train_deep_ensemble(
    X_train: np.ndarray, y_train: np.ndarray, n_members: int = 5, seed: int = 42, **train_kwargs
) -> list[TabularMLP]:
    """Independently-initialized networks trained on the same data — disagreement
    between members is the uncertainty signal (Lakshminarayanan et al., 2017)."""
    return [
        train_single_net(X_train, y_train, seed=seed + i, **train_kwargs)
        for i in range(n_members)
    ]


def ensemble_predict(models: list[TabularMLP], X: np.ndarray, device: torch.device | None = None):
    """Returns (mean_proba, per_member_proba). Ensemble variance across
    per_member_proba is a natural uncertainty score: high variance = the
    members disagree = the model is unsure."""
    per_member = np.stack([predict_proba(m, X, device) for m in models], axis=0)
    return per_member.mean(axis=0), per_member


def mc_dropout_predict(model: TabularMLP, X: np.ndarray, n_samples: int = 20, device: torch.device | None = None):
    """Keep dropout active at inference time and take repeated stochastic
    forward passes (Gal & Ghahramani, 2016) — a single-network alternative
    to a full deep ensemble."""
    device = device or get_device()
    model = model.to(device)
    model.train()  # keep dropout active
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        samples = np.stack(
            [torch.sigmoid(model(X_t)).cpu().numpy() for _ in range(n_samples)], axis=0
        )
    return samples.mean(axis=0), samples
