"""Loss functions represented in the historical experiment notebooks."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LossKind


class WeightedHuberLoss(nn.Module):
    """Huber loss with extra attention to both low and high GPP values."""

    def __init__(
        self,
        delta: float = 1.0,
        alpha_hi: float = 1.0,
        alpha_lo: float = 1.0,
        p: float = 2.0,
        y_max: float = 30.0,
    ) -> None:
        super().__init__()
        if y_max <= 0:
            raise ValueError("y_max must be positive")
        self.delta = delta
        self.alpha_hi = alpha_hi
        self.alpha_lo = alpha_lo
        self.p = p
        self.y_max = y_max

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        per_element = F.huber_loss(
            y_pred, y_true, delta=self.delta, reduction="none"
        )
        y_norm = torch.clamp(y_true / self.y_max, min=0.0, max=1.0)
        weights = (
            1.0
            + self.alpha_hi * y_norm.pow(self.p)
            + self.alpha_lo * (1.0 - y_norm).pow(self.p)
        )
        return torch.mean(weights * per_element)


def build_loss(kind: LossKind | str, **options) -> nn.Module:
    kind = LossKind(kind)
    if kind is LossKind.MSE:
        return nn.MSELoss(**options)
    if kind is LossKind.MAE:
        return nn.L1Loss(**options)
    if kind is LossKind.HUBER:
        return nn.HuberLoss(**options)
    return WeightedHuberLoss(**options)
