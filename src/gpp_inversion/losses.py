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


class TailAwareLoss(nn.Module):
    """Capped target-bin weighting plus a high-target underprediction penalty."""

    def __init__(
        self,
        p50: float,
        p80: float,
        p95: float,
        weights: tuple[float, float, float, float] | list[float] = (1.0, 1.0, 1.5, 2.5),
        underprediction_weight: float = 0.25,
        base: str = "mse",
        delta: float = 1.0,
    ) -> None:
        super().__init__()
        if not p50 <= p80 <= p95:
            raise ValueError("Tail thresholds must satisfy p50 <= p80 <= p95")
        if len(weights) != 4 or min(weights) <= 0:
            raise ValueError("weights must contain four positive values")
        if underprediction_weight < 0:
            raise ValueError("underprediction_weight cannot be negative")
        if base not in {"mse", "huber"}:
            raise ValueError("base must be mse or huber")
        self.register_buffer("thresholds", torch.tensor([p50, p80, p95], dtype=torch.float32))
        self.register_buffer("bin_weights", torch.tensor(weights, dtype=torch.float32))
        self.underprediction_weight = float(underprediction_weight)
        self.base = base
        self.delta = float(delta)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if self.base == "huber":
            per_element = F.huber_loss(
                y_pred, y_true, delta=self.delta, reduction="none"
            )
        else:
            per_element = (y_pred - y_true).square()
        bin_index = torch.bucketize(y_true.detach(), self.thresholds)
        weighted = self.bin_weights[bin_index] * per_element
        high = y_true >= self.thresholds[1]
        under = torch.relu(y_true - y_pred).square()
        penalty = self.underprediction_weight * high.to(under.dtype) * under
        return torch.mean(weighted + penalty)


def build_loss(kind: LossKind | str, **options) -> nn.Module:
    kind = LossKind(kind)
    if kind is LossKind.MSE:
        return nn.MSELoss(**options)
    if kind is LossKind.MAE:
        return nn.L1Loss(**options)
    if kind is LossKind.HUBER:
        return nn.HuberLoss(**options)
    if kind is LossKind.TAIL_AWARE:
        return TailAwareLoss(**options)
    return WeightedHuberLoss(**options)
