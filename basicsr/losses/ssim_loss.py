import torch
from torch import nn
from kornia.metrics import ssim

from basicsr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class SSIMLoss(nn.Module):
    """Loss estrutural: 1 - SSIM."""

    def __init__(
        self,
        loss_weight=1.0,
        window_size=11,
        max_val=1.0,
        reduction="mean",
        eps=1e-12,
    ):
        super().__init__()

        if reduction not in ("mean", "sum", "none"):
            raise ValueError(
                f"Reduction inválida: {reduction}"
            )

        self.loss_weight = loss_weight
        self.window_size = window_size
        self.max_val = max_val
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target):
        if pred.shape != target.shape:
            raise ValueError(
                f"Dimensões diferentes: {pred.shape} e {target.shape}"
            )

        # FP32 melhora a estabilidade numérica quando AMP está ativo.
        score = ssim(
            pred.float(),
            target.float(),
            window_size=self.window_size,
            max_val=self.max_val,
            eps=self.eps,
            padding="same",
        )

        loss = 1.0 - score

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return self.loss_weight * loss
