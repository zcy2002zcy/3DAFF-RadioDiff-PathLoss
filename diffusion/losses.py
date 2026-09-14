from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    标准 MSE loss
    """
    return F.mse_loss(pred, target, reduction=reduction)


def l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    标准 L1 loss
    """
    return F.l1_loss(pred, target, reduction=reduction)


def smooth_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
    beta: float = 1.0,
) -> torch.Tensor:
    """
    Smooth L1 / Huber loss
    """
    return F.smooth_l1_loss(pred, target, reduction=reduction, beta=beta)


def nmse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Normalized MSE loss

    支持输入:
        pred, target: [B, C, H, W]

    计算:
        NMSE_i = ||pred_i - target_i||^2 / (||target_i||^2 + eps)

    Args:
        eps: 防止分母为 0
        reduction:
            - "mean"
            - "sum"
            - "none"
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"pred 和 target 的 shape 不一致: pred={pred.shape}, target={target.shape}"
        )

    diff2 = (pred - target) ** 2
    target2 = target ** 2

    num = diff2.flatten(start_dim=1).sum(dim=1)              # [B]
    den = target2.flatten(start_dim=1).sum(dim=1) + eps      # [B]

    loss = num / den

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss

    raise ValueError(f"不支持的 reduction: {reduction}")


class DiffusionLoss(nn.Module):
    """
    Diffusion 训练损失统一封装

    当前支持：
        - mse
        - l1
        - smooth_l1
        - nmse

    用法：
        criterion = DiffusionLoss(loss_name="mse")
        loss = criterion(pred_noise, noise)

    如果你后面要扩展成多损失加权，也可以在这里继续加。
    """

    def __init__(
        self,
        loss_name: str = "mse",
        reduction: str = "mean",
        smooth_l1_beta: float = 1.0,
        nmse_eps: float = 1e-8,
    ) -> None:
        super().__init__()

        self.loss_name = loss_name.lower()
        self.reduction = reduction
        self.smooth_l1_beta = smooth_l1_beta
        self.nmse_eps = nmse_eps

        supported = ["mse", "l1", "smooth_l1", "nmse"]
        if self.loss_name not in supported:
            raise ValueError(
                f"不支持的 loss_name: {self.loss_name}，可选: {supported}"
            )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_name == "mse":
            return mse_loss(pred, target, reduction=self.reduction)

        if self.loss_name == "l1":
            return l1_loss(pred, target, reduction=self.reduction)

        if self.loss_name == "smooth_l1":
            return smooth_l1_loss(
                pred,
                target,
                reduction=self.reduction,
                beta=self.smooth_l1_beta,
            )

        if self.loss_name == "nmse":
            return nmse_loss(
                pred,
                target,
                eps=self.nmse_eps,
                reduction=self.reduction,
            )

        raise RuntimeError(f"未处理的 loss_name: {self.loss_name}")


def build_loss_from_config(cfg: Dict[str, Any]) -> DiffusionLoss:
    """
    从配置字典构建损失函数

    兼容类似：
        loss:
          name: "mse"

    或：
        loss:
          name: "smooth_l1"
          reduction: "mean"
          smooth_l1_beta: 1.0
    """
    loss_cfg = cfg.get("loss", cfg)

    return DiffusionLoss(
        loss_name=str(loss_cfg.get("name", "mse")),
        reduction=str(loss_cfg.get("reduction", "mean")),
        smooth_l1_beta=float(loss_cfg.get("smooth_l1_beta", 1.0)),
        nmse_eps=float(loss_cfg.get("nmse_eps", 1e-8)),
    )


if __name__ == "__main__":
    # ===== 简单测试 =====
    B, C, H, W = 2, 1, 256, 256
    pred = torch.randn(B, C, H, W)
    target = torch.randn(B, C, H, W)

    for name in ["mse", "l1", "smooth_l1", "nmse"]:
        criterion = DiffusionLoss(loss_name=name)
        loss = criterion(pred, target)
        print(f"{name} loss =", float(loss.item()))