import math
import torch
import torch.nn.functional as F


def mse_metric(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    pred, target: [B, C, H, W]
    return: [B]
    """
    return torch.mean((pred - target) ** 2, dim=(1, 2, 3))


def rmse_metric(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    RMSE = sqrt(MSE)
    return: [B]
    """
    mse = mse_metric(pred, target)
    return torch.sqrt(mse + 1e-12)


def nmse_metric(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    NMSE = sum((pred-target)^2) / sum(target^2)
    return: [B]
    """
    num = torch.sum((pred - target) ** 2, dim=(1, 2, 3))
    den = torch.sum(target ** 2, dim=(1, 2, 3)) + eps
    return num / den


def psnr_metric(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 2.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    PSNR = 10 * log10( r^2 / MSE )

    你的 psi 是 [-1, 1]，所以动态范围 r = 2
    return: [B]
    """
    mse = mse_metric(pred, target)
    psnr = 10.0 * torch.log10((data_range ** 2) / (mse + eps))
    return psnr


def _gaussian_kernel(window_size: int, sigma: float, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g


def _create_gaussian_window(window_size: int, sigma: float, channels: int, device, dtype):
    g1d = _gaussian_kernel(window_size, sigma, device=device, dtype=dtype)
    g2d = torch.outer(g1d, g1d)
    g2d = g2d / g2d.sum()
    window = g2d.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    window = window.expand(channels, 1, window_size, window_size).contiguous()
    return window


def ssim_metric(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 2.0,
    window_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    SSIM for [B, C, H, W]
    return: [B]
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred and target shape mismatch: {pred.shape} vs {target.shape}")

    b, c, h, w = pred.shape
    device = pred.device
    dtype = pred.dtype

    window = _create_gaussian_window(window_size, sigma, c, device, dtype)

    padding = window_size // 2

    mu_x = F.conv2d(pred, window, padding=padding, groups=c)
    mu_y = F.conv2d(target, window, padding=padding, groups=c)

    mu_x2 = mu_x ** 2
    mu_y2 = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(pred * pred, window, padding=padding, groups=c) - mu_x2
    sigma_y2 = F.conv2d(target * target, window, padding=padding, groups=c) - mu_y2
    sigma_xy = F.conv2d(pred * target, window, padding=padding, groups=c) - mu_xy

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    c3 = c2 / 2.0

    # 亮度项
    l_xy = (2 * mu_xy + c1) / (mu_x2 + mu_y2 + c1 + eps)

    # 对比度项
    sigma_x = torch.sqrt(torch.clamp(sigma_x2, min=0.0) + eps)
    sigma_y = torch.sqrt(torch.clamp(sigma_y2, min=0.0) + eps)
    c_xy = (2 * sigma_x * sigma_y + c2) / (sigma_x2 + sigma_y2 + c2 + eps)

    # 结构项
    s_xy = (sigma_xy + c3) / (sigma_x * sigma_y + c3 + eps)

    ssim_map = l_xy * c_xy * s_xy

    # 对每个样本求平均
    return ssim_map.mean(dim=(1, 2, 3))