from typing import Literal

import torch


ScheduleName = Literal["linear", "cosine", "quadratic", "sigmoid"]


def linear_beta_schedule(
    timesteps: int,
    beta_start: float = 4e-5,
    beta_end: float = 5e-3,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    线性 beta 调度

    对应论文里最常用、也是你当前 3D-RadioDiff 复现应优先使用的版本。
    论文中给出:
        beta_1 = 4e-5
        beta_T = 5e-3
    且采用 linear variance schedule。

    Args:
        timesteps: 扩散步数 T
        beta_start: 起始 beta
        beta_end: 结束 beta
        dtype: 返回张量的数据类型

    Returns:
        betas: shape [timesteps]
    """
    if timesteps <= 0:
        raise ValueError(f"timesteps 必须大于 0，当前为 {timesteps}")
    if not (0.0 < beta_start < 1.0):
        raise ValueError(f"beta_start 必须在 (0,1) 内，当前为 {beta_start}")
    if not (0.0 < beta_end < 1.0):
        raise ValueError(f"beta_end 必须在 (0,1) 内，当前为 {beta_end}")
    if beta_start >= beta_end:
        raise ValueError(
            f"beta_start 应小于 beta_end，当前 beta_start={beta_start}, beta_end={beta_end}"
        )

    return torch.linspace(beta_start, beta_end, timesteps, dtype=dtype)


def cosine_beta_schedule(
    timesteps: int,
    s: float = 0.008,
    max_beta: float = 0.999,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Cosine beta schedule
    参考 Improved DDPM 常用做法。

    Args:
        timesteps: 扩散步数
        s: 小偏移项
        max_beta: beta 上限，防止数值问题
        dtype: 返回类型

    Returns:
        betas: shape [timesteps]
    """
    if timesteps <= 0:
        raise ValueError(f"timesteps 必须大于 0，当前为 {timesteps}")
    if not (0.0 < max_beta < 1.0):
        raise ValueError(f"max_beta 必须在 (0,1) 内，当前为 {max_beta}")

    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=dtype)

    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clamp(betas, min=1e-8, max=max_beta)
    return betas


def quadratic_beta_schedule(
    timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Quadratic beta schedule

    通过对 sqrt(beta) 做线性插值，再平方，得到更平滑的增长。

    Args:
        timesteps: 扩散步数
        beta_start: 起始 beta
        beta_end: 结束 beta
        dtype: 返回类型
    """
    if timesteps <= 0:
        raise ValueError(f"timesteps 必须大于 0，当前为 {timesteps}")
    if not (0.0 < beta_start < 1.0):
        raise ValueError(f"beta_start 必须在 (0,1) 内，当前为 {beta_start}")
    if not (0.0 < beta_end < 1.0):
        raise ValueError(f"beta_end 必须在 (0,1) 内，当前为 {beta_end}")
    if beta_start >= beta_end:
        raise ValueError(
            f"beta_start 应小于 beta_end，当前 beta_start={beta_start}, beta_end={beta_end}"
        )

    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps, dtype=dtype) ** 2


def sigmoid_beta_schedule(
    timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Sigmoid beta schedule

    在中间区域变化更快，两端更平缓。

    Args:
        timesteps: 扩散步数
        beta_start: 起始 beta
        beta_end: 结束 beta
        dtype: 返回类型
    """
    if timesteps <= 0:
        raise ValueError(f"timesteps 必须大于 0，当前为 {timesteps}")
    if not (0.0 < beta_start < 1.0):
        raise ValueError(f"beta_start 必须在 (0,1) 内，当前为 {beta_start}")
    if not (0.0 < beta_end < 1.0):
        raise ValueError(f"beta_end 必须在 (0,1) 内，当前为 {beta_end}")
    if beta_start >= beta_end:
        raise ValueError(
            f"beta_start 应小于 beta_end，当前 beta_start={beta_start}, beta_end={beta_end}"
        )

    x = torch.linspace(-6, 6, timesteps, dtype=dtype)
    betas = torch.sigmoid(x) * (beta_end - beta_start) + beta_start
    return betas


def get_beta_schedule(
    schedule_name: ScheduleName,
    timesteps: int,
    beta_start: float = 4e-5,
    beta_end: float = 5e-3,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    统一获取 beta schedule 的接口

    Args:
        schedule_name:
            - "linear"
            - "cosine"
            - "quadratic"
            - "sigmoid"
        timesteps: 扩散步数
        beta_start: 对 linear / quadratic / sigmoid 有效
        beta_end: 对 linear / quadratic / sigmoid 有效
        dtype: 返回类型

    Returns:
        betas: shape [timesteps]
    """
    schedule_name = schedule_name.lower()

    if schedule_name == "linear":
        return linear_beta_schedule(
            timesteps=timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            dtype=dtype,
        )

    if schedule_name == "cosine":
        return cosine_beta_schedule(
            timesteps=timesteps,
            dtype=dtype,
        )

    if schedule_name == "quadratic":
        return quadratic_beta_schedule(
            timesteps=timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            dtype=dtype,
        )

    if schedule_name == "sigmoid":
        return sigmoid_beta_schedule(
            timesteps=timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            dtype=dtype,
        )

    raise ValueError(
        f"不支持的 schedule_name: {schedule_name}，"
        f"可选: linear / cosine / quadratic / sigmoid"
    )


if __name__ == "__main__":
    # 简单测试
    T = 200

    betas_linear = get_beta_schedule(
        schedule_name="linear",
        timesteps=T,
        beta_start=4e-5,
        beta_end=5e-3,
    )
    print("linear betas shape:", betas_linear.shape)
    print("linear first beta:", betas_linear[0].item())
    print("linear last beta :", betas_linear[-1].item())

    betas_cosine = get_beta_schedule(
        schedule_name="cosine",
        timesteps=T,
    )
    print("cosine betas shape:", betas_cosine.shape)
    print("cosine first beta:", betas_cosine[0].item())
    print("cosine last beta :", betas_cosine[-1].item())