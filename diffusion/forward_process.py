from typing import Dict, Optional, Tuple, Union

import torch


class DiffusionForwardProcess:
    """
    DDPM 前向扩散过程工具类

    负责：
    - 根据 betas 构造 alpha / alpha_bar 等系数
    - 随机采样时间步 t
    - 将真实样本 x0 加噪得到 xt
    - 提供训练阶段需要的各种辅助函数

    记号说明：
        x0  : 干净样本（你这里对应 psi）
        xt  : t 时刻带噪样本（你这里对应 psi_t）
        eps : 加入的高斯噪声
    """

    def __init__(
        self,
        betas: torch.Tensor,
        device: Union[torch.device, str] = "cpu",
    ) -> None:
        """
        Args:
            betas: shape [T]
            device: 运行设备
        """
        if betas.ndim != 1:
            raise ValueError(f"betas 必须是一维张量，当前 shape={betas.shape}")

        if torch.any(betas <= 0) or torch.any(betas >= 1):
            raise ValueError("betas 中每个元素都必须在 (0,1) 范围内")

        self.device = torch.device(device)

        # ---- 基本量 ----
        self.betas = betas.to(self.device)                                  # [T]
        self.timesteps = betas.shape[0]

        self.alphas = 1.0 - self.betas                                      # [T]
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)             # [T]

        # alpha_bar_{t-1}
        self.alphas_cumprod_prev = torch.cat(
            [
                torch.tensor([1.0], device=self.device, dtype=self.betas.dtype),
                self.alphas_cumprod[:-1],
            ],
            dim=0,
        )

        # ---- 常用平方根项 ----
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        self.log_one_minus_alphas_cumprod = torch.log(
            torch.clamp(1.0 - self.alphas_cumprod, min=1e-20)
        )

        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(
            torch.clamp(1.0 / self.alphas_cumprod - 1.0, min=0.0)
        )

        # ---- posterior q(x_{t-1} | x_t, x_0) 相关项 ----
        self.posterior_variance = (
            self.betas
            * (1.0 - self.alphas_cumprod_prev)
            / torch.clamp(1.0 - self.alphas_cumprod, min=1e-20)
        )

        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )

        self.posterior_mean_coef1 = (
            self.betas
            * torch.sqrt(self.alphas_cumprod_prev)
            / torch.clamp(1.0 - self.alphas_cumprod, min=1e-20)
        )

        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / torch.clamp(1.0 - self.alphas_cumprod, min=1e-20)
        )

    def to(self, device: Union[torch.device, str]) -> "DiffusionForwardProcess":
        """
        将内部缓存张量搬到新设备
        """
        new_device = torch.device(device)
        self.device = new_device

        for name, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(new_device))

        return self

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """
        随机采样时间步 t

        Returns:
            t: shape [B], 每个值范围在 [0, T-1]
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size 必须大于 0，当前为 {batch_size}")

        return torch.randint(
            low=0,
            high=self.timesteps,
            size=(batch_size,),
            device=self.device,
            dtype=torch.long,
        )

    @staticmethod
    def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        """
        根据 batch 中每个样本的时间步 t，从长度为 T 的系数张量 a 中提取对应值，
        再 reshape 成可广播到 x_shape 的形状。

        Args:
            a: [T]
            t: [B]
            x_shape: 通常是 x 的 shape，例如 [B, C, H, W]

        Returns:
            out: [B, 1, 1, 1]（或按 x 维度扩展）
        """
        batch_size = t.shape[0]
        out = a.gather(0, t)  # [B]
        return out.view(batch_size, *([1] * (len(x_shape) - 1)))

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从 q(x_t | x_0) 采样，即将 x0 加噪得到 xt

        公式：
            x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps

        Args:
            x0: [B, C, H, W]
            t:  [B]
            noise: 可选，若不传则自动生成标准高斯噪声

        Returns:
            xt: [B, C, H, W]
            noise: [B, C, H, W]
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha_bar_t = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus_alpha_bar_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x0.shape
        )

        xt = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise
        return xt, noise

    def predict_x0_from_noise(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        已知 x_t 和噪声 eps，反推出 x_0

        公式：
            x0 = (x_t - sqrt(1 - alpha_bar_t) * eps) / sqrt(alpha_bar_t)
        """
        sqrt_alpha_bar_t = self._extract(self.sqrt_alphas_cumprod, t, xt.shape)
        sqrt_one_minus_alpha_bar_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, xt.shape
        )

        x0 = (xt - sqrt_one_minus_alpha_bar_t * noise) / torch.clamp(
            sqrt_alpha_bar_t, min=1e-20
        )
        return x0

    def predict_noise_from_x0(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        x0: torch.Tensor,
    ) -> torch.Tensor:
        """
        已知 x_t 和 x_0，反推出噪声 eps
        """
        sqrt_alpha_bar_t = self._extract(self.sqrt_alphas_cumprod, t, xt.shape)
        sqrt_one_minus_alpha_bar_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, xt.shape
        )

        noise = (xt - sqrt_alpha_bar_t * x0) / torch.clamp(
            sqrt_one_minus_alpha_bar_t, min=1e-20
        )
        return noise

    def q_posterior(
        self,
        x0: torch.Tensor,
        xt: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算 q(x_{t-1} | x_t, x_0) 的 posterior 参数

        Returns:
            posterior_mean
            posterior_variance
            posterior_log_variance_clipped
        """
        posterior_mean_coef1_t = self._extract(self.posterior_mean_coef1, t, xt.shape)
        posterior_mean_coef2_t = self._extract(self.posterior_mean_coef2, t, xt.shape)

        posterior_mean = posterior_mean_coef1_t * x0 + posterior_mean_coef2_t * xt
        posterior_variance = self._extract(self.posterior_variance, t, xt.shape)
        posterior_log_variance_clipped = self._extract(
            self.posterior_log_variance_clipped, t, xt.shape
        )

        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def get_training_pair(
        self,
        x0: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        训练时最常用的封装函数

        Returns:
            {
                "x0": 原始干净样本,
                "t": 时间步,
                "noise": 加入的噪声,
                "xt": 带噪样本
            }
        """
        batch_size = x0.shape[0]

        if t is None:
            t = self.sample_timesteps(batch_size)

        xt, noise = self.q_sample(x0=x0, t=t, noise=noise)

        return {
            "x0": x0,
            "t": t,
            "noise": noise,
            "xt": xt,
        }


if __name__ == "__main__":
    # ====== 简单测试 ======
    T = 200
    B, C, H, W = 4, 1, 256, 256

    betas = torch.linspace(4e-5, 5e-3, T, dtype=torch.float32)
    diffusion = DiffusionForwardProcess(betas=betas, device="cpu")

    x0 = torch.randn(B, C, H, W)
    t = diffusion.sample_timesteps(B)

    out = diffusion.get_training_pair(x0, t=t)

    print("timesteps:", diffusion.timesteps)
    print("x0 shape:", out["x0"].shape)
    print("t shape:", out["t"].shape)
    print("noise shape:", out["noise"].shape)
    print("xt shape:", out["xt"].shape)

    # 测试从 xt 和 noise 反推 x0
    x0_recon = diffusion.predict_x0_from_noise(
        xt=out["xt"],
        t=out["t"],
        noise=out["noise"],
    )

    diff = (x0 - x0_recon).abs().mean().item()
    print("reconstruction error:", diff)
