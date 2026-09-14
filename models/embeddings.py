import math
from typing import Optional

import torch
import torch.nn as nn


class ScalarEmbedding(nn.Module):
    """
    标量 embedding
    适合对高度 z 这种单一标量做简单 MLP 编码

    默认结构：
        Linear(1, hidden_dim) -> SiLU -> Linear(hidden_dim, out_dim)

    这和你前面项目里使用的思路一致，也与论文表格中的
    fc1: Linear(1,64), fc2: Linear(64,64)
    的设计兼容。
    """

    def __init__(
        self,
        out_dim: int = 64,
        hidden_dim: int = 64,
        activation: str = "silu",
    ) -> None:
        super().__init__()

        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.act = self._build_activation(activation)

    @staticmethod
    def _build_activation(name: str) -> nn.Module:
        name = name.lower()
        if name == "relu":
            return nn.ReLU(inplace=True)
        if name == "gelu":
            return nn.GELU()
        if name == "silu" or name == "swish":
            return nn.SiLU()
        raise ValueError(f"不支持的激活函数: {name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:
                shape [B] or [B,1]

        Returns:
            emb: [B, out_dim]
        """
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        x = x.float()
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x


class SinusoidalTimeEmbedding(nn.Module):
    """
    正弦时间步 embedding
    常用于 diffusion model 的时间步 t 编码。

    输入:
        t: [B]
    输出:
        emb: [B, dim]

    这是 diffusion 模型里最常见的时间步编码方式，
    比直接把 t 喂 MLP 更稳定。
    """

    def __init__(self, dim: int) -> None:
        super().__init__()

        if dim <= 0:
            raise ValueError(f"dim 必须大于 0，当前为 {dim}")

        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [B]

        Returns:
            emb: [B, dim]
        """
        if t.dim() != 1:
            t = t.view(-1)

        device = t.device
        half_dim = self.dim // 2

        if half_dim == 0:
            return t.float().unsqueeze(-1)

        emb_factor = math.log(10000.0) / max(half_dim - 1, 1)
        emb = torch.exp(
            torch.arange(half_dim, device=device, dtype=torch.float32) * (-emb_factor)
        )

        t = t.float().unsqueeze(1)   # [B,1]
        emb = t * emb.unsqueeze(0)   # [B, half_dim]

        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # [B, 2*half_dim]

        if self.dim % 2 == 1:
            emb = torch.cat(
                [emb, torch.zeros((emb.shape[0], 1), device=device, dtype=emb.dtype)],
                dim=1,
            )

        return emb


class MLPEmbedding(nn.Module):
    """
    通用 embedding projector

    用途：
    - 把 sinusoidal t embedding 再映射到目标维度
    - 也可以用于其他 embedding 的非线性投影

    结构：
        Linear(in_dim, hidden_dim)
        -> activation
        -> Linear(hidden_dim, out_dim)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: Optional[int] = None,
        activation: str = "silu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if hidden_dim is None:
            hidden_dim = out_dim

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.act = self._build_activation(activation)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @staticmethod
    def _build_activation(name: str) -> nn.Module:
        name = name.lower()
        if name == "relu":
            return nn.ReLU(inplace=True)
        if name == "gelu":
            return nn.GELU()
        if name == "silu" or name == "swish":
            return nn.SiLU()
        raise ValueError(f"不支持的激活函数: {name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ConditionProjector(nn.Module):
    """
    将一维 embedding 向量投影到 feature 通道维

    例如：
        t_emb: [B, 64]
        z_emb: [B, 64]
        -> proj 到 [B, 128]

    常用于把 embedding 加到卷积特征图上。
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: str = "silu",
    ) -> None:
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            self._build_activation(activation),
            nn.Linear(out_dim, out_dim),
        )

    @staticmethod
    def _build_activation(name: str) -> nn.Module:
        name = name.lower()
        if name == "relu":
            return nn.ReLU(inplace=True)
        if name == "gelu":
            return nn.GELU()
        if name == "silu" or name == "swish":
            return nn.SiLU()
        raise ValueError(f"不支持的激活函数: {name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.float())


class FeatureConditioning(nn.Module):
    """
    将时间 embedding 与高度 embedding 融合成特征条件向量

    输入:
        t_emb: [B, emb_dim]
        z_emb: [B, emb_dim]

    输出:
        cond_feat: [B, feat_dim]

    用法：
        cond_feat = self.feature_conditioning(t_emb, z_emb)
        feature = feature + cond_feat[:, :, None, None]
    """

    def __init__(
        self,
        emb_dim: int = 64,
        feat_dim: int = 128,
        activation: str = "silu",
    ) -> None:
        super().__init__()

        self.proj = ConditionProjector(
            in_dim=emb_dim * 2,
            out_dim=feat_dim,
            activation=activation,
        )

    def forward(self, t_emb: torch.Tensor, z_emb: torch.Tensor) -> torch.Tensor:
        if t_emb.dim() != 2 or z_emb.dim() != 2:
            raise ValueError(
                f"t_emb 和 z_emb 都应为二维 [B,D]，当前 got t={t_emb.shape}, z={z_emb.shape}"
            )

        x = torch.cat([t_emb, z_emb], dim=-1)
        return self.proj(x)


def add_condition_to_feature(
    feature: torch.Tensor,
    cond_vec: torch.Tensor,
) -> torch.Tensor:
    """
    将 [B, C] 的条件向量加到 [B, C, H, W] 的 feature map 上

    Args:
        feature: [B, C, H, W]
        cond_vec: [B, C]

    Returns:
        out: [B, C, H, W]
    """
    if feature.dim() != 4:
        raise ValueError(f"feature 应为四维 [B,C,H,W]，当前为 {feature.shape}")

    if cond_vec.dim() != 2:
        raise ValueError(f"cond_vec 应为二维 [B,C]，当前为 {cond_vec.shape}")

    if feature.shape[0] != cond_vec.shape[0]:
        raise ValueError(
            f"batch 维不匹配: feature={feature.shape[0]}, cond_vec={cond_vec.shape[0]}"
        )

    if feature.shape[1] != cond_vec.shape[1]:
        raise ValueError(
            f"channel 维不匹配: feature={feature.shape[1]}, cond_vec={cond_vec.shape[1]}"
        )

    return feature + cond_vec.unsqueeze(-1).unsqueeze(-1)


class TimeEmbedding(nn.Module):
    """
    完整时间步 embedding 模块

    结构：
        t
        -> sinusoidal embedding
        -> MLP projection

    输出:
        [B, out_dim]
    """

    def __init__(
        self,
        base_dim: int = 64,
        out_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        activation: str = "silu",
    ) -> None:
        super().__init__()

        if out_dim is None:
            out_dim = base_dim
        if hidden_dim is None:
            hidden_dim = out_dim

        self.sinusoidal = SinusoidalTimeEmbedding(base_dim)
        self.mlp = MLPEmbedding(
            in_dim=base_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            activation=activation,
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = self.sinusoidal(t)
        emb = self.mlp(emb)
        return emb


class AltitudeEmbedding(nn.Module):
    """
    完整高度 embedding 模块

    结构：
        z
        -> ScalarEmbedding / MLP

    输出:
        [B, out_dim]
    """

    def __init__(
        self,
        out_dim: int = 64,
        hidden_dim: int = 64,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.embedding = ScalarEmbedding(
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            activation=activation,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.embedding(z)


if __name__ == "__main__":
    # ===== 简单测试 =====
    B = 4

    t = torch.randint(0, 200, (B,))
    z = torch.rand(B)

    # 1) 时间 embedding
    time_emb = TimeEmbedding(base_dim=64, out_dim=64)
    t_emb = time_emb(t)
    print("t_emb shape:", t_emb.shape)

    # 2) 高度 embedding
    alt_emb = AltitudeEmbedding(out_dim=64, hidden_dim=64)
    z_emb = alt_emb(z)
    print("z_emb shape:", z_emb.shape)

    # 3) 融合成 feature conditioning
    feat_cond = FeatureConditioning(emb_dim=64, feat_dim=128)
    cond_vec = feat_cond(t_emb, z_emb)
    print("cond_vec shape:", cond_vec.shape)

    # 4) 加到 feature map
    feat = torch.randn(B, 128, 64, 64)
    feat_out = add_condition_to_feature(feat, cond_vec)
    print("feat_out shape:", feat_out.shape)