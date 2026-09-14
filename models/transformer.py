from typing import Optional

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """
    Transformer 中的前馈网络 FFN

    结构：
        Linear(dim, mlp_dim)
        -> activation
        -> dropout
        -> Linear(mlp_dim, dim)
        -> dropout
    """

    def __init__(
        self,
        dim: int,
        mlp_dim: int,
        activation: str = "gelu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.fc1 = nn.Linear(dim, mlp_dim)
        self.act = self._build_activation(activation)
        self.drop1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(mlp_dim, dim)
        self.drop2 = nn.Dropout(dropout)

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
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.drop2(x)
        return x


class TransformerBlock(nn.Module):
    """
    适用于 feature map 的 Transformer Block

    输入:
        x: [B, C, H, W]

    过程:
        [B, C, H, W]
          -> flatten -> [B, HW, C]
          -> self-attention
          -> feed-forward
          -> reshape back -> [B, C, H, W]

    参数:
        dim: 通道维度 C
        num_heads: 多头注意力头数
        mlp_dim: FFN 中间维度
        dropout: dropout 概率
    """

    def __init__(
        self,
        dim: int = 128,
        num_heads: int = 4,
        mlp_dim: int = 256,
        dropout: float = 0.0,
        activation: str = "gelu",
    ) -> None:
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(
                f"dim 必须能被 num_heads 整除，当前 dim={dim}, num_heads={num_heads}"
            )

        self.dim = dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(
            dim=dim,
            mlp_dim=mlp_dim,
            activation=activation,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]

        Returns:
            out: [B, C, H, W]
        """
        if x.dim() != 4:
            raise ValueError(f"TransformerBlock 输入应为 [B,C,H,W]，当前为 {x.shape}")

        b, c, h, w = x.shape
        if c != self.dim:
            raise ValueError(
                f"输入通道数与 Transformer dim 不一致，got C={c}, expected={self.dim}"
            )

        # [B, C, H, W] -> [B, HW, C]
        x_seq = x.flatten(2).transpose(1, 2)

        # Self-attention with residual
        x_norm = self.norm1(x_seq)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_seq = x_seq + attn_out

        # Feed-forward with residual
        x_norm = self.norm2(x_seq)
        ffn_out = self.ffn(x_norm)
        x_seq = x_seq + ffn_out

        # [B, HW, C] -> [B, C, H, W]
        out = x_seq.transpose(1, 2).reshape(b, c, h, w)
        return out


if __name__ == "__main__":
    # ===== 简单测试 =====
    B, C, H, W = 2, 128, 64, 64

    x = torch.randn(B, C, H, W)

    block = TransformerBlock(
        dim=128,
        num_heads=4,
        mlp_dim=256,
        dropout=0.0,
        activation="gelu",
    )

    y = block(x)

    print("input shape :", x.shape)
    print("output shape:", y.shape)