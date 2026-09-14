from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_activation(name: str = "relu") -> nn.Module:
    """
    构造激活函数
    """
    name = name.lower()

    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "silu" or name == "swish":
        return nn.SiLU()
    if name == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.2, inplace=True)

    raise ValueError(f"不支持的激活函数: {name}")


def build_norm(
    norm_type: str,
    num_channels: int,
    num_groups: int = 8,
) -> nn.Module:
    """
    构造归一化层

    Args:
        norm_type:
            - "batch"
            - "group"
            - "instance"
            - "none"
    """
    norm_type = norm_type.lower()

    if norm_type == "batch":
        return nn.BatchNorm2d(num_channels)

    if norm_type == "group":
        groups = min(num_groups, num_channels)
        while num_channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, num_channels)

    if norm_type == "instance":
        return nn.InstanceNorm2d(num_channels, affine=True)

    if norm_type == "none":
        return nn.Identity()

    raise ValueError(f"不支持的归一化类型: {norm_type}")


class ResidualBlock(nn.Module):
    """
    残差块

    结构：
        conv -> norm -> act
        conv -> norm
        + skip
        -> act

    对应你现在项目里的基本卷积残差单元。
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        norm_type: str = "batch",
        activation: str = "relu",
        kernel_size: int = 3,
        use_dropout: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        padding = kernel_size // 2

        self.conv1 = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.norm1 = build_norm(norm_type, out_ch)
        self.act1 = build_activation(activation)

        self.conv2 = nn.Conv2d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.norm2 = build_norm(norm_type, out_ch)

        self.dropout = nn.Dropout2d(dropout) if use_dropout and dropout > 0 else nn.Identity()

        # skip connection
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        else:
            self.skip = nn.Identity()

        self.out_act = build_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)

        out = self.dropout(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out = out + identity
        out = self.out_act(out)

        return out


class DownBlock(nn.Module):
    """
    U-Net encoder 下采样模块

    结构：
        stride=2 conv downsample
        -> residual block
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        norm_type: str = "batch",
        activation: str = "relu",
        use_dropout: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.down = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )

        self.res = ResidualBlock(
            in_ch=out_ch,
            out_ch=out_ch,
            norm_type=norm_type,
            activation=activation,
            use_dropout=use_dropout,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        x = self.res(x)
        return x


class UpBlock(nn.Module):
    """
    U-Net decoder 上采样模块

    结构：
        ConvTranspose2d 上采样
        -> 与 skip concat
        -> residual block
    """

    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        norm_type: str = "batch",
        activation: str = "relu",
        use_dropout: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )

        self.res = ResidualBlock(
            in_ch=out_ch + skip_ch,
            out_ch=out_ch,
            norm_type=norm_type,
            activation=activation,
            use_dropout=use_dropout,
            dropout=dropout,
        )

    @staticmethod
    def _match_size(x: torch.Tensor, skip: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        当上采样后尺寸与 skip 不完全一致时，做对齐。
        """
        _, _, hx, wx = x.shape
        _, _, hs, ws = skip.shape

        if hx == hs and wx == ws:
            return x, skip

        # 用插值把 x 调整到 skip 的尺寸
        x = F.interpolate(x, size=(hs, ws), mode="bilinear", align_corners=False)
        return x, skip

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x, skip = self._match_size(x, skip)

        x = torch.cat([x, skip], dim=1)
        x = self.res(x)
        return x


if __name__ == "__main__":
    # ===== 简单测试 =====
    B = 2

    # ResidualBlock
    x = torch.randn(B, 3, 256, 256)
    block = ResidualBlock(in_ch=3, out_ch=128, norm_type="batch", activation="relu")
    y = block(x)
    print("ResidualBlock out:", y.shape)

    # DownBlock
    down = DownBlock(in_ch=128, out_ch=128, norm_type="batch", activation="relu")
    y_down = down(y)
    print("DownBlock out:", y_down.shape)

    # UpBlock
    skip = torch.randn(B, 128, 256, 256)
    up = UpBlock(in_ch=128, skip_ch=128, out_ch=64, norm_type="batch", activation="relu")
    y_up = up(y_down, skip)
    print("UpBlock out:", y_up.shape)
