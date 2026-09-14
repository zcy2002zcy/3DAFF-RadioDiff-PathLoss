from typing import Optional, Dict, Any

import torch
import torch.nn as nn

from models.blocks import ResidualBlock, DownBlock, UpBlock
from models.transformer import TransformerBlock
from models.embeddings import add_condition_to_feature


class ConditionalUNet(nn.Module):
    """
    条件 U-Net 主干

    输入:
        x: [B, in_ch, H, W]
        cond_vec: [B, base_ch] 或 None

    输出:
        out: [B, out_ch, H, W]

    设计说明：
    - encoder / decoder 采用 U-Net 结构
    - bottleneck 中插入 TransformerBlock
    - cond_vec 可在多个层级注入，用于融合 t/z 等条件

    这份实现适合作为 3D-RadioDiff 主干的一部分。
    """

    def __init__(
        self,
        in_ch: int = 128,
        out_ch: int = 64,
        base_ch: int = 128,
        transformer_dim: int = 128,
        transformer_heads: int = 4,
        transformer_mlp_dim: int = 256,
        norm_type: str = "batch",
        activation: str = "relu",
        use_dropout: bool = False,
        dropout: float = 0.0,
        use_bottleneck_transformer: bool = True,
    ) -> None:
        super().__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.base_ch = base_ch
        self.use_bottleneck_transformer = use_bottleneck_transformer

        # ==============
        # input projection
        # ==============
        self.input_block = ResidualBlock(
            in_ch=in_ch,
            out_ch=base_ch,
            norm_type=norm_type,
            activation=activation,
            use_dropout=use_dropout,
            dropout=dropout,
        )

        # ==============
        # encoder
        # ==============
        self.enc1 = DownBlock(
            in_ch=base_ch,
            out_ch=base_ch,
            norm_type=norm_type,
            activation=activation,
            use_dropout=use_dropout,
            dropout=dropout,
        )   # H -> H/2

        self.enc2 = DownBlock(
            in_ch=base_ch,
            out_ch=base_ch,
            norm_type=norm_type,
            activation=activation,
            use_dropout=use_dropout,
            dropout=dropout,
        )   # H/2 -> H/4

        # ==============
        # bottleneck
        # ==============
        self.bottleneck_res1 = ResidualBlock(
            in_ch=base_ch,
            out_ch=base_ch,
            norm_type=norm_type,
            activation=activation,
            use_dropout=use_dropout,
            dropout=dropout,
        )

        if self.use_bottleneck_transformer:
            self.bottleneck_transformer = TransformerBlock(
                dim=transformer_dim,
                num_heads=transformer_heads,
                mlp_dim=transformer_mlp_dim,
                dropout=dropout,
                activation="gelu",
            )
        else:
            self.bottleneck_transformer = nn.Identity()

        self.bottleneck_res2 = ResidualBlock(
            in_ch=base_ch,
            out_ch=base_ch,
            norm_type=norm_type,
            activation=activation,
            use_dropout=use_dropout,
            dropout=dropout,
        )

        # ==============
        # decoder
        # ==============
        self.dec1 = UpBlock(
            in_ch=base_ch,
            skip_ch=base_ch,
            out_ch=64,
            norm_type=norm_type,
            activation=activation,
            use_dropout=use_dropout,
            dropout=dropout,
        )   # H/4 -> H/2

        self.dec2 = UpBlock(
            in_ch=64,
            skip_ch=base_ch,
            out_ch=64,
            norm_type=norm_type,
            activation=activation,
            use_dropout=use_dropout,
            dropout=dropout,
        )   # H/2 -> H

        # ==============
        # output projection
        # ==============
        self.output_block = ResidualBlock(
            in_ch=64,
            out_ch=64,
            norm_type=norm_type,
            activation=activation,
            use_dropout=use_dropout,
            dropout=dropout,
        )

        self.out_conv = nn.Conv2d(
            in_channels=64,
            out_channels=out_ch,
            kernel_size=3,
            padding=1,
        )

    def _inject_condition(
        self,
        feat: torch.Tensor,
        cond_vec: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        如果 cond_vec 不为空，则把条件向量加到 feature 上。
        """
        if cond_vec is None:
            return feat

        if feat.shape[1] != cond_vec.shape[1]:
            raise ValueError(
                f"cond_vec channel 与 feature 不匹配: feature={feat.shape}, cond_vec={cond_vec.shape}"
            )

        return add_condition_to_feature(feat, cond_vec)

    def forward(
        self,
        x: torch.Tensor,
        cond_vec: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, in_ch, H, W]
            cond_vec: [B, base_ch] or None

        Returns:
            out: [B, out_ch, H, W]
        """
        # input block
        x0 = self.input_block(x)          # [B, base_ch, H, W]
        x0 = self._inject_condition(x0, cond_vec)

        # encoder
        x1 = self.enc1(x0)                # [B, base_ch, H/2, W/2]
        x1 = self._inject_condition(x1, cond_vec)

        x2 = self.enc2(x1)                # [B, base_ch, H/4, W/4]
        x2 = self._inject_condition(x2, cond_vec)

        # bottleneck
        xb = self.bottleneck_res1(x2)
        xb = self._inject_condition(xb, cond_vec)

        xb = self.bottleneck_transformer(xb)

        xb = self.bottleneck_res2(xb)
        xb = self._inject_condition(xb, cond_vec)

        # decoder
        x = self.dec1(xb, x1)             # [B, 64, H/2, W/2]
        x = self.dec2(x, x0)              # [B, 64, H, W]

        # output
        x = self.output_block(x)
        out = self.out_conv(x)

        return out


def build_unet_from_config(cfg: Dict[str, Any]) -> ConditionalUNet:
    """
    从配置字典构建 ConditionalUNet
    """
    model_cfg = cfg.get("model", cfg)

    return ConditionalUNet(
        in_ch=int(model_cfg.get("unet_in_channels", 128)),
        out_ch=int(model_cfg.get("unet_out_channels", 64)),
        base_ch=int(model_cfg.get("base_channels", 128)),
        transformer_dim=int(model_cfg.get("transformer_dim", 128)),
        transformer_heads=int(model_cfg.get("transformer_heads", 4)),
        transformer_mlp_dim=int(model_cfg.get("transformer_mlp_dim", 256)),
        norm_type=str(model_cfg.get("norm_type", "batch")),
        activation=str(model_cfg.get("activation", "relu")),
        use_dropout=bool(model_cfg.get("use_dropout", False)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        use_bottleneck_transformer=bool(model_cfg.get("use_bottleneck_transformer", True)),
    )


if __name__ == "__main__":
    # ===== 简单测试 =====
    B, C, H, W = 2, 128, 256, 256

    x = torch.randn(B, C, H, W)
    cond_vec = torch.randn(B, 128)

    model = ConditionalUNet(
        in_ch=128,
        out_ch=64,
        base_ch=128,
        transformer_dim=128,
        transformer_heads=4,
        transformer_mlp_dim=256,
        norm_type="batch",
        activation="relu",
        use_dropout=False,
        dropout=0.0,
        use_bottleneck_transformer=True,
    )

    y = model(x, cond_vec=cond_vec)

    print("input shape :", x.shape)
    print("cond shape  :", cond_vec.shape)
    print("output shape:", y.shape)