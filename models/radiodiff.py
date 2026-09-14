from typing import Optional, Dict, Any

import torch
import torch.nn as nn

from models.embeddings import (
    TimeEmbedding,
    AltitudeEmbedding,
    FeatureConditioning,
    add_condition_to_feature,
)
from models.blocks import ResidualBlock, DownBlock, UpBlock
from models.transformer import TransformerBlock
from models.adaptive_fusion import AdaptiveFusion


class SpatialTransformerStack(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int = 4,
        mlp_dim: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


class RadioDiffUNet(nn.Module):
    """
    RadioDiff U-Net + Bottleneck Transformer

    普通模式:
        无采样:
            cond = [q, l]

        有采样:
            cond = [q, l, s]

    AFF模式:
        geo = AFF(los, depth)

        q 不参与 AFF 权重学习，而是作为固定 BS 条件保留。

        无采样:
            cond = [q, geo]
            cond_in_channels = 2

        有采样:
            cond = [q, geo, s]
            cond_in_channels = 3

    AFF 权重:
        weight[:, 0] = los_weight
        weight[:, 1] = depth_weight

    注意:
        这里的 AFF 是 LoS vs Depth 的 softmax 比较。
        los_weight + depth_weight = 1。
    """

    def __init__(
        self,
        use_sampling: bool = False,
        use_aff: bool = False,
        image_channels: int = 1,
        q_channels: int = 1,
        l_channels: int = 1,
        s_channels: int = 1,
        base_channels: int = 128,
        embed_dim: int = 64,
        transformer_dim: int = 128,
        transformer_heads: int = 4,
        transformer_mlp_dim: int = 256,
        transformer_depth: int = 2,
        out_channels: int = 1,
        aff_hidden_dim: int = 16,
        aff_temperature: float = 1.0,
    ) -> None:
        super().__init__()

        self.use_sampling = use_sampling
        self.use_aff = use_aff

        self.image_channels = image_channels
        self.q_channels = q_channels
        self.l_channels = l_channels
        self.s_channels = s_channels
        self.base_channels = base_channels
        self.embed_dim = embed_dim
        self.transformer_dim = transformer_dim
        self.transformer_heads = transformer_heads
        self.transformer_mlp_dim = transformer_mlp_dim
        self.transformer_depth = transformer_depth
        self.out_channels = out_channels

        if self.use_aff:
            self.aff = AdaptiveFusion(
                n_inputs=2,
                hidden_dim=aff_hidden_dim,
                temperature=aff_temperature,
            )

            self.cond_in_channels = 2 + (
                s_channels if self.use_sampling else 0
            )

        else:
            self.aff = None

            self.cond_in_channels = q_channels + l_channels

            if self.use_sampling:
                self.cond_in_channels += s_channels

        self.time_embed = TimeEmbedding(
            base_dim=embed_dim,
            out_dim=embed_dim,
            hidden_dim=embed_dim,
            activation="silu",
        )

        self.alt_embed = AltitudeEmbedding(
            out_dim=embed_dim,
            hidden_dim=embed_dim,
            activation="silu",
        )

        self.feature_conditioning = FeatureConditioning(
            emb_dim=embed_dim,
            feat_dim=base_channels,
            activation="silu",
        )

        self.noisy_block = ResidualBlock(
            in_ch=image_channels,
            out_ch=base_channels,
            norm_type="batch",
            activation="relu",
        )

        self.cond_block = ResidualBlock(
            in_ch=self.cond_in_channels,
            out_ch=base_channels,
            norm_type="batch",
            activation="relu",
        )

        self.fuse_conv = nn.Conv2d(
            in_channels=base_channels * 2,
            out_channels=base_channels,
            kernel_size=1,
        )

        self.pre_transformers = nn.Identity()

        self.enc1 = DownBlock(
            in_ch=base_channels,
            out_ch=base_channels,
            norm_type="batch",
            activation="relu",
        )

        self.enc2 = DownBlock(
            in_ch=base_channels,
            out_ch=base_channels,
            norm_type="batch",
            activation="relu",
        )

        self.bottleneck_in = ResidualBlock(
            in_ch=base_channels,
            out_ch=base_channels,
            norm_type="batch",
            activation="relu",
        )

        if transformer_dim != base_channels:
            raise ValueError(
                f"当前实现要求 transformer_dim == base_channels，"
                f"但收到 transformer_dim={transformer_dim}, base_channels={base_channels}"
            )

        self.bottleneck_transformers = SpatialTransformerStack(
            dim=base_channels,
            depth=transformer_depth,
            num_heads=transformer_heads,
            mlp_dim=transformer_mlp_dim,
            dropout=0.0,
        )

        self.bottleneck_out = ResidualBlock(
            in_ch=base_channels,
            out_ch=base_channels,
            norm_type="batch",
            activation="relu",
        )

        self.dec1 = UpBlock(
            in_ch=base_channels,
            skip_ch=base_channels,
            out_ch=64,
            norm_type="batch",
            activation="relu",
        )

        self.dec2 = UpBlock(
            in_ch=64,
            skip_ch=base_channels,
            out_ch=64,
            norm_type="batch",
            activation="relu",
        )

        self.out_conv = nn.Conv2d(
            in_channels=64,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )

        self.last_aff_weight = None

    def _build_condition(
        self,
        q: torch.Tensor,
        l: torch.Tensor,
        s: Optional[torch.Tensor] = None,
        los: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_aff:
            if los is None:
                raise ValueError("use_aff=True 时必须传入 los")

            geo, weight = self.aff(
                los=los,
                depth=l,
            )

            self.last_aff_weight = weight.detach()

            if self.use_sampling:
                if s is None:
                    raise ValueError("use_sampling=True 时必须传入 s")

                cond = torch.cat(
                    [
                        q,
                        geo,
                        s,
                    ],
                    dim=1,
                )
            else:
                cond = torch.cat(
                    [
                        q,
                        geo,
                    ],
                    dim=1,
                )

            return cond

        if self.use_sampling:
            if s is None:
                raise ValueError("当前 use_sampling=True，但 forward 没有传入 s")

            return torch.cat(
                [
                    q,
                    l,
                    s,
                ],
                dim=1,
            )

        return torch.cat(
            [
                q,
                l,
            ],
            dim=1,
        )

    def forward(
        self,
        psi_t: torch.Tensor,
        q: torch.Tensor,
        l: torch.Tensor,
        t: torch.Tensor,
        z: torch.Tensor,
        s: Optional[torch.Tensor] = None,
        los: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        t_emb = self.time_embed(t)

        z_emb = self.alt_embed(z)

        cond_vec = self.feature_conditioning(
            t_emb,
            z_emb,
        )

        cond = self._build_condition(
            q=q,
            l=l,
            s=s,
            los=los,
        )

        psi_feat = self.noisy_block(psi_t)

        cond_feat = self.cond_block(cond)

        psi_feat = add_condition_to_feature(
            psi_feat,
            cond_vec,
        )

        cond_feat = add_condition_to_feature(
            cond_feat,
            cond_vec,
        )

        x = torch.cat(
            [
                psi_feat,
                cond_feat,
            ],
            dim=1,
        )

        x = self.fuse_conv(x)

        x = self.pre_transformers(x)

        skip0 = x

        x1 = self.enc1(x)

        skip1 = x1

        x2 = self.enc2(x1)

        x2 = self.bottleneck_in(x2)

        x2 = self.bottleneck_transformers(x2)

        x2 = self.bottleneck_out(x2)

        x = self.dec1(
            x2,
            skip1,
        )

        x = self.dec2(
            x,
            skip0,
        )

        eps_hat = self.out_conv(x)

        return eps_hat


def build_radiodiff_from_config(cfg: Dict[str, Any]) -> RadioDiffUNet:
    model_cfg = cfg.get("model", cfg)
    data_cfg = cfg.get("data", {})

    use_sampling = bool(
        model_cfg.get(
            "use_sampling",
            data_cfg.get("use_sampling", False),
        )
    )

    use_aff = bool(
        model_cfg.get(
            "use_aff",
            False,
        )
    )

    aff_cfg = model_cfg.get(
        "aff",
        {},
    )

    model = RadioDiffUNet(
        use_sampling=use_sampling,
        use_aff=use_aff,
        image_channels=int(model_cfg.get("image_channels", 1)),
        q_channels=int(model_cfg.get("q_channels", 1)),
        l_channels=int(model_cfg.get("l_channels", 1)),
        s_channels=int(model_cfg.get("s_channels", 1)),
        base_channels=int(model_cfg.get("base_channels", 128)),
        embed_dim=int(model_cfg.get("embed_dim", 64)),
        transformer_dim=int(model_cfg.get("transformer_dim", 128)),
        transformer_heads=int(model_cfg.get("transformer_heads", 4)),
        transformer_mlp_dim=int(model_cfg.get("transformer_mlp_dim", 256)),
        transformer_depth=int(model_cfg.get("transformer_depth", 2)),
        out_channels=int(model_cfg.get("out_channels", 1)),
        aff_hidden_dim=int(aff_cfg.get("hidden_dim", 16)),
        aff_temperature=float(aff_cfg.get("temperature", 1.0)),
    )

    return model