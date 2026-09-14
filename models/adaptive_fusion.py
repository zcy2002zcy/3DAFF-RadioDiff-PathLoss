import torch
import torch.nn as nn


class AdaptiveFusion(nn.Module):
    """
    LoS-Depth Adaptive Fusion

    只比较:
        los
        depth

    q 不参与权重学习，q 在 radiodiff.py 中作为独立条件 concat。

    输出:
        fused  : [B, 1, H, W]
        weight : [B, 2]

    weight 是 softmax 权重，表示:
        weight[:,0] = los_weight
        weight[:,1] = depth_weight

    因为这里只比较 LoS 和 Depth，所以 softmax 是合理的。
    """

    def __init__(
        self,
        n_inputs: int = 2,
        hidden_dim: int = 16,
        temperature: float = 1.0,
    ):
        super().__init__()

        if n_inputs != 2:
            raise ValueError("AdaptiveFusion 当前只支持 n_inputs=2: los, depth")

        self.n_inputs = n_inputs
        self.hidden_dim = hidden_dim
        self.temperature = temperature

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.mlp = nn.Sequential(
            nn.Linear(n_inputs, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_inputs),
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(
        self,
        los: torch.Tensor,
        depth: torch.Tensor,
    ):
        """
        Args:
            los   : [B, 1, H, W]
            depth : [B, 1, H, W]

        Returns:
            fused  : [B, 1, H, W]
            weight : [B, 2]
        """

        los_avg = self.pool(los).flatten(1)
        depth_avg = self.pool(depth).flatten(1)

        x = torch.cat(
            [
                los_avg,
                depth_avg,
            ],
            dim=1,
        )

        logits = self.mlp(x)

        weight = self.softmax(
            logits / self.temperature
        )

        w_los = weight[:, 0].view(-1, 1, 1, 1)
        w_depth = weight[:, 1].view(-1, 1, 1, 1)

        fused = (
            w_los * los
            + w_depth * depth
        )

        return fused, weight


if __name__ == "__main__":
    B, H, W = 2, 256, 256

    los = torch.rand(B, 1, H, W)
    depth = torch.rand(B, 1, H, W)

    aff = AdaptiveFusion(
        n_inputs=2,
        hidden_dim=16,
        temperature=1.0,
    )

    fused, weight = aff(
        los=los,
        depth=depth,
    )

    print("fused shape :", fused.shape)
    print("weight shape:", weight.shape)
    print("weight:", weight)
    print("weight sum:", weight.sum(dim=1))