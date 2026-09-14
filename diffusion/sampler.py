from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from diffusion.forward_process import DiffusionForwardProcess


class DDPMSampler:
    """
    DDPM 采样器

    适配当前 3D-RadioDiff 第一版：
        model 输入:
            psi_t, q, l, t, z, s=None
        model 输出:
            pred_noise
    """

    def __init__(
        self,
        model: torch.nn.Module,
        diffusion: DiffusionForwardProcess,
        device: Union[torch.device, str] = "cpu",
        clip_denoised: bool = False,
        clip_range: Tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        self.model = model
        self.diffusion = diffusion
        self.device = torch.device(device)
        self.clip_denoised = clip_denoised
        self.clip_range = clip_range

        self.model.to(self.device)
        self.diffusion.to(self.device)

    def _predict_x0_and_noise(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        q: torch.Tensor,
        l: torch.Tensor,
        z: torch.Tensor,
        s: Optional[torch.Tensor] = None,
        los: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pred_noise = self.model(
            psi_t=xt,
            q=q,
            l=l,
            t=t,
            z=z,
            s=s,
            los=los,
        )

        pred_x0 = self.diffusion.predict_x0_from_noise(
            xt=xt,
            t=t,
            noise=pred_noise,
        )

        if self.clip_denoised:
            pred_x0 = pred_x0.clamp(self.clip_range[0], self.clip_range[1])

        return pred_x0, pred_noise

    def p_mean_variance(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        q: torch.Tensor,
        l: torch.Tensor,
        z: torch.Tensor,
        s: Optional[torch.Tensor] = None,
        los: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pred_x0, _ = self._predict_x0_and_noise(
            xt=xt,
            t=t,
            q=q,
            l=l,
            z=z,
            s=s,
            los=los,
        )

        model_mean, posterior_variance, posterior_log_variance = self.diffusion.q_posterior(
            x0=pred_x0,
            xt=xt,
            t=t,
        )

        return model_mean, posterior_variance, posterior_log_variance, pred_x0

    @torch.no_grad()
    def p_sample(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        q: torch.Tensor,
        l: torch.Tensor,
        z: torch.Tensor,
        s: Optional[torch.Tensor] = None,
        los: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        model_mean, _, posterior_log_variance, pred_x0 = self.p_mean_variance(
            xt=xt,
            t=t,
            q=q,
            l=l,
            z=z,
            s=s,
            los=los,
        )

        nonzero_mask = (t != 0).float().view(-1, *([1] * (xt.dim() - 1)))
        noise = torch.randn_like(xt)

        x_prev = model_mean + nonzero_mask * torch.exp(0.5 * posterior_log_variance) * noise
        return x_prev, pred_x0

    @torch.no_grad()
    def sample_loop(
        self,
        shape: Tuple[int, int, int, int],
        q: torch.Tensor,
        l: torch.Tensor,
        z: torch.Tensor,
        s: Optional[torch.Tensor] = None,
        los: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        return_all_steps: bool = False,
    ) -> Dict[str, Any]:
        b, c, h, w = shape

        if noise is None:
            xt = torch.randn(shape, device=self.device)
        else:
            xt = noise.to(self.device)

        q = q.to(self.device)
        l = l.to(self.device)
        z = z.to(self.device)
        if z.dim() > 1:
            z = z.view(-1)
        if s is not None:
            s = s.to(self.device)
        if los is not None:
            los = los.to(self.device)

        all_steps: List[torch.Tensor] = []
        pred_x0_steps: List[torch.Tensor] = []

        if return_all_steps:
            all_steps.append(xt.detach().cpu())

        for time_step in reversed(range(self.diffusion.timesteps)):
            t = torch.full((b,), time_step, device=self.device, dtype=torch.long)

            xt, pred_x0 = self.p_sample(
                xt=xt,
                t=t,
                q=q,
                l=l,
                z=z,
                s=s,
                los=los,
            )

            if return_all_steps:
                all_steps.append(xt.detach().cpu())
                pred_x0_steps.append(pred_x0.detach().cpu())

        result = {
            "sample": xt,
        }

        if return_all_steps:
            result["all_steps"] = all_steps
            result["pred_x0_steps"] = pred_x0_steps

        return result

    @torch.no_grad()
    def sample(
        self,
        q: torch.Tensor,
        l: torch.Tensor,
        z: torch.Tensor,
        s: Optional[torch.Tensor] = None,
        los: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        image_channels: int = 1,
    ) -> torch.Tensor:
        b, _, h, w = q.shape

        out = self.sample_loop(
            shape=(b, image_channels, h, w),
            q=q,
            l=l,
            z=z,
            s=s,
            los=los,
            noise=noise,
            return_all_steps=False,
        )
        return out["sample"]

    @torch.no_grad()
    def reconstruct_from_xt(
        self,
        xt: torch.Tensor,
        start_t: int,
        q: torch.Tensor,
        l: torch.Tensor,
        z: torch.Tensor,
        s: Optional[torch.Tensor] = None,
        los: Optional[torch.Tensor] = None,
        return_all_steps: bool = False,
    ) -> Dict[str, Any]:
        if start_t < 0 or start_t >= self.diffusion.timesteps:
            raise ValueError(
                f"start_t 越界: {start_t}, 应在 [0, {self.diffusion.timesteps - 1}]"
            )

        xt = xt.to(self.device)
        q = q.to(self.device)
        l = l.to(self.device)
        z = z.to(self.device)
        if z.dim() > 1:
            z = z.view(-1)
        if s is not None:
            s = s.to(self.device)
        if los is not None:
            los = los.to(self.device)

        b = xt.shape[0]

        all_steps: List[torch.Tensor] = []
        pred_x0_steps: List[torch.Tensor] = []

        if return_all_steps:
            all_steps.append(xt.detach().cpu())

        for time_step in reversed(range(start_t + 1)):
            t = torch.full((b,), time_step, device=self.device, dtype=torch.long)

            xt, pred_x0 = self.p_sample(
                xt=xt,
                t=t,
                q=q,
                l=l,
                z=z,
                s=s,
                los=los,
            )

            if return_all_steps:
                all_steps.append(xt.detach().cpu())
                pred_x0_steps.append(pred_x0.detach().cpu())

        result = {
            "sample": xt,
        }

        if return_all_steps:
            result["all_steps"] = all_steps
            result["pred_x0_steps"] = pred_x0_steps

        return result
