import os
import sys
from pathlib import Path
from typing import Dict, Any, List

import yaml
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# ========= 璺緞淇 =========
FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.radiomap_dataset import build_dataloader_from_config
from models.radiodiff import build_radiodiff_from_config
from diffusion.beta_schedule import get_beta_schedule
from diffusion.forward_process import DiffusionForwardProcess
from diffusion.losses import build_loss_from_config
from diffusion.sampler import DDPMSampler


# =========================================================
# 宸ュ叿
# =========================================================
def load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"閰嶇疆鏂囦欢涓嶅瓨鍦? {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


# =========================================================
# 鎸囨爣
# =========================================================
def compute_nmse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    pred / target: [B, C, H, W]
    return: [B]
    """
    num = torch.sum((pred - target) ** 2, dim=(1, 2, 3))
    den = torch.sum(target ** 2, dim=(1, 2, 3)) + eps
    return num / den


def compute_rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    pred / target: [B, C, H, W]
    return: [B]
    """
    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))
    return torch.sqrt(mse)


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 2.0, eps: float = 1e-8) -> torch.Tensor:
    """
    pred / target: [B, C, H, W]
    return: [B]
    榛樿 [-1,1]锛屾墍浠?data_range=2
    """
    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))
    return 10.0 * torch.log10((data_range ** 2) / (mse + eps))


def _gaussian_window(window_size: int = 11, sigma: float = 1.5, channels: int = 1, device: str = "cpu"):
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = torch.outer(g, g)
    window_2d = window_2d / window_2d.sum()
    window = window_2d.view(1, 1, window_size, window_size).repeat(channels, 1, 1, 1)
    return window


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window: torch.Tensor,
    data_range: float = 2.0,
    K1: float = 0.01,
    K2: float = 0.03,
) -> torch.Tensor:
    """
    pred / target: [B, C, H, W]
    window: [C,1,ws,ws]
    return: [B]
    """
    import torch.nn.functional as F

    B, C, H, W = pred.shape
    window_size = window.shape[-1]
    padding = window_size // 2

    mu_x = F.conv2d(pred, window, padding=padding, groups=C)
    mu_y = F.conv2d(target, window, padding=padding, groups=C)

    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(pred * pred, window, padding=padding, groups=C) - mu_x2
    sigma_y2 = F.conv2d(target * target, window, padding=padding, groups=C) - mu_y2
    sigma_xy = F.conv2d(pred * target, window, padding=padding, groups=C) - mu_xy

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
        (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    )

    return ssim_map.mean(dim=(1, 2, 3))


# =========================================================
# 涓昏瘎浼板嚱鏁?
# =========================================================
@torch.no_grad()
def evaluate(
    model,
    loader,
    diffusion,
    criterion,
    device,
    cfg,
):
    model.eval()

    eval_cfg = cfg.get("eval", {})
    train_cfg = cfg.get("train", {})
    eval_mode = str(eval_cfg.get("mode", "")).lower().strip()
    if not eval_mode:
        eval_mode = "sample" if bool(eval_cfg.get("use_sampler", False)) else "denoise"
    if eval_mode not in {"sample", "denoise", "direct"}:
        raise ValueError(f"Unsupported eval.mode: {eval_mode}, expected sample, denoise or direct")
    use_sampler = eval_mode == "sample"
    save_dir = eval_cfg.get("save_dir", "outputs/eval_results")
    save_first_n = int(eval_cfg.get("save_first_n", 10))
    max_batches = eval_cfg.get("max_batches", None)
    max_samples = eval_cfg.get("max_samples", None)
    max_batches = int(max_batches) if max_batches is not None else None
    max_samples = int(max_samples) if max_samples is not None else None
    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"

    save_dir_abs = str((PROJECT_ROOT / save_dir).resolve()) if not os.path.isabs(save_dir) else save_dir
    os.makedirs(save_dir_abs, exist_ok=True)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    if use_sampler:
        sampler = DDPMSampler(
            model=model,
            diffusion=diffusion,
            device=device,
            clip_denoised=True,
        )

    # SSIM 楂樻柉鏍革紝鎻愬墠鏋勫缓锛岄伩鍏嶆瘡涓?batch 閲嶅缓
    ssim_window = _gaussian_window(
        window_size=11,
        sigma=1.5,
        channels=1,
        device=device,
    )

    total_loss = 0.0
    total_nmse = 0.0
    total_rmse = 0.0
    total_ssim = 0.0
    total_psnr = 0.0
    total_count = 0
    total_rmse_before_consistency = 0.0
    total_sample_rmse_before_consistency = 0.0
    total_sample_fraction = 0.0
    consistency_count = 0

    per_sample_records: List[Dict[str, Any]] = []

    for i, batch in enumerate(tqdm(loader, desc="Evaluating")):
        if max_batches is not None and i >= max_batches:
            break
        if max_samples is not None and total_count >= max_samples:
            break

        batch = move_batch_to_device(batch, device)

        psi = batch["psi"]              # [B,1,H,W]
        q = batch["q"]
        l = batch["l"]
        los = batch.get("los", None)
        z = batch["z"].view(-1)
        s = batch.get("s", None)
        s_mask = batch.get("s_mask", None)
        ids = batch.get("id", None)

        if eval_mode == "denoise":
            denoise_t_cfg = eval_cfg.get("denoise_t", None)
            if denoise_t_cfg is None:
                pair = diffusion.get_training_pair(psi)
                psi_t = pair["xt"]
                t = pair["t"]
                noise = pair["noise"]
            else:
                denoise_t = int(denoise_t_cfg)
                denoise_t = max(0, min(denoise_t, diffusion.timesteps - 1))
                t = torch.full(
                    (psi.shape[0],),
                    denoise_t,
                    device=device,
                    dtype=torch.long,
                )
                psi_t, noise = diffusion.q_sample(psi, t)

            if device.type == "cuda":
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred_noise = model(
                        psi_t=psi_t,
                        q=q,
                        l=l,
                        t=t,
                        z=z,
                        s=s,
                        los=los,
                    )
                    loss = criterion(pred_noise, noise)
                    pred_x0 = diffusion.predict_x0_from_noise(
                        xt=psi_t,
                        t=t,
                        noise=pred_noise,
                    )
            else:
                pred_noise = model(
                    psi_t=psi_t,
                    q=q,
                    l=l,
                    t=t,
                    z=z,
                    s=s,
                    los=los,
                )
                loss = criterion(pred_noise, noise)
                pred_x0 = diffusion.predict_x0_from_noise(
                    xt=psi_t,
                    t=t,
                    noise=pred_noise,
                )
        elif eval_mode == "direct":
            direct_t_cfg = eval_cfg.get("direct_t", "last")
            if str(direct_t_cfg).lower() == "last":
                direct_t = diffusion.timesteps - 1
            else:
                direct_t = int(direct_t_cfg)
                direct_t = max(0, min(direct_t, diffusion.timesteps - 1))

            t = torch.full(
                (psi.shape[0],),
                direct_t,
                device=device,
                dtype=torch.long,
            )

            direct_init = str(eval_cfg.get("direct_init", "noise")).lower()
            if direct_init == "zeros":
                psi_t = torch.zeros_like(psi)
            elif direct_init == "sparse" and s is not None:
                psi_t = s
            else:
                seed = int(cfg.get("project", {}).get("seed", cfg.get("seed", 42)))
                generator = torch.Generator(device=device)
                generator.manual_seed(seed + total_count)
                psi_t = torch.randn(
                    psi.shape,
                    device=device,
                    dtype=psi.dtype,
                    generator=generator,
                )

            if device.type == "cuda":
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred_noise = model(
                        psi_t=psi_t,
                        q=q,
                        l=l,
                        t=t,
                        z=z,
                        s=s,
                        los=los,
                    )
                    pred_x0 = diffusion.predict_x0_from_noise(
                        xt=psi_t,
                        t=t,
                        noise=pred_noise,
                    )
                    pred_x0 = pred_x0.clamp(-1.0, 1.0)
                    loss = torch.mean((pred_x0.float() - psi.float()) ** 2)
            else:
                pred_noise = model(
                    psi_t=psi_t,
                    q=q,
                    l=l,
                    t=t,
                    z=z,
                    s=s,
                    los=los,
                )
                pred_x0 = diffusion.predict_x0_from_noise(
                    xt=psi_t,
                    t=t,
                    noise=pred_noise,
                )
                pred_x0 = pred_x0.clamp(-1.0, 1.0)
                loss = torch.mean((pred_x0.float() - psi.float()) ** 2)
        else:
            pred_x0 = sampler.sample(
                q=q,
                l=l,
                z=z,
                s=s,
                los=los,
                image_channels=1,
            )
            loss = torch.mean((pred_x0.float() - psi.float()) ** 2)

        if bool(eval_cfg.get("enforce_measurement_consistency", False)):
            if s is None:
                raise ValueError("enforce_measurement_consistency=True requires sparse map s")
            mask = (s_mask > 0) if s_mask is not None else (s != 0)
            pred_before_consistency = pred_x0.float()
            psi_before_consistency = psi.float()
            batch_rmse_before_consistency = compute_rmse(
                pred_before_consistency,
                psi_before_consistency,
            )
            mask_f = mask.float()
            sample_counts = mask_f.flatten(1).sum(dim=1).clamp_min(1.0)
            sample_mse_before_consistency = (
                ((pred_before_consistency - psi_before_consistency) ** 2 * mask_f)
                .flatten(1)
                .sum(dim=1)
                / sample_counts
            )
            batch_sample_rmse_before_consistency = torch.sqrt(sample_mse_before_consistency)
            batch_sample_fraction = mask_f.flatten(1).mean(dim=1)
            pred_x0 = torch.where(mask, s, pred_x0)
        else:
            batch_rmse_before_consistency = None
            batch_sample_rmse_before_consistency = None
            batch_sample_fraction = None

        # 涓轰簡鎸囨爣绋冲畾锛岃浆 float32 鍐嶇畻
        pred_x0_metric = pred_x0.float()
        psi_metric = psi.float()

        batch_nmse = compute_nmse(pred_x0_metric, psi_metric)                  # [B]
        batch_rmse = compute_rmse(pred_x0_metric, psi_metric)                  # [B]
        batch_psnr = compute_psnr(pred_x0_metric, psi_metric, data_range=2.0) # [B]
        batch_ssim = compute_ssim(
            pred_x0_metric,
            psi_metric,
            window=ssim_window,
            data_range=2.0,
        )                                                                     # [B]

        B = psi.shape[0]
        if max_samples is not None and total_count + B > max_samples:
            keep = max_samples - total_count
            if keep <= 0:
                break
            psi = psi[:keep]
            q = q[:keep]
            l = l[:keep]
            z = z[:keep]
            if s is not None:
                s = s[:keep]
            if s_mask is not None:
                s_mask = s_mask[:keep]
            if los is not None:
                los = los[:keep]
            if ids is not None:
                ids = ids[:keep]
            pred_x0 = pred_x0[:keep]
            pred_x0_metric = pred_x0_metric[:keep]
            psi_metric = psi_metric[:keep]
            batch_nmse = batch_nmse[:keep]
            batch_rmse = batch_rmse[:keep]
            batch_psnr = batch_psnr[:keep]
            batch_ssim = batch_ssim[:keep]
            if batch_rmse_before_consistency is not None:
                batch_rmse_before_consistency = batch_rmse_before_consistency[:keep]
                batch_sample_rmse_before_consistency = batch_sample_rmse_before_consistency[:keep]
                batch_sample_fraction = batch_sample_fraction[:keep]
            B = keep

        total_loss += float(loss.item()) * B
        total_nmse += float(batch_nmse.sum().item())
        total_rmse += float(batch_rmse.sum().item())
        total_ssim += float(batch_ssim.sum().item())
        total_psnr += float(batch_psnr.sum().item())
        total_count += B
        if batch_rmse_before_consistency is not None:
            total_rmse_before_consistency += float(batch_rmse_before_consistency.sum().item())
            total_sample_rmse_before_consistency += float(
                batch_sample_rmse_before_consistency.sum().item()
            )
            total_sample_fraction += float(batch_sample_fraction.sum().item())
            consistency_count += B

        # 淇濆瓨鍓嶅嚑涓?batch 鐨勭粨鏋滐紝涓嶅奖鍝嶅叏閲忕粺璁?
        if i < save_first_n:
            save_path = os.path.join(save_dir_abs, f"sample_{i:04d}.npz")
            np.savez_compressed(
                save_path,
                pred=pred_x0.detach().cpu().numpy(),
                gt=psi.detach().cpu().numpy(),
                q=q.detach().cpu().numpy(),
                l=l.detach().cpu().numpy(),
                s=s.detach().cpu().numpy() if s is not None else None,
                z=z.detach().cpu().numpy(),
                id=np.array(ids) if ids is not None else None,
            )

        # 姣忔牱鏈寚鏍囦繚瀛?
        ids_list = list(ids) if ids is not None else [f"sample_{i}_{b}" for b in range(B)]
        nmse_np = batch_nmse.detach().cpu().numpy()
        rmse_np = batch_rmse.detach().cpu().numpy()
        ssim_np = batch_ssim.detach().cpu().numpy()
        psnr_np = batch_psnr.detach().cpu().numpy()

        for b in range(B):
            per_sample_records.append(
                {
                    "id": ids_list[b],
                    "mode": eval_mode,
                    "nmse": float(nmse_np[b]),
                    "rmse": float(rmse_np[b]),
                    "ssim": float(ssim_np[b]),
                    "psnr": float(psnr_np[b]),
                }
            )

    avg_loss = total_loss / max(total_count, 1)
    avg_nmse = total_nmse / max(total_count, 1)
    avg_rmse = total_rmse / max(total_count, 1)
    avg_ssim = total_ssim / max(total_count, 1)
    avg_psnr = total_psnr / max(total_count, 1)
    avg_rmse_before_consistency = total_rmse_before_consistency / max(consistency_count, 1)
    avg_sample_rmse_before_consistency = (
        total_sample_rmse_before_consistency / max(consistency_count, 1)
    )
    avg_sample_fraction = total_sample_fraction / max(consistency_count, 1)

    # 淇濆瓨 per-sample 鎸囨爣
    pd.DataFrame(per_sample_records).to_csv(
        os.path.join(save_dir_abs, "metrics_per_sample.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    summary = {
        "mode": eval_mode,
        "denoise_t": eval_cfg.get("denoise_t", ""),
        "loss": avg_loss,
        "nmse": avg_nmse,
        "rmse": avg_rmse,
        "ssim": avg_ssim,
        "psnr": avg_psnr,
        "count": total_count,
    }
    if consistency_count > 0:
        summary.update(
            {
                "rmse_before_consistency": avg_rmse_before_consistency,
                "sample_rmse_before_consistency": avg_sample_rmse_before_consistency,
                "sample_fraction": avg_sample_fraction,
            }
        )

    pd.DataFrame([summary]).to_csv(
        os.path.join(save_dir_abs, "metrics_summary.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print("\n==============================")
    print(f"Eval Mode : {eval_mode}")
    if eval_mode == "denoise" and eval_cfg.get("denoise_t", None) is not None:
        print(f"Denoise t : {int(eval_cfg.get('denoise_t'))}")
    print(f"Eval Loss : {avg_loss:.6f}")
    print(f"Eval NMSE : {avg_nmse:.6f}")
    print(f"Eval RMSE : {avg_rmse:.6f}")
    print(f"Eval SSIM : {avg_ssim:.6f}")
    print(f"Eval PSNR : {avg_psnr:.6f}")
    if consistency_count > 0:
        print(f"Consistency RMSE before : {avg_rmse_before_consistency:.6f}")
        print(f"Consistency sample RMSE before : {avg_sample_rmse_before_consistency:.6f}")
        print(f"Consistency sample fraction : {avg_sample_fraction:.6f}")
    print("==============================\n")

    return summary


# =========================================================
# 涓诲叆鍙?
# =========================================================
def main(config_path: str = None):
    if config_path is None:
        config_path = str(PROJECT_ROOT / "configs" / "base.yaml")

    cfg = load_yaml(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_cfg = cfg["data"]
    test_dir = data_cfg["test_dir"]

    # ===== DataLoader =====
    test_loader = build_dataloader_from_config(
        data_dir=test_dir,
        cfg=cfg,
        shuffle=False,
        drop_last=False,
        split="test",
    )

    print(f"Test size: {len(test_loader.dataset)}")

    # ===== Model =====
    model = build_radiodiff_from_config(cfg).to(device)

    # Use checkpoint path from config instead of a hard-coded experiment folder.
    ckpt_dir = cfg.get("checkpoint", {}).get("save_dir", "outputs/checkpoints")
    ckpt_path = Path(ckpt_dir) / "best.pt"
    if not ckpt_path.is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path
    ckpt_path = str(ckpt_path.resolve())

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"鎵句笉鍒?checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded model from: {ckpt_path}")

    # ===== Diffusion =====
    diffusion_cfg = cfg["diffusion"]
    betas = get_beta_schedule(
        schedule_name=diffusion_cfg.get("beta_schedule", "linear"),
        timesteps=int(diffusion_cfg.get("timesteps", 200)),
        beta_start=float(diffusion_cfg.get("beta_start", 4e-5)),
        beta_end=float(diffusion_cfg.get("beta_end", 5e-3)),
    )

    diffusion = DiffusionForwardProcess(
        betas=betas,
        device=device,
    )

    # ===== Loss =====
    criterion = build_loss_from_config(cfg)

    # ===== Evaluate =====
    evaluate(
        model=model,
        loader=test_loader,
        diffusion=diffusion,
        criterion=criterion,
        device=device,
        cfg=cfg,
    )


if __name__ == "__main__":
    cli_config = sys.argv[1] if len(sys.argv) > 1 else None
    main(cli_config)



