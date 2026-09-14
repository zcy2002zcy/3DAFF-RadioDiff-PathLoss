import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.radiomap_dataset import build_dataloader_from_config
from diffusion.beta_schedule import get_beta_schedule
from diffusion.forward_process import DiffusionForwardProcess
from diffusion.sampler import DDPMSampler
from models.radiodiff import build_radiodiff_from_config
from scripts.evaluate import (
    _gaussian_window,
    compute_nmse,
    compute_psnr,
    compute_rmse,
    compute_ssim,
    move_batch_to_device,
)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_project_path(path: str) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str((PROJECT_ROOT / p).resolve())


def build_diffusion(cfg: Dict[str, Any], device: torch.device) -> DiffusionForwardProcess:
    diffusion_cfg = cfg["diffusion"]
    betas = get_beta_schedule(
        schedule_name=diffusion_cfg.get("beta_schedule", "linear"),
        timesteps=int(diffusion_cfg.get("timesteps", 200)),
        beta_start=float(diffusion_cfg.get("beta_start", 4e-5)),
        beta_end=float(diffusion_cfg.get("beta_end", 5e-3)),
    )
    diffusion = DiffusionForwardProcess(betas=betas, device=device)

    alpha_bar_t = diffusion.alphas_cumprod[-1].detach().float().cpu()
    signal_scale = torch.sqrt(alpha_bar_t).item()
    noise_scale = torch.sqrt(1.0 - alpha_bar_t).item()
    print(
        "Diffusion terminal state | "
        f"alpha_bar_T={alpha_bar_t.item():.6f} | "
        f"signal_scale={signal_scale:.6f} | "
        f"noise_scale={noise_scale:.6f}"
    )
    if signal_scale > 0.1:
        print(
            "Warning: this schedule does not fully destroy the clean signal by the "
            "last diffusion step. Full random-noise sampling is still a valid "
            "generation protocol, but it may be harder than fixed-step denoising."
        )

    return diffusion


def load_model(cfg: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    model = build_radiodiff_from_config(cfg).to(device)

    ckpt_dir = cfg.get("checkpoint", {}).get("save_dir", "outputs/checkpoints")
    ckpt_path = Path(ckpt_dir) / "best.pt"
    if not ckpt_path.is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path
    ckpt_path = ckpt_path.resolve()

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded model from: {ckpt_path}")
    return model


@torch.no_grad()
def evaluate_full_random_noise(
    cfg: Dict[str, Any],
    save_dir: str,
    max_samples: int,
    seed: int,
    enforce_measurement_consistency: bool,
) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Evaluation protocol: full DDPM reverse sampling from psi_T ~ N(0, I)")
    print(
        "The complete ground-truth psi is used only for metrics; "
        "only its authorized 10% measurements enter the model through s."
    )

    data_cfg = cfg["data"]
    eval_cfg = cfg.setdefault("eval", {})
    eval_cfg["mode"] = "sample_full_random_noise"
    eval_cfg["use_sampler"] = True
    eval_cfg.pop("denoise_t", None)
    if max_samples > 0:
        eval_cfg["max_samples"] = max_samples

    test_loader = build_dataloader_from_config(
        data_dir=data_cfg["test_dir"],
        cfg=cfg,
        shuffle=False,
        drop_last=False,
        split="test",
    )
    print(f"Test size: {len(test_loader.dataset)}")

    model = load_model(cfg, device)
    diffusion = build_diffusion(cfg, device)
    sampler = DDPMSampler(
        model=model,
        diffusion=diffusion,
        device=device,
        clip_denoised=True,
    )

    save_dir_abs = resolve_project_path(save_dir)
    os.makedirs(save_dir_abs, exist_ok=True)
    save_first_n = int(eval_cfg.get("save_first_n", 10))

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

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
    records: List[Dict[str, Any]] = []

    for i, batch in enumerate(tqdm(test_loader, desc="Full random-noise eval")):
        if max_samples > 0 and total_count >= max_samples:
            break

        batch = move_batch_to_device(batch, device)
        psi = batch["psi"]
        q = batch["q"]
        l = batch["l"]
        los = batch.get("los", None)
        z = batch["z"].view(-1)
        s = batch.get("s", None)
        s_mask = batch.get("s_mask", None)
        ids = batch.get("id", None)

        b = psi.shape[0]
        if max_samples > 0 and total_count + b > max_samples:
            keep = max_samples - total_count
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
            b = keep

        generator = torch.Generator(device=device)
        generator.manual_seed(seed + total_count)
        init_noise = torch.randn(
            psi.shape,
            device=device,
            dtype=psi.dtype,
            generator=generator,
        )

        pred_x0 = sampler.sample(
            q=q,
            l=l,
            z=z,
            s=s,
            los=los,
            noise=init_noise,
            image_channels=1,
        )

        if enforce_measurement_consistency:
            if s is None:
                raise ValueError("Measurement consistency requires sparse map s.")
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

        pred_metric = pred_x0.float()
        psi_metric = psi.float()
        batch_loss = torch.mean((pred_metric - psi_metric) ** 2, dim=(1, 2, 3))
        batch_nmse = compute_nmse(pred_metric, psi_metric)
        batch_rmse = compute_rmse(pred_metric, psi_metric)
        batch_ssim = compute_ssim(pred_metric, psi_metric, ssim_window, data_range=2.0)
        batch_psnr = compute_psnr(pred_metric, psi_metric, data_range=2.0)

        total_loss += float(batch_loss.sum().item())
        total_nmse += float(batch_nmse.sum().item())
        total_rmse += float(batch_rmse.sum().item())
        total_ssim += float(batch_ssim.sum().item())
        total_psnr += float(batch_psnr.sum().item())
        total_count += b

        if batch_rmse_before_consistency is not None:
            total_rmse_before_consistency += float(batch_rmse_before_consistency.sum().item())
            total_sample_rmse_before_consistency += float(
                batch_sample_rmse_before_consistency.sum().item()
            )
            total_sample_fraction += float(batch_sample_fraction.sum().item())
            consistency_count += b

        if i < save_first_n:
            np.savez_compressed(
                os.path.join(save_dir_abs, f"sample_{i:04d}.npz"),
                pred=pred_x0.detach().cpu().numpy(),
                gt=psi.detach().cpu().numpy(),
                q=q.detach().cpu().numpy(),
                l=l.detach().cpu().numpy(),
                s=s.detach().cpu().numpy() if s is not None else None,
                init_noise=init_noise.detach().cpu().numpy(),
                z=z.detach().cpu().numpy(),
                id=np.array(ids) if ids is not None else None,
            )

        ids_list = list(ids) if ids is not None else [f"sample_{i}_{j}" for j in range(b)]
        for j in range(b):
            records.append(
                {
                    "id": ids_list[j],
                    "mode": "sample_full_random_noise",
                    "nmse": float(batch_nmse[j].item()),
                    "rmse": float(batch_rmse[j].item()),
                    "ssim": float(batch_ssim[j].item()),
                    "psnr": float(batch_psnr[j].item()),
                }
            )

    summary: Dict[str, float] = {
        "mode": "sample_full_random_noise",
        "loss": total_loss / max(total_count, 1),
        "nmse": total_nmse / max(total_count, 1),
        "rmse": total_rmse / max(total_count, 1),
        "ssim": total_ssim / max(total_count, 1),
        "psnr": total_psnr / max(total_count, 1),
        "count": total_count,
        "seed": seed,
        "enforce_measurement_consistency": enforce_measurement_consistency,
    }
    if consistency_count > 0:
        summary.update(
            {
                "rmse_before_consistency": total_rmse_before_consistency
                / max(consistency_count, 1),
                "sample_rmse_before_consistency": total_sample_rmse_before_consistency
                / max(consistency_count, 1),
                "sample_fraction": total_sample_fraction / max(consistency_count, 1),
            }
        )

    pd.DataFrame(records).to_csv(
        os.path.join(save_dir_abs, "metrics_per_sample.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame([summary]).to_csv(
        os.path.join(save_dir_abs, "metrics_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    print("\n========================================")
    print("Eval Mode : sample_full_random_noise")
    print(f"Eval Loss : {summary['loss']:.6f}")
    print(f"Eval NMSE : {summary['nmse']:.6f}")
    print(f"Eval RMSE : {summary['rmse']:.6f}")
    print(f"Eval SSIM : {summary['ssim']:.6f}")
    print(f"Eval PSNR : {summary['psnr']:.6f}")
    if consistency_count > 0:
        print(f"Consistency RMSE before : {summary['rmse_before_consistency']:.6f}")
        print(
            "Consistency sample RMSE before : "
            f"{summary['sample_rmse_before_consistency']:.6f}"
        )
        print(f"Consistency sample fraction : {summary['sample_fraction']:.6f}")
    print(f"Count : {total_count}")
    print(f"Saved to : {save_dir_abs}")
    print("========================================\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate RadioDiff by full reverse sampling from random Gaussian noise."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--save-dir", default="")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--no-consistency", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    project_seed = int(cfg.get("project", {}).get("seed", cfg.get("seed", 42)))
    seed = project_seed if args.seed < 0 else args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    if args.save_dir:
        save_dir = args.save_dir
    else:
        output_dir = cfg.get("train", {}).get("output_dir", "outputs")
        suffix = "sample_full" if args.max_samples <= 0 else f"sample_full_{args.max_samples}"
        save_dir = str(Path(output_dir) / f"eval_results_{suffix}")

    eval_cfg = cfg.setdefault("eval", {})
    enforce_consistency = bool(
        eval_cfg.get("enforce_measurement_consistency", True)
    ) and not args.no_consistency

    evaluate_full_random_noise(
        cfg=cfg,
        save_dir=save_dir,
        max_samples=args.max_samples,
        seed=seed,
        enforce_measurement_consistency=enforce_consistency,
    )


if __name__ == "__main__":
    main()
