import argparse
import sys
from pathlib import Path

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.radiomap_dataset import build_dataloader_from_config
from diffusion.beta_schedule import get_beta_schedule
from models.radiodiff import build_radiodiff_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate one PL AFF experiment.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config_path = config_path.resolve()

    with config_path.open("r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    data_cfg = cfg["data"]
    diffusion_cfg = cfg["diffusion"]
    assert data_cfg["use_sampling"] is True
    sampling_mode = str(data_cfg["sampling_mode"]).lower()
    assert sampling_mode in {"random", "uniform"}
    assert float(data_cfg["sample_rate"]) == 0.10
    assert cfg["model"]["use_aff"] is True
    assert int(diffusion_cfg["timesteps"]) == 200

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check the PyTorch/CUDA installation.")

    device = torch.device("cuda:0")
    print(f"CUDA_VISIBLE_DEVICES maps physical GPU to: {torch.cuda.get_device_name(0)}")
    print(f"Config: {config_path}")
    print(
        "Task: path loss | condition: AFF(depth + LoS) | "
        f"sampling: {sampling_mode} 10%"
    )

    loader = build_dataloader_from_config(
        data_dir=data_cfg["train_dir"],
        cfg=cfg,
        shuffle=False,
        drop_last=False,
        split="train",
    )
    batch = next(iter(loader))
    sample_fraction = batch["s_mask"].float().mean().item()
    print(f"Dataset samples: {len(loader.dataset)}")
    print(f"Actual {sampling_mode} sample fraction: {sample_fraction:.6f}")

    model = build_radiodiff_from_config(cfg).to(device).eval()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    with torch.no_grad():
        psi = batch["psi"][:1].to(device)
        output = model(
            psi_t=torch.randn_like(psi),
            q=batch["q"][:1].to(device),
            l=batch["l"][:1].to(device),
            t=torch.tensor([199], device=device, dtype=torch.long),
            z=batch["z"][:1].view(-1).to(device),
            s=batch["s"][:1].to(device),
            los=batch["los"][:1].to(device),
        )
    print(f"Model parameters: {parameters / 1e6:.2f} M")
    print(f"Input shape: {tuple(psi.shape)} | output shape: {tuple(output.shape)}")

    betas = get_beta_schedule(
        schedule_name=diffusion_cfg["beta_schedule"],
        timesteps=int(diffusion_cfg["timesteps"]),
        beta_start=float(diffusion_cfg["beta_start"]),
        beta_end=float(diffusion_cfg["beta_end"]),
    )
    alpha_bar = torch.cumprod(1.0 - betas, dim=0)[-1]
    print(
        "Terminal diffusion state: "
        f"alpha_bar={alpha_bar.item():.8f}, "
        f"signal_scale={alpha_bar.sqrt().item():.8f}, "
        f"noise_scale={(1.0 - alpha_bar).sqrt().item():.8f}"
    )
    print("CHECK PASSED")


if __name__ == "__main__":
    main()
