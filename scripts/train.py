import os
import sys
import time
import random
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# ========= 璁╄剼鏈兘浠庨」鐩牴鐩綍瀵煎叆 =========
FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.radiomap_dataset import build_dataloader_from_config
from models.radiodiff import build_radiodiff_from_config
from diffusion.beta_schedule import get_beta_schedule
from diffusion.forward_process import DiffusionForwardProcess
from diffusion.losses import build_loss_from_config


# =========================================================
# 鍩虹宸ュ叿
# =========================================================
def load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"閰嶇疆鏂囦欢涓嶅瓨鍦? {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_torch_runtime(cfg: Dict[str, Any]) -> None:
    deterministic = bool(cfg.get("train", {}).get("deterministic", False))
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        print("Torch runtime: deterministic=True, cudnn.benchmark=False")
    else:
        torch.backends.cudnn.benchmark = True
        print("Torch runtime: deterministic=False, cudnn.benchmark=True")


def cuda_autocast(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def build_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_selected_config(cfg: Dict[str, Any], log_dir: str) -> None:
    ensure_dir(log_dir)
    path = os.path.join(log_dir, "selected_config.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    print(f"Selected config saved: {path}")


def print_diffusion_summary(diffusion: DiffusionForwardProcess) -> None:
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
            "Warning: terminal diffusion state still contains visible signal. "
            "Full random-noise sampling may be harder unless the model is trained "
            "with a stronger/noisier schedule."
        )


def get_device(cfg: Dict[str, Any]) -> torch.device:
    train_cfg = cfg.get("train", {})
    device_name = str(train_cfg.get("device", "cuda")).lower()

    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")

    if device_name == "cpu":
        return torch.device("cpu")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def maybe_wrap_dataparallel(model: torch.nn.Module, cfg: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    device_ids = [int(x) for x in cfg.get("train", {}).get("device_ids", [])]
    if device.type == "cuda" and len(device_ids) > 1:
        visible_count = torch.cuda.device_count()
        valid_ids = [idx for idx in device_ids if idx < visible_count]
        if len(valid_ids) > 1:
            print("DataParallel device_ids:", valid_ids)
            return torch.nn.DataParallel(model, device_ids=valid_ids, output_device=valid_ids[0])
        print(f"DataParallel skipped: requested={device_ids}, visible={visible_count}")
    return model


def save_checkpoint(
    save_path: str,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[Any],
    best_val_loss: float,
    cfg: Dict[str, Any],
) -> None:
    state = {
        "epoch": epoch,
        "model_state_dict": unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "best_val_loss": best_val_loss,
        "config": cfg,
    }
    torch.save(state, save_path)


def nmse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    pred / target: [B, C, H, W]
    """
    num = torch.sum((pred - target) ** 2, dim=(1, 2, 3))
    den = torch.sum(target ** 2, dim=(1, 2, 3)) + eps
    return (num / den).mean()


# =========================================================
# 鍗曡疆璁粌
# =========================================================
def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    diffusion: DiffusionForwardProcess,
    criterion: torch.nn.Module,
    device: torch.device,
    epoch: int,
    cfg: Dict[str, Any],
    scaler: Optional[Any],
) -> Dict[str, float]:
    model.train()

    train_cfg = cfg.get("train", {})
    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    grad_clip = float(train_cfg.get("grad_clip", 0.0))
    log_cfg = cfg.get("log", {})
    log_interval = int(log_cfg.get("log_interval", 50))

    running_loss = 0.0
    running_nmse = 0.0
    total_steps = 0

    start_time = time.time()

    for step, batch in enumerate(loader):
        batch = move_batch_to_device(batch, device)

        psi = batch["psi"]          # [B,1,H,W]
        q = batch["q"]              # [B,1,H,W]
        l = batch["l"]              # [B,1,H,W]
        los = batch.get("los", None)
        z = batch["z"].view(-1)     # [B]
        s = batch.get("s", None)    # [B,1,H,W] or None

        # 鍓嶅悜鎵╂暎锛歺0 -> xt
        pair = diffusion.get_training_pair(psi)
        psi_t = pair["xt"]
        t = pair["t"]
        noise = pair["noise"]

        optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda":
            with cuda_autocast(enabled=use_amp):
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
                batch_nmse = nmse(pred_x0, psi)
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
            batch_nmse = nmse(pred_x0, psi)

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()

            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

        running_loss += float(loss.item())
        running_nmse += float(batch_nmse.item())
        total_steps += 1

        if (step + 1) % log_interval == 0:
            elapsed = time.time() - start_time
            avg_loss = running_loss / total_steps
            avg_nmse = running_nmse / total_steps
            print(
                f"[Train] Epoch {epoch:03d} | Step {step+1:05d}/{len(loader):05d} "
                f"| Loss {avg_loss:.6f} | NMSE {avg_nmse:.6f} | Time {elapsed:.1f}s"
            )

    return {
        "loss": running_loss / max(total_steps, 1),
        "nmse": running_nmse / max(total_steps, 1),
    }


# =========================================================
# 鍗曡疆楠岃瘉
# =========================================================
@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    diffusion: DiffusionForwardProcess,
    criterion: torch.nn.Module,
    device: torch.device,
    epoch: int,
    cfg: Dict[str, Any],
) -> Dict[str, float]:
    model.eval()

    train_cfg = cfg.get("train", {})
    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"

    running_loss = 0.0
    running_nmse = 0.0
    total_steps = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        psi = batch["psi"]
        q = batch["q"]
        l = batch["l"]
        los = batch.get("los", None)
        z = batch["z"].view(-1)
        s = batch.get("s", None)

        pair = diffusion.get_training_pair(psi)
        psi_t = pair["xt"]
        t = pair["t"]
        noise = pair["noise"]

        if device.type == "cuda":
            with cuda_autocast(enabled=use_amp):
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
                batch_nmse = nmse(pred_x0, psi)
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
            batch_nmse = nmse(pred_x0, psi)

        running_loss += float(loss.item())
        running_nmse += float(batch_nmse.item())
        total_steps += 1

    metrics = {
        "loss": running_loss / max(total_steps, 1),
        "nmse": running_nmse / max(total_steps, 1),
    }

    print(
        f"[Val]   Epoch {epoch:03d} | Loss {metrics['loss']:.6f} | NMSE {metrics['nmse']:.6f}"
    )
    return metrics


# =========================================================
# 涓诲嚱鏁?
# =========================================================
def main(config_path: str = None) -> None:
    if config_path is None:
        config_path = str(PROJECT_ROOT / "configs" / "base.yaml")

    cfg = load_yaml(config_path)

    # ---------- 鍩虹閰嶇疆 ----------
    seed = int(cfg.get("project", {}).get("seed", 42))
    set_seed(seed)
    configure_torch_runtime(cfg)

    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})
    ckpt_cfg = cfg.get("checkpoint", {})
    logging_cfg = cfg.get("logging", {})
    diffusion_cfg = cfg.get("diffusion", {})

    device = get_device(cfg)
    print(f"Using device: {device}")

    # ---------- 杈撳嚭鐩綍 ----------
    output_dir = train_cfg.get("output_dir", "outputs10T")
    checkpoint_dir = ckpt_cfg.get("save_dir", os.path.join(output_dir, "checkpoints"))
    log_dir = logging_cfg.get("log_dir", os.path.join(output_dir, "logs"))

    ensure_dir(output_dir)
    ensure_dir(checkpoint_dir)
    ensure_dir(log_dir)
    save_selected_config(cfg, log_dir)

    # ---------- 鏁版嵁鐩綍 ----------
    train_dir = data_cfg["train_dir"]
    val_dir = data_cfg["val_dir"]

    # ---------- DataLoader ----------
    train_loader = build_dataloader_from_config(
        data_dir=train_dir,
        cfg=cfg,
        shuffle=True,
        drop_last=True,
        split="train",
    )
    val_loader = build_dataloader_from_config(
        data_dir=val_dir,
        cfg=cfg,
        shuffle=False,
        drop_last=False,
        split="val",
    )

    print(f"Train size: {len(train_loader.dataset)}")
    print(f"Val size  : {len(val_loader.dataset)}")

    # ---------- Model ----------
    model = build_radiodiff_from_config(cfg).to(device)
    model = maybe_wrap_dataparallel(model, cfg, device)
    print("Model built successfully.")
    print("Model use_aff:", getattr(unwrap_model(model), "use_aff", None))
    print("Model use_sampling:", getattr(unwrap_model(model), "use_sampling", None))
    print("Visible CUDA devices:", torch.cuda.device_count() if torch.cuda.is_available() else 0)

    # ---------- Diffusion ----------
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
    print_diffusion_summary(diffusion)

    # ---------- Loss ----------
    criterion = build_loss_from_config(cfg)

    # ---------- Optimizer ----------
    lr = float(train_cfg.get("lr", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    optimizer_name = str(train_cfg.get("optimizer", "adam")).lower()

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # ---------- AMP ----------
    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = build_grad_scaler(enabled=use_amp) if device.type == "cuda" else None

    # ---------- Resume ----------
    start_epoch = 1
    best_val_loss = float("inf")

    resume = bool(train_cfg.get("resume", False))
    resume_path = str(train_cfg.get("resume_path", ""))

    if resume:
        if not resume_path or not os.path.exists(resume_path):
            raise FileNotFoundError(f"resume=True 浣嗘壘涓嶅埌 checkpoint: {resume_path}")

        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        if scaler is not None and ckpt.get("scaler_state_dict", None) is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])

        start_epoch = int(ckpt["epoch"]) + 1
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))

        print(f"Resumed from: {resume_path}")
        print(f"Start epoch : {start_epoch}")
        print(f"Best val loss so far: {best_val_loss:.6f}")

    # ---------- Training Loop ----------
    epochs = int(train_cfg.get("epochs", 50))
    save_best_only = bool(ckpt_cfg.get("save_best_only", True))

    history = []

    for epoch in range(start_epoch, epochs + 1):
        print("=" * 80)
        print(f"Epoch {epoch}/{epochs}")

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            diffusion=diffusion,
            criterion=criterion,
            device=device,
            epoch=epoch,
            cfg=cfg,
            scaler=scaler,
        )

        val_metrics = validate_one_epoch(
            model=model,
            loader=val_loader,
            diffusion=diffusion,
            criterion=criterion,
            device=device,
            epoch=epoch,
            cfg=cfg,
        )

        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_nmse": train_metrics["nmse"],
            "val_loss": val_metrics["loss"],
            "val_nmse": val_metrics["nmse"],
        }
        history.append(record)

        # 淇濆瓨 history csv
        history_path = os.path.join(log_dir, "train_history.csv")
        pd.DataFrame(history).to_csv(history_path, index=False, encoding="utf-8-sig")

        # 淇濆瓨 latest checkpoint
        latest_ckpt_path = os.path.join(checkpoint_dir, "latest.pt")
        save_checkpoint(
            save_path=latest_ckpt_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            best_val_loss=best_val_loss,
            cfg=cfg,
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_ckpt_path = os.path.join(checkpoint_dir, "best.pt")
            save_checkpoint(
                save_path=best_ckpt_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                best_val_loss=best_val_loss,
                cfg=cfg,
            )
            print(f"New best model saved: {best_ckpt_path}")

        elif not save_best_only:
            epoch_ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pt")
            save_checkpoint(
                save_path=epoch_ckpt_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                best_val_loss=best_val_loss,
                cfg=cfg,
            )

        if device.type == "cuda":
            torch.cuda.empty_cache()

        print(
            f"Epoch {epoch:03d} done | "
            f"Train Loss {train_metrics['loss']:.6f} | "
            f"Val Loss {val_metrics['loss']:.6f} | "
            f"Best Val Loss {best_val_loss:.6f}"
        )

    print("=" * 80)
    print("Training finished.")
    print(f"Best Val Loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved in: {checkpoint_dir}")


if __name__ == "__main__":
    cli_config = sys.argv[1] if len(sys.argv) > 1 else None
    main(cli_config)


