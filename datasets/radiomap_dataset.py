import os
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def seed_dataloader_worker(worker_id: int) -> None:
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_sampling_map_from_psi(
    psi: np.ndarray,
    sample_rate: float = 0.10,
    sampling_mode: str = "random",
    seed: Optional[int] = None,
    return_mask: bool = False,
) -> np.ndarray:
    if psi.ndim != 2:
        raise ValueError(f"psi 应为二维数组 [H,W]，当前 shape={psi.shape}")

    if not (0.0 < sample_rate <= 1.0):
        raise ValueError(f"sample_rate 必须在 (0,1]，当前是 {sample_rate}")

    H, W = psi.shape
    N = H * W
    K = max(1, int(round(N * sample_rate)))

    sampling_mode = sampling_mode.lower()

    if sampling_mode == "random":
        rng = np.random.default_rng(seed)
        flat_idx = rng.choice(N, size=K, replace=False)

    elif sampling_mode == "uniform":
        rows_n = max(1, int(round(np.sqrt(K * H / W))))
        cols_n = max(1, int(round(K / rows_n)))

        rows = np.linspace(0, H - 1, rows_n, dtype=np.int64)
        cols = np.linspace(0, W - 1, cols_n, dtype=np.int64)

        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        flat_idx = (rr.reshape(-1) * W + cc.reshape(-1)).astype(np.int64)
        flat_idx = np.unique(flat_idx)

    else:
        raise ValueError(
            f"Unsupported sampling_mode: {sampling_mode}, expected random or uniform"
        )

    s = np.zeros_like(psi, dtype=np.float32)
    mask = np.zeros_like(psi, dtype=np.float32)
    psi_flat = psi.reshape(-1)
    s_flat = s.reshape(-1)
    s_flat[flat_idx] = psi_flat[flat_idx]
    mask.reshape(-1)[flat_idx] = 1.0

    if return_mask:
        return s, mask
    return s


class RadioMapNPZDataset(Dataset):
    """
    回退稳定版 Dataset：无 edge。

    兼容两类 npz：

    旧版:
        q, l, psi, z, id

    AFF新版:
        q, los, depth, psi, z, id

    返回:
        id
        q       -> [1,H,W]
        l       -> [1,H,W]    # depth 映射为 l
        psi     -> [1,H,W]
        z       -> [1]
        los     -> [1,H,W]    # 如果 npz 中存在
        s       -> [1,H,W]    # use_sampling=True 时动态生成
    """

    def __init__(
        self,
        data_dir: str,
        image_size: int = 256,
        use_sampling: bool = False,
        sample_rate: float = 0.10,
        sampling_mode: str = "random",
        verify_files: bool = False,
        deterministic_sampling: bool = False,
        condition_l_key: str = "depth",
    ) -> None:
        super().__init__()

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"data_dir 不存在: {data_dir}")

        self.data_dir = data_dir
        self.image_size = image_size
        self.use_sampling = use_sampling
        self.sample_rate = sample_rate
        self.sampling_mode = sampling_mode.lower()
        self.deterministic_sampling = deterministic_sampling
        self.condition_l_key = condition_l_key.lower()

        self.files: List[str] = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.lower().endswith(".npz")
        ]

        self.files.sort()

        if len(self.files) == 0:
            raise ValueError(f"{data_dir} 下没有找到 .npz 文件")

        print(f"[Dataset] {data_dir} -> {len(self.files)} samples")

        if verify_files:
            self._verify_files()

    def _verify_files(self) -> None:
        bad_files = []

        for path in self.files:
            try:
                with np.load(path, allow_pickle=True) as data:
                    required_base = ["q", "psi", "z", "id"]

                    for key in required_base:
                        if key not in data.files:
                            raise KeyError(f"缺少 key: {key}")

                    if "l" not in data.files and "depth" not in data.files:
                        raise KeyError("缺少 key: l 或 depth")

                    q = data["q"]
                    if self.condition_l_key == "los":
                        if "los" not in data.files:
                            raise KeyError("Missing key: los")
                        l = data["los"]
                    else:
                        l = data["depth"] if "depth" in data.files else data["l"]
                    psi = data["psi"]

                    if q.shape != (self.image_size, self.image_size):
                        raise ValueError(f"q shape 错误: {q.shape}")

                    if l.shape != (self.image_size, self.image_size):
                        raise ValueError(f"l/depth shape 错误: {l.shape}")

                    if psi.shape != (self.image_size, self.image_size):
                        raise ValueError(f"psi shape 错误: {psi.shape}")

                    if "los" in data.files:
                        los = data["los"]
                        if los.shape != (self.image_size, self.image_size):
                            raise ValueError(f"los shape 错误: {los.shape}")

            except Exception as e:
                bad_files.append((path, str(e)))

        if bad_files:
            msg = "\n".join([f"{p} | {e}" for p, e in bad_files[:20]])
            raise RuntimeError(f"发现损坏/格式错误的 npz（前20条）：\n{msg}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        path = self.files[index]

        try:
            with np.load(path, allow_pickle=True) as data:
                q = data["q"].astype(np.float32)

                if self.condition_l_key == "los":
                    if "los" not in data.files:
                        raise KeyError("Missing key: los")
                    l = data["los"].astype(np.float32)
                elif "depth" in data.files:
                    l = data["depth"].astype(np.float32)
                elif "l" in data.files:
                    l = data["l"].astype(np.float32)
                else:
                    raise KeyError("缺少 key: depth 或 l")

                psi = data["psi"].astype(np.float32)
                z = np.float32(data["z"])
                sample_id = str(data["id"])

                los = None
                if "los" in data.files:
                    los = data["los"].astype(np.float32)

        except Exception as e:
            raise RuntimeError(f"读取 npz 失败: {path} | {e}")

        if q.shape != (self.image_size, self.image_size):
            raise ValueError(f"{path} 中 q shape 错误: {q.shape}")

        if l.shape != (self.image_size, self.image_size):
            raise ValueError(f"{path} 中 l/depth shape 错误: {l.shape}")

        if psi.shape != (self.image_size, self.image_size):
            raise ValueError(f"{path} 中 psi shape 错误: {psi.shape}")

        sample = {
            "id": sample_id,
            "q": torch.from_numpy(q).float().unsqueeze(0),
            "l": torch.from_numpy(l).float().unsqueeze(0),
            "psi": torch.from_numpy(psi).float().unsqueeze(0),
            "z": torch.tensor([z], dtype=torch.float32),
        }

        if los is not None:
            if los.shape != (self.image_size, self.image_size):
                raise ValueError(f"{path} 中 los shape 错误: {los.shape}")

            sample["los"] = torch.from_numpy(los).float().unsqueeze(0)

        if self.use_sampling:
            seed = index if self.deterministic_sampling else None

            s, s_mask = make_sampling_map_from_psi(
                psi=psi,
                sample_rate=self.sample_rate,
                sampling_mode=self.sampling_mode,
                seed=seed,
                return_mask=True,
            )

            sample["s"] = torch.from_numpy(s).float().unsqueeze(0)
            sample["s_mask"] = torch.from_numpy(s_mask).float().unsqueeze(0)

        return sample


def build_dataset_from_config(
    data_dir: str,
    cfg: Dict[str, Any],
    split: Optional[str] = None,
) -> RadioMapNPZDataset:
    data_cfg = cfg.get("data", cfg)

    if split == "train":
        deterministic_sampling = bool(
            data_cfg.get("deterministic_sampling_for_train", False)
        )
    elif split in ["val", "test"]:
        deterministic_sampling = bool(
            data_cfg.get("deterministic_sampling_for_eval", True)
        )
    else:
        deterministic_sampling = False

    return RadioMapNPZDataset(
        data_dir=data_dir,
        image_size=int(data_cfg.get("image_size", 256)),
        use_sampling=bool(data_cfg.get("use_sampling", False)),
        sample_rate=float(data_cfg.get("sample_rate", 0.10)),
        sampling_mode=str(data_cfg.get("sampling_mode", "random")),
        verify_files=bool(data_cfg.get("verify_files", False)),
        deterministic_sampling=deterministic_sampling,
        condition_l_key=str(data_cfg.get("condition_l_key", "depth")),
    )


def build_dataloader_from_config(
    data_dir: str,
    cfg: Dict[str, Any],
    shuffle: bool = False,
    drop_last: bool = False,
    split: Optional[str] = None,
) -> DataLoader:
    dataset = build_dataset_from_config(
        data_dir=data_dir,
        cfg=cfg,
        split=split,
    )

    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})
    eval_cfg = cfg.get("eval", {})

    if split in ["val", "test"]:
        batch_size = int(
            eval_cfg.get(
                "batch_size",
                train_cfg.get("batch_size", 1),
            )
        )
    else:
        batch_size = int(train_cfg.get("batch_size", 1))

    num_workers = int(data_cfg.get("num_workers", 4))
    pin_memory = bool(data_cfg.get("pin_memory", True))
    persistent_workers = bool(data_cfg.get("persistent_workers", False))

    if num_workers == 0:
        persistent_workers = False

    seed = int(cfg.get("project", {}).get("seed", cfg.get("seed", 42)))
    generator = torch.Generator()
    generator.manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        worker_init_fn=seed_dataloader_worker,
        generator=generator,
    )

    return loader


if __name__ == "__main__":
    import yaml

    cfg_path = "../configs/base_pl_aff_nosampling.yaml"

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_dir = cfg["data"]["train_dir"]

    loader = build_dataloader_from_config(
        data_dir=train_dir,
        cfg=cfg,
        shuffle=False,
        drop_last=False,
        split="train",
    )

    batch = next(iter(loader))

    print("keys     :", batch.keys())
    print("q shape  :", batch["q"].shape)
    print("l shape  :", batch["l"].shape)
    print("psi shape:", batch["psi"].shape)
    print("z shape  :", batch["z"].shape)
    print("id       :", batch["id"][0])

    if "los" in batch:
        print("los shape:", batch["los"].shape)

    if "s" in batch:
        print("s shape  :", batch["s"].shape)
        s = batch["s"][0, 0].numpy()
        ratio = (s != 0).sum() / s.size
        print("sample rate (approx):", ratio)
