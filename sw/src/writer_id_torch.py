from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


IMU_COLUMNS = ["ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps"]


def resample_sequence(x: np.ndarray, target_len: int) -> np.ndarray:
    """Linear resampling from (T, C) to (target_len, C)."""
    t, c = x.shape
    if t == target_len:
        return x.astype(np.float32)
    old_idx = np.linspace(0.0, 1.0, t)
    new_idx = np.linspace(0.0, 1.0, target_len)
    out = np.zeros((target_len, c), dtype=np.float32)
    for i in range(c):
        out[:, i] = np.interp(new_idx, old_idx, x[:, i])
    return out


def preprocess_segment(x: np.ndarray, target_len: int) -> np.ndarray:
    """
    x is (T, 6): [ax, ay, az, gx, gy, gz].

    Returns (target_len, 8) after per-segment centering + magnitudes.
    """
    x = x.astype(np.float32)
    x[:, :3] -= x[:, :3].mean(axis=0, keepdims=True)
    x[:, 3:] -= x[:, 3:].mean(axis=0, keepdims=True)

    acc_mag = np.linalg.norm(x[:, :3], axis=1, keepdims=True)
    gyr_mag = np.linalg.norm(x[:, 3:], axis=1, keepdims=True)
    x = np.hstack([x, acc_mag, gyr_mag])
    return resample_sequence(x, target_len=target_len)


def load_segment_csv(path: Path, imu_columns: Sequence[str] = IMU_COLUMNS) -> np.ndarray:
    """Loads one IMU sample CSV and returns an array with shape (T, 6)."""
    df = pd.read_csv(path, comment="#")

    if not set(imu_columns).issubset(df.columns):
        # Supports files with metadata rows above the header from prompted_record.py.
        with path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        header_idx = None
        for i, line in enumerate(lines):
            if "ax_g" in line and "gz_dps" in line:
                header_idx = i
                break
        if header_idx is None:
            raise ValueError(f"Could not find IMU header in {path}")
        df = pd.read_csv(path, skiprows=header_idx)

    return df.loc[:, imu_columns].values.astype(np.float32)


class WriterDataset(Dataset):
    """
    Expected layout:

    dataset_root/
      writer_alice/
        sample_001.csv
        sample_002.csv
      writer_bob/
        sample_001.csv
    """

    def __init__(
        self,
        dataset_root: Path,
        target_len: int = 96,
        min_timesteps: int = 16,
        channel_mean: Optional[np.ndarray] = None,
        channel_std: Optional[np.ndarray] = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.target_len = target_len
        self.min_timesteps = min_timesteps

        self.samples: List[Tuple[Path, int]] = []
        self.writer_to_idx: Dict[str, int] = {}

        writer_dirs = sorted([p for p in self.dataset_root.iterdir() if p.is_dir()])
        for writer_idx, writer_dir in enumerate(writer_dirs):
            self.writer_to_idx[writer_dir.name] = writer_idx
            for csv_path in sorted(writer_dir.rglob("*.csv")):
                self.samples.append((csv_path, writer_idx))

        if not self.samples:
            raise ValueError(f"No CSV files found under {self.dataset_root}")

        self.idx_to_writer = {idx: name for name, idx in self.writer_to_idx.items()}

        if channel_mean is None or channel_std is None:
            self.channel_mean, self.channel_std = self._estimate_norm_stats()
        else:
            self.channel_mean = channel_mean.astype(np.float32)
            self.channel_std = channel_std.astype(np.float32)

    def _estimate_norm_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        stack = []
        for csv_path, _ in self.samples:
            seq = load_segment_csv(csv_path)
            if seq.shape[0] < self.min_timesteps:
                continue
            proc = preprocess_segment(seq, target_len=self.target_len)
            stack.append(proc)

        if not stack:
            raise ValueError("No valid samples survived min_timesteps filter")

        arr = np.concatenate(stack, axis=0)  # (N * T, C)
        mean = arr.mean(axis=0).astype(np.float32)
        std = arr.std(axis=0).astype(np.float32)
        std = np.maximum(std, 1e-6)
        return mean, std

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path, writer_idx = self.samples[idx]
        seq = load_segment_csv(path)

        if seq.shape[0] < self.min_timesteps:
            raise ValueError(f"Sequence too short: {path}")

        x = preprocess_segment(seq, target_len=self.target_len)
        x = (x - self.channel_mean) / self.channel_std

        # Conv1d expects (B, C, T)
        x_tensor = torch.from_numpy(x.T).float()
        y_tensor = torch.tensor(writer_idx, dtype=torch.long)
        return x_tensor, y_tensor


class ResidualTemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 5, dropout: float = 0.2):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.proj(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.gelu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        out = F.gelu(out + identity)
        return out


class WriterEncoder(nn.Module):
    """Temporal CNN + BiGRU encoder returning normalized embeddings."""

    def __init__(self, in_channels: int = 8, embed_dim: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            ResidualTemporalBlock(in_channels, 64),
            ResidualTemporalBlock(64, 96),
            ResidualTemporalBlock(96, 128),
        )
        self.bigru = nn.GRU(
            input_size=128,
            hidden_size=96,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.proj = nn.Linear(192, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        feat = self.backbone(x)  # (B, 128, T)
        seq = feat.transpose(1, 2)  # (B, T, 128)
        gru_out, _ = self.bigru(seq)  # (B, T, 192)

        # Attention-like weighted pooling from temporal energy.
        weights = torch.softmax(gru_out.pow(2).mean(dim=2), dim=1).unsqueeze(-1)
        pooled = (gru_out * weights).sum(dim=1)

        emb = self.proj(pooled)
        emb = F.normalize(emb, p=2, dim=1)
        return emb


class WriterIdNet(nn.Module):
    def __init__(self, num_writers: int, embed_dim: int = 128):
        super().__init__()
        self.encoder = WriterEncoder(in_channels=8, embed_dim=embed_dim)
        self.classifier = nn.Linear(embed_dim, num_writers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.encoder(x)
        logits = self.classifier(emb)
        return logits, emb


@dataclass
class TrainConfig:
    dataset_root: str
    output_dir: str = "writer_runs/latest"
    target_len: int = 96
    batch_size: int = 64
    epochs: int = 40
    lr: float = 1e-3
    weight_decay: float = 1e-4
    val_split: float = 0.2
    seed: int = 42
    device: str = "cuda"


def split_indices(n: int, val_split: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(math.floor(n * val_split)))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return train_idx, val_idx


def evaluate(model: WriterIdNet, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits, _ = model(xb)
            loss = F.cross_entropy(logits, yb)

            total_loss += float(loss.item()) * xb.size(0)
            pred = logits.argmax(dim=1)
            correct += int((pred == yb).sum().item())
            total += xb.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def train_writer_model(cfg: TrainConfig) -> Dict[str, object]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    full_ds = WriterDataset(Path(cfg.dataset_root), target_len=cfg.target_len)
    train_idx, val_idx = split_indices(len(full_ds), cfg.val_split, cfg.seed)

    train_ds = torch.utils.data.Subset(full_ds, indices=train_idx)
    val_ds = torch.utils.data.Subset(full_ds, indices=val_idx)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    has_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cfg.device == "cuda" and has_cuda else "cpu")

    model = WriterIdNet(num_writers=len(full_ds.writer_to_idx), embed_dim=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best = {"val_acc": -1.0, "epoch": -1}
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()

        scheduler.step()

        train_loss, train_acc = evaluate(model, train_loader, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best["val_acc"]:
            best = {"val_acc": val_acc, "epoch": epoch}
            torch.save(model.state_dict(), out_dir / "writer_id_model.pt")

    np.save(out_dir / "channel_mean.npy", full_ds.channel_mean)
    np.save(out_dir / "channel_std.npy", full_ds.channel_std)
    np.save(out_dir / "writer_labels.npy", np.array(list(full_ds.writer_to_idx.keys()), dtype=object))

    with (out_dir / "train_config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    meta = {
        "best_val_acc": best["val_acc"],
        "best_epoch": best["epoch"],
        "num_writers": len(full_ds.writer_to_idx),
        "num_samples": len(full_ds),
        "output_dir": str(out_dir),
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


class WriterRegistry:
    """
    Tracks known users as embedding prototypes and supports unknown-user rejection.

    Typical real-time flow:
    1) Segment one writing stroke (from your existing motion threshold code).
    2) Call predict_or_unknown(segment).
    3) If unknown but user confirms identity, call enroll(writer_id, [segment]).
    """

    def __init__(
        self,
        model: WriterIdNet,
        channel_mean: np.ndarray,
        channel_std: np.ndarray,
        target_len: int = 96,
        unknown_threshold: float = 0.72,
        device: str = "cpu",
    ):
        self.model = model.eval()
        self.target_len = target_len
        self.unknown_threshold = unknown_threshold
        self.channel_mean = channel_mean.astype(np.float32)
        self.channel_std = np.maximum(channel_std.astype(np.float32), 1e-6)
        self.device = torch.device(device)
        self.model.to(self.device)
        self.prototypes: Dict[str, np.ndarray] = {}

    def _embed_one(self, segment: np.ndarray) -> np.ndarray:
        x = preprocess_segment(segment, target_len=self.target_len)
        x = (x - self.channel_mean) / self.channel_std
        x = torch.from_numpy(x.T).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            emb = self.model.encoder(x).squeeze(0).cpu().numpy().astype(np.float32)
        emb /= np.linalg.norm(emb) + 1e-8
        return emb

    def enroll(self, writer_id: str, segments: Iterable[np.ndarray]) -> None:
        embs = [self._embed_one(seg) for seg in segments]
        if not embs:
            raise ValueError("No segments passed to enroll")
        proto = np.mean(np.stack(embs, axis=0), axis=0)
        proto /= np.linalg.norm(proto) + 1e-8
        self.prototypes[writer_id] = proto.astype(np.float32)

    def update_writer(self, writer_id: str, segment: np.ndarray, momentum: float = 0.85) -> None:
        if writer_id not in self.prototypes:
            self.enroll(writer_id, [segment])
            return
        emb = self._embed_one(segment)
        new_proto = momentum * self.prototypes[writer_id] + (1.0 - momentum) * emb
        new_proto /= np.linalg.norm(new_proto) + 1e-8
        self.prototypes[writer_id] = new_proto.astype(np.float32)

    def predict_or_unknown(self, segment: np.ndarray) -> Tuple[Optional[str], float]:
        if not self.prototypes:
            return None, 0.0
        emb = self._embed_one(segment)

        best_writer = None
        best_sim = -1.0
        for writer_id, proto in self.prototypes.items():
            sim = float(np.dot(emb, proto))
            if sim > best_sim:
                best_sim = sim
                best_writer = writer_id

        if best_sim < self.unknown_threshold:
            return None, best_sim
        return best_writer, best_sim

    def save_registry(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: v.tolist() for k, v in self.prototypes.items()}
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def load_registry(self, path: Path) -> None:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        self.prototypes = {k: np.asarray(v, dtype=np.float32) for k, v in payload.items()}


def load_model_bundle(run_dir: Path, device: str = "cpu") -> Tuple[WriterIdNet, np.ndarray, np.ndarray, np.ndarray]:
    writer_labels = np.load(run_dir / "writer_labels.npy", allow_pickle=True)
    channel_mean = np.load(run_dir / "channel_mean.npy")
    channel_std = np.load(run_dir / "channel_std.npy")

    model = WriterIdNet(num_writers=len(writer_labels), embed_dim=128)
    state = torch.load(run_dir / "writer_id_model.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, channel_mean, channel_std, writer_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train writer-ID model on IMU segments")
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="writer_runs/latest")
    parser.add_argument("--target-len", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        target_len=args.target_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_split=args.val_split,
        seed=args.seed,
        device=args.device,
    )
    summary = train_writer_model(cfg)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
