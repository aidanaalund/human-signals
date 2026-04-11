from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from writer_id_torch import load_model_bundle, preprocess_segment


def collect_writer_files(dataset_root: Path) -> Dict[str, List[Path]]:
    out: Dict[str, List[Path]] = {}
    for writer_dir in sorted([p for p in dataset_root.iterdir() if p.is_dir()]):
        files = sorted(writer_dir.rglob("*.csv"))
        if files:
            out[writer_dir.name] = files
    return out


def read_segment_csv(path: Path) -> np.ndarray:
    import pandas as pd

    df = pd.read_csv(path)
    required = ["ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps"]
    if not set(required).issubset(df.columns):
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
    return df.loc[:, required].values.astype(np.float32)


def embed_segment(
    segment: np.ndarray,
    model,
    channel_mean: np.ndarray,
    channel_std: np.ndarray,
    target_len: int,
    device: torch.device,
) -> np.ndarray:
    x = preprocess_segment(segment, target_len=target_len)
    x = (x - channel_mean) / np.maximum(channel_std, 1e-6)
    xt = torch.from_numpy(x.T).unsqueeze(0).float().to(device)
    with torch.no_grad():
        emb = model.encoder(xt).squeeze(0).cpu().numpy().astype(np.float32)
    emb /= np.linalg.norm(emb) + 1e-8
    return emb


def calibrate(
    run_dir: Path,
    dataset_root: Path,
    device_name: str,
    target_len: int,
) -> Dict[str, float]:
    model, channel_mean, channel_std, _ = load_model_bundle(run_dir, device=device_name)
    device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")
    model.to(device).eval()

    files_by_writer = collect_writer_files(dataset_root)
    if len(files_by_writer) < 2:
        raise ValueError("Need at least 2 writers for threshold calibration")

    embs_by_writer: Dict[str, List[np.ndarray]] = {}
    for writer_id, files in files_by_writer.items():
        embs = []
        for fp in files:
            seg = read_segment_csv(fp)
            if seg.shape[0] < 16:
                continue
            embs.append(embed_segment(seg, model, channel_mean, channel_std, target_len, device))
        if len(embs) >= 2:
            embs_by_writer[writer_id] = embs

    if len(embs_by_writer) < 2:
        raise ValueError("Not enough valid writers with >=2 samples after filtering")

    known_scores: List[float] = []
    unknown_scores: List[float] = []

    writer_ids = sorted(embs_by_writer.keys())
    for writer_id in writer_ids:
        writer_embs = embs_by_writer[writer_id]
        others = [w for w in writer_ids if w != writer_id]

        for i, query in enumerate(writer_embs):
            same_set = [e for j, e in enumerate(writer_embs) if j != i]
            same_proto = np.mean(np.stack(same_set, axis=0), axis=0)
            same_proto /= np.linalg.norm(same_proto) + 1e-8
            known_scores.append(float(np.dot(query, same_proto)))

            other_sims = []
            for other_id in others:
                proto = np.mean(np.stack(embs_by_writer[other_id], axis=0), axis=0)
                proto /= np.linalg.norm(proto) + 1e-8
                other_sims.append(float(np.dot(query, proto)))
            unknown_scores.append(max(other_sims))

    candidates = np.linspace(0.35, 0.95, num=121)
    best_t = 0.72
    best_score = -1.0

    known_arr = np.asarray(known_scores, dtype=np.float32)
    unknown_arr = np.asarray(unknown_scores, dtype=np.float32)

    for t in candidates:
        tpr = float((known_arr >= t).mean())
        tnr = float((unknown_arr < t).mean())
        bal_acc = 0.5 * (tpr + tnr)
        if bal_acc > best_score:
            best_score = bal_acc
            best_t = float(t)

    result = {
        "recommended_threshold": best_t,
        "balanced_accuracy": best_score,
        "known_mean": float(known_arr.mean()),
        "known_p10": float(np.percentile(known_arr, 10)),
        "known_p90": float(np.percentile(known_arr, 90)),
        "unknown_mean": float(unknown_arr.mean()),
        "unknown_p10": float(np.percentile(unknown_arr, 10)),
        "unknown_p90": float(np.percentile(unknown_arr, 90)),
        "n_known": int(known_arr.size),
        "n_unknown": int(unknown_arr.size),
    }
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate unknown threshold for writer ID")
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--dataset-root", type=str, required=True)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--target-len", type=int, default=96)
    p.add_argument("--out", type=str, default="writer_runs/threshold_report.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = calibrate(
        run_dir=Path(args.run_dir),
        dataset_root=Path(args.dataset_root),
        device_name=args.device,
        target_len=args.target_len,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"Saved report to {out_path}")


if __name__ == "__main__":
    main()
