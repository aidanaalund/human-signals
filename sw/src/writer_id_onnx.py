from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch

from writer_id_torch import WriterIdNet, load_model_bundle, preprocess_segment


def export_writer_encoder_onnx(
    run_dir: Path,
    output_path: Path,
    target_len: int = 96,
    opset: int = 17,
    device: str = "cpu",
) -> Path:
    """Exports only the embedding encoder (B, 8, T) -> (B, embed_dim)."""
    model, _, _, writer_labels = load_model_bundle(run_dir=run_dir, device=device)
    model.eval()

    class EncoderWrapper(torch.nn.Module):
        def __init__(self, inner_model: WriterIdNet):
            super().__init__()
            self.encoder = inner_model.encoder

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.encoder(x)

    wrapper = EncoderWrapper(model)
    dummy = torch.randn(1, 8, target_len, dtype=torch.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy,
        str(output_path),
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={"input": {0: "batch", 2: "time"}, "embedding": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
    )

    meta = {
        "target_len": target_len,
        "num_writers": int(len(writer_labels)),
        "opset": int(opset),
        "onnx_file": str(output_path),
    }
    with (output_path.parent / "onnx_export_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return output_path


class ONNXWriterRegistry:
    """
    Writer registry using ONNX Runtime for embeddings and numpy for identity matching.
    """

    def __init__(
        self,
        onnx_path: Path,
        channel_mean: np.ndarray,
        channel_std: np.ndarray,
        target_len: int = 96,
        unknown_threshold: float = 0.72,
        providers: Optional[list] = None,
    ):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for ONNX inference. "
                "Install with: pip install onnxruntime"
            ) from exc

        if providers is None:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.target_len = target_len
        self.unknown_threshold = unknown_threshold
        self.channel_mean = channel_mean.astype(np.float32)
        self.channel_std = np.maximum(channel_std.astype(np.float32), 1e-6)
        self.prototypes: Dict[str, np.ndarray] = {}

    def _embed_one(self, segment: np.ndarray) -> np.ndarray:
        x = preprocess_segment(segment, target_len=self.target_len)
        x = (x - self.channel_mean) / self.channel_std
        x = x.T[np.newaxis, :, :].astype(np.float32)

        emb = self.session.run([self.output_name], {self.input_name: x})[0][0]
        emb = emb.astype(np.float32)
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export writer encoder to ONNX")
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--output", type=str, default="writer_runs/latest/writer_encoder.onnx")
    p.add_argument("--target-len", type=int, default=96)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = export_writer_encoder_onnx(
        run_dir=Path(args.run_dir),
        output_path=Path(args.output),
        target_len=args.target_len,
        opset=args.opset,
        device=args.device,
    )
    print(f"Exported ONNX encoder to {out}")


if __name__ == "__main__":
    main()
