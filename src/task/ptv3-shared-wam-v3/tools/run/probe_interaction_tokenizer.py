"""Train a small interaction codebook on cached WAMv3 future windows.

This is a bounded feasibility probe.  It deliberately does not modify the WAMv3
checkpoint or connect tokens to the robot policy.  Each sample is rebuilt from
the frozen processed sequence named in the multimodality cache metadata:
object-frame bilateral hand points, object pose motion, and a coarse contact
signal.  The output is a VQ reconstruction checkpoint and an auditable summary.
"""

from __future__ import annotations

import argparse
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch import nn


HORIZON = 24
HANDS = 2
KEYPOINTS = 11


def _rot9(rotation: np.ndarray) -> np.ndarray:
    return rotation.reshape(rotation.shape[:-2] + (9,))


@lru_cache(maxsize=256)
def _load_sequence(capture_string: str):
    capture = Path(capture_string)
    sequence_meta = json.loads((capture / "meta.json").read_text())
    poses = np.load(capture / "poses.npy", mmap_mode="r")
    hands = np.load(capture / "hand.npy", mmap_mode="r")
    near = np.load(capture / "near.npy", mmap_mode="r")
    return sequence_meta, poses, hands, near


def _sequence_feature(meta: Dict[str, object], horizon: int = HORIZON) -> np.ndarray:
    capture = Path(str(meta["capture"]))
    tick = int(meta["tick"])
    sequence_meta, poses, hands, near = _load_sequence(str(capture))
    object_index = sequence_meta["objects"].index(str(meta["geometry_key"]))
    if tick < 0 or tick + horizon >= len(hands):
        raise ValueError(f"window exceeds sequence: {capture} tick={tick}")
    frame_ids = np.arange(tick + 1, tick + horizon + 1)
    base = np.asarray(poses[tick, object_index], dtype=np.float32)
    base_inv = np.linalg.inv(base).astype(np.float32)
    future_pose = np.asarray(poses[frame_ids, object_index], dtype=np.float32)
    future_hand = np.asarray(hands[frame_ids], dtype=np.float32)
    world_h = np.concatenate(
        [future_hand, np.ones((horizon, HANDS, KEYPOINTS, 1), np.float32)], axis=-1
    )
    # Transform each hand point into the object frame at the corresponding frame.
    object_h = np.einsum(
        "tij,thkj->thki", np.linalg.inv(future_pose), world_h
    )[..., :3]
    relative_pose = np.einsum("ij,tjk->tik", base_inv, future_pose)
    translation = relative_pose[:, :3, 3]
    rotation = _rot9(relative_pose[:, :3, :3])
    contact_flag = np.asarray(near[frame_ids, object_index], dtype=np.float32)[:, None]
    min_distance = np.linalg.norm(object_h, axis=-1).min(axis=(1, 2))[:, None]
    result = np.concatenate(
        [object_h.reshape(horizon, -1), translation, rotation, contact_flag, min_distance],
        axis=-1,
    ).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"non-finite interaction feature: {capture} tick={tick}")
    return result


@lru_cache(maxsize=256)
def _cached_feature(meta_json: str) -> np.ndarray:
    return _sequence_feature(json.loads(meta_json))


def load_split(root: Path, split: str, limit: int, seed: int) -> np.ndarray:
    files = sorted(root.glob(f"{split}-*.npz"))
    if not files:
        raise FileNotFoundError(f"no {split}-*.npz under {root}")
    records: List[Tuple[str, np.ndarray]] = []
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            metadata = data["metadata"]
            count = min(len(metadata), max(1, limit // max(1, len(files)) + 8))
            file_seed = sum((index + 1) * byte for index, byte in enumerate(path.name.encode("utf-8")))
            rng = np.random.default_rng(np.random.SeedSequence([seed, file_seed]))
            selected = np.sort(rng.choice(len(metadata), count, replace=False))
            for index in selected:
                meta_json = str(metadata[int(index)])
                # The current multimodality cache also contains native payloads
                # whose capture is a geometry-only NPZ.  They do not carry a
                # time-indexed object pose, so they cannot satisfy this probe's
                # object-motion contract; keep the first probe on complete
                # processed sequences and report that scope explicitly.
                if not Path(json.loads(meta_json)["capture"]).is_dir():
                    continue
                records.append((meta_json, np.asarray(data["future"][int(index)])))
    if len(records) > limit:
        rng = np.random.default_rng(np.random.SeedSequence([seed, 991 if split == "train" else 997]))
        selected = np.sort(rng.choice(len(records), limit, replace=False))
        records = [records[int(i)] for i in selected]
    features = [_cached_feature(meta) for meta, _ in records]
    if not features:
        raise ValueError(f"empty {split} split")
    return np.stack(features, axis=0)


class InteractionVQVAE(nn.Module):
    def __init__(self, input_dim: int, codebook_size: int, latent_dim: int = 128) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(nn.Linear(input_dim, 256), nn.GELU(), nn.Linear(256, latent_dim))
        self.codebook = nn.Embedding(codebook_size, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 256), nn.GELU(), nn.Linear(256, input_dim))
        self.use_vq = True
        nn.init.normal_(self.codebook.weight, std=0.05)

    def forward(self, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.encoder(values)
        if not self.use_vq:
            indices = torch.zeros(values.shape[0], dtype=torch.long, device=values.device)
            return self.decoder(latent), indices, values.new_zeros(())
        distances = (
            latent.square().sum(-1, keepdim=True)
            - 2 * latent @ self.codebook.weight.t()
            + self.codebook.weight.square().sum(-1)[None]
        )
        indices = distances.argmin(-1)
        quantized = self.codebook(indices)
        straight_through = latent + (quantized - latent).detach()
        reconstruction = self.decoder(straight_through)
        codebook_loss = (quantized - latent.detach()).square().mean()
        commitment_loss = (latent - quantized.detach()).square().mean()
        return reconstruction, indices, codebook_loss + 0.25 * commitment_loss

    @torch.no_grad()
    def initialize_codebook(self, values: torch.Tensor) -> None:
        latent = self.encoder(values)
        if len(latent) < self.codebook_size:
            repeats = math.ceil(self.codebook_size / len(latent))
            latent = latent.repeat((repeats, 1))
        self.codebook.weight.copy_(latent[: self.codebook_size])


def _array_metrics(values: np.ndarray, recon: np.ndarray, indices: np.ndarray, codebook_size: int, mean: np.ndarray) -> Dict[str, float]:
    hand_dim = HANDS * KEYPOINTS * 3
    hand = np.linalg.norm(recon[..., :hand_dim] - values[..., :hand_dim], axis=-1).mean()
    motion = np.linalg.norm(recon[..., hand_dim : hand_dim + 3] - values[..., hand_dim : hand_dim + 3], axis=-1).mean()
    mean_recon = np.broadcast_to(mean, values.shape)
    mean_hand = np.linalg.norm(mean_recon[..., :hand_dim] - values[..., :hand_dim], axis=-1).mean()
    mean_motion = np.linalg.norm(mean_recon[..., hand_dim : hand_dim + 3] - values[..., hand_dim : hand_dim + 3], axis=-1).mean()
    counts = np.bincount(indices.reshape(-1), minlength=codebook_size).astype(np.float64)
    probabilities = counts / max(1.0, counts.sum())
    used = int((counts > 0).sum())
    perplexity = float(np.exp(-(probabilities[probabilities > 0] * np.log(probabilities[probabilities > 0])).sum()))
    return {
        "hand_epe_mm": float(hand * 1000.0),
        "object_translation_epe_mm": float(motion * 1000.0),
        "mean_baseline_hand_epe_mm": float(mean_hand * 1000.0),
        "mean_baseline_object_translation_epe_mm": float(mean_motion * 1000.0),
        "codebook_used": used,
        "codebook_perplexity": perplexity,
        "samples": int(len(values)),
    }


def _metrics(model: InteractionVQVAE, values: np.ndarray, mean: np.ndarray, std: np.ndarray, device: torch.device) -> Dict[str, float]:
    model.eval()
    with torch.inference_mode():
        x = torch.from_numpy(((values - mean) / std).astype(np.float32)).to(device)
        recon, indices, _ = model(x.reshape(-1, x.shape[-1]))
        recon = recon.reshape(x.shape).cpu().numpy() * std + mean
        indices = indices.reshape(x.shape[:-1]).cpu().numpy()
    return _array_metrics(values, recon, indices, model.codebook_size, mean)


def _kmeans(values: np.ndarray, codebook_size: int, steps: int, seed: int, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    flat = torch.from_numpy(values.reshape(-1, values.shape[-1]).astype(np.float32)).to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    initial = torch.randperm(len(flat), generator=generator, device=device)[:codebook_size]
    centers = flat[initial].clone()
    for _ in range(steps):
        distances = torch.cdist(flat, centers)
        indices = distances.argmin(-1)
        sums = torch.zeros_like(centers)
        sums.index_add_(0, indices, flat)
        counts = torch.bincount(indices, minlength=codebook_size).to(flat.dtype).unsqueeze(-1)
        centers = torch.where(counts > 0, sums / counts.clamp_min(1), centers)
    distances = torch.cdist(flat, centers)
    indices = distances.argmin(-1)
    return centers.cpu().numpy(), indices.cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=2048)
    parser.add_argument("--dev-limit", type=int, default=512)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--codebook-size", type=int, default=128)
    parser.add_argument("--mode", choices=("vq", "ae", "kmeans"), default="vq")
    parser.add_argument("--seed", type=int, default=271)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.train_limit < 64 or args.dev_limit < 32 or args.steps < 1:
        raise ValueError("limits and steps are too small for a probe")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train = load_split(args.input_root, "train", args.train_limit, args.seed)
    dev = load_split(args.input_root, "dev", args.dev_limit, args.seed)
    if train.shape[1] != HORIZON:
        raise ValueError(f"expected horizon {HORIZON}, got {train.shape}")
    chunk = 4
    if HORIZON % chunk:
        raise ValueError("horizon must be divisible by chunk")
    train = train.reshape(len(train), HORIZON // chunk, -1)
    dev = dev.reshape(len(dev), HORIZON // chunk, -1)
    mean = train.reshape(-1, train.shape[-1]).mean(0)
    std = train.reshape(-1, train.shape[-1]).std(0).clip(1e-4)
    input_dim = train.shape[-1]
    model = InteractionVQVAE(input_dim, args.codebook_size).to(device)
    model.use_vq = args.mode == "vq"
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    rng = np.random.default_rng(args.seed)
    flat_train = ((train - mean) / std).reshape(-1, input_dim).astype(np.float32)
    if args.mode == "kmeans":
        centers, train_indices = _kmeans(flat_train, args.codebook_size, args.steps, args.seed, device)
        dev_flat = ((dev - mean) / std).reshape(-1, input_dim).astype(np.float32)
        dev_tensor = torch.from_numpy(dev_flat).to(device)
        center_tensor = torch.from_numpy(centers).to(device)
        dev_indices = torch.cdist(dev_tensor, center_tensor).argmin(-1).cpu().numpy()
        train_recon = centers[train_indices].reshape(train.shape) * std + mean
        dev_recon = centers[dev_indices].reshape(dev.shape) * std + mean
        train_metrics = _array_metrics(train, train_recon, train_indices.reshape(train.shape[:-1]), args.codebook_size, mean)
        dev_metrics = _array_metrics(dev, dev_recon, dev_indices.reshape(dev.shape[:-1]), args.codebook_size, mean)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.mkdir()
        np.savez(args.output / "kmeans.npz", centers=centers, mean=mean, std=std,
                 input_dim=input_dim, chunk_frames=chunk, horizon=HORIZON, seed=args.seed)
        summary = {"status": "COMPLETED", "protocol": "main3-directory-sequences-reconstructed-bilateral-object-frame-v1",
                   "train": train_metrics, "dev": dev_metrics, "steps": args.steps, "batch_size": args.batch_size,
                   "codebook_size": args.codebook_size, "mode": args.mode, "input_dim": input_dim, "chunk_frames": chunk,
                   "feature_contract": "object-frame 2x11 hand points + relative object translation/rotation + near/contact proxy",
                   "device": str(device)}
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        return
    init_batch = torch.from_numpy(flat_train[: max(args.codebook_size, args.batch_size)]).to(device)
    if model.use_vq:
        model.initialize_codebook(init_batch)
    for step in range(args.steps):
        indices = rng.integers(0, len(flat_train), size=args.batch_size)
        batch = torch.from_numpy(flat_train[indices]).to(device)
        reconstruction, _, vq_loss = model(batch)
        loss = (reconstruction - batch).square().mean() + 0.25 * vq_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    train_metrics = _metrics(model, train, mean, std, device)
    dev_metrics = _metrics(model, dev, mean, std, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir()
    torch.save({"model": model.state_dict(), "input_dim": input_dim, "chunk_frames": chunk,
                "horizon": HORIZON, "mean": mean, "std": std, "seed": args.seed}, args.output / "tokenizer.pt")
    summary = {"status": "COMPLETED", "protocol": "main3-directory-sequences-reconstructed-bilateral-object-frame-v1",
               "train": train_metrics, "dev": dev_metrics, "steps": args.steps, "batch_size": args.batch_size,
               "codebook_size": args.codebook_size, "mode": args.mode, "input_dim": input_dim, "chunk_frames": chunk,
               "feature_contract": "object-frame 2x11 hand points + relative object translation/rotation + near/contact proxy",
               "device": str(device)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
