"""Compare the inherited per-row sampler with an equivalent batched gather."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(TASK / "src"))


def batched_tokens(packed, valid, token_count=256):
    import torch
    batch, points, dimensions = packed.shape
    counts = valid.sum(1)
    if (counts == 0).any():
        raise ValueError("scene encoder received no valid point")
    positions = torch.arange(points, device=packed.device)[None].expand(batch, -1)
    indices = positions.masked_fill(~valid, points).sort(dim=1).values
    offsets = torch.arange(token_count, device=packed.device)[None].expand(batch, -1)
    uniform = (offsets.float() * (counts - 1)[:, None] / max(token_count - 1, 1)).round().long()
    choice = torch.where(counts[:, None] > token_count, uniform,
                         offsets.minimum((counts - 1)[:, None]))
    selected = indices.gather(1, choice)
    tokens = packed.gather(1, selected[..., None].expand(-1, -1, dimensions))
    mask = offsets < counts[:, None]
    return torch.where(mask[..., None], tokens, 0.), mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import numpy as np
    import torch
    from ptv3_shared_wam.data import SharedWindows, shared_collate
    from ptv3_shared_wam.training import DEFAULT_DATA
    from goal_point_wam.model import SceneHistoryEncoder
    from oakink_wm.multisource import mixed_indices
    if args.output.exists() or ROOT not in args.output.resolve().parents:
        raise ValueError("fresh project profile output required")
    torch.set_num_threads(2)
    dataset = SharedWindows(DEFAULT_DATA, "train")
    seed = int(np.random.SeedSequence([100, 241, 0]).generate_state(1)[0])
    batch = shared_collate([dataset[int(i)] for i in mixed_indices(dataset.base, 128, seed)])
    valid = (batch["point_valid"] & (batch["features"][..., 14] < .5)).cuda()
    packed = torch.randn(*valid.shape, 131, device="cuda")
    old = lambda: SceneHistoryEncoder._sample_tokens(SimpleNamespace(token_count=256), packed, valid)
    new = lambda: batched_tokens(packed, valid)
    expected, actual = old(), new()
    if not all(torch.equal(a, b) for a, b in zip(expected, actual)):
        raise AssertionError("batched sampler changes selected point tokens")
    result = dict(pass_=True, scope="sampling only; no neural training", batch=128)
    for name, operation in (("per_row", old), ("batched", new)):
        operation(); torch.cuda.synchronize()
        started = time.perf_counter()
        for repeat in range(10):
            operation()
        torch.cuda.synchronize()
        result[name + "_ms"] = (time.perf_counter() - started) * 100
    result["speedup"] = result["per_row_ms"] / result["batched_ms"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
