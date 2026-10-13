"""Real three-source tensors must match synchronous construction exactly."""
import argparse
import json
from pathlib import Path
import sys

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(TASK / "src"))

import torch
from ptv3_shared_wam.data import SharedWindows
from ptv3_shared_wam.loading import pretrain_loader
from ptv3_shared_wam.training import DEFAULT_DATA


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache", type=int, default=512)
    args = parser.parse_args()
    if args.output.exists() or ROOT not in args.output.resolve().parents:
        raise ValueError("fresh repository-local result required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    synchronous = pretrain_loader(SharedWindows(DEFAULT_DATA, "train"), 128, 100, 0, 241, 244)
    expected = list(synchronous)
    rng = torch.get_rng_state().clone()
    asynchronous = pretrain_loader(SharedWindows(DEFAULT_DATA, "train"), 128, 100, 0, 241, 244,
                                    workers=args.workers, sequence_cache=args.cache)
    compared = 0
    for reference, candidate in zip(expected, asynchronous):
        if reference.keys() != candidate.keys():
            raise AssertionError("prefetch changes batch fields")
        for key in reference:
            if not torch.equal(reference[key], candidate[key]):
                raise AssertionError("prefetch changes tensor: " + key)
            compared += 1
    if not torch.equal(rng, torch.get_rng_state()):
        raise AssertionError("prefetch consumes the model's CPU RNG")
    result = dict(pass_=True, batches=len(expected), tensors_compared=compared,
                  workers=args.workers, sequence_cache=args.cache,
                  contract="ordered bitwise-equal data and unchanged global torch RNG")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
