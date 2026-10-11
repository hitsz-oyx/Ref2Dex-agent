#!/usr/bin/env python3
"""Debug-only fitting check on the first fit clip; no dev evaluation/tuning."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / "src"))
from oakink_wm.pointwam_data import sha256
from oakink_wm.pointwam_learning import JointForecaster, batch, joint_loss, load_qualified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    result = dict(status="RUNNING", run_id=args.output.name, debug_seed=7, steps=600, lr=.001,
                  points=512, seconds_cap=120, dev_used=False, pid=os.getpid(),
                  cuda_visible_devices=os.getenv("CUDA_VISIBLE_DEVICES"),
                  git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  code_sha256=sha256(Path(__file__)), command=[sys.executable, *sys.argv])
    destination = args.output / "manifest.json"
    def save():
        destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    save()
    try:
        torch.set_num_threads(4); torch.manual_seed(7); torch.cuda.manual_seed_all(7)
        if not torch.cuda.is_available():
            raise RuntimeError("GPU required")
        previous = json.loads((args.run / "manifest.json").read_text())
        if previous["status"] != "COMPLETED" or sha256(Path(previous["qualification"])) != previous["qualification_sha256"]:
            raise ValueError("original qualification identity changed")
        clips = load_qualified(Path(previous["qualification"]))
        clip = next(c for c in clips if c["partition"] == "fit_candidate")
        contract = json.loads((args.run / "contract.json").read_text())
        if contract["fit_pack_hashes"] != [c["sha256"] for c in clips if c["partition"] == "fit_candidate"]:
            raise ValueError("fit normalization identity changed")
        result.update(pack=clip["path"], pack_sha256=clip["sha256"], source_episode=clip["episode"],
                      contract_sha256=sha256(args.run / "contract.json"))
        ids = np.random.default_rng(7).choice(clip["pack"]["scene_xyz_m"].shape[1], 512, replace=False)
        np.save(args.output / "fixed_initial_scene_ids.npy", ids)
        inputs, target = batch([clip], [ids], contract, "cuda:0")
        model = JointForecaster(len(contract["vocabulary"]) + 1).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
        def measure():
            with torch.no_grad():
                _, parts = joint_loss(model(**inputs), target)
            return {key: float(value) for key, value in parts.items()}
        result["initial"] = measure(); history = []
        for step in range(1, 601):
            if time.monotonic() - start > 120:
                raise TimeoutError("debug budget exhausted")
            optimizer.zero_grad(set_to_none=True)
            loss, _ = joint_loss(model(**inputs), target)
            loss.backward()
            if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise ValueError("non-finite fitting computation")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            if step % 100 == 0:
                row = dict(step=step, **measure()); history.append(row)
                print(json.dumps(row), flush=True)
        result.update(status="COMPLETED", initial_ids_only=True, final=measure(), history=history,
                      peak_gpu_allocated_mib=torch.cuda.max_memory_allocated() / 1024 ** 2)
        result["loss_ratio"] = {key: result["final"][key] / result["initial"][key] for key in ("scene", "hand")}
        result["interpretation"] = "engineering diagnosis only; no dev or method benefit claim"
    except Exception as error:
        result.update(status="FAILED", error=repr(error)); raise
    finally:
        result["elapsed_s"] = time.monotonic() - start
        save()


if __name__ == "__main__":
    main()
