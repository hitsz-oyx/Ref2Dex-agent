"""Bounded GPU tau-only geometric retarget; no later-q or object-label fitting."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path[:0] = [str(TASK / "src"), str(ROOT), str(TASK / "tools/run")]
from probe_reference_tracking import gpu_state, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    output = args.output.resolve(); output.relative_to(ROOT / "outputs/consequence-evaluator")
    if output.exists():
        raise ValueError("fresh task-owned output required")
    before = gpu_state(args.gpu)
    if before["used_mib"] > 512 or before["utilization"] > 10 or before["total_mib"] - before["used_mib"] < 20000:
        raise RuntimeError("GPU busy: " + repr(before))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    from consequence_evaluator.tau_tracking import fit_geometry, tau_calibration
    torch.set_num_threads(2)
    with args.reference.open("rb") as stream:
        packet = pickle.load(stream)
    # Materialize only future hand and the initially observed robot q.
    hand, reset_q = tau_calibration(packet)
    del packet
    urdf = ROOT / "third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf"
    output.mkdir(parents=True)
    started = time.monotonic()
    manifest = dict(schema="ref2dex.tau-geometry-reference.v1", status="RUNNING", gpu=args.gpu,
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    gpu_before=before, inputs="tau hand points, initial live q, static URDF; no later-q/object/command labels",
                    hand_sha256=hashlib.sha256(hand.tobytes()).hexdigest(),
                    reference_sha256=sha(args.reference), urdf_sha256=sha(urdf), iterations=300,
                    input_sha256={str(path.resolve()): sha(path) for path in (
                        args.reference, urdf, Path(__file__), TASK / "src/consequence_evaluator/tau_tracking.py",
                        TASK / "src/consequence_evaluator/reset_kinematics.py")})
    def write():
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write()
    try:
        result = fit_geometry(hand, reset_q, urdf, "cuda:0")
        np.savez_compressed(output / "geometry.npz", **result)
        error = result["fitted_points"] - hand
        manifest.update(status="COMPLETED", elapsed_s=time.monotonic() - started,
                        coordinate_rmse_mm=float(np.sqrt(np.mean(error**2)) * 1000),
                        per_point_rmse_mm=np.sqrt(np.mean(np.sum(error**2, -1), 0)).tolist(),
                        geometry_sha256=sha(output / "geometry.npz"), gpu_after=gpu_state(args.gpu))
        manifest["per_point_rmse_mm"] = [v * 1000 for v in manifest["per_point_rmse_mm"]]
        write(); print(json.dumps(manifest, indent=2), flush=True)
    except BaseException as error:
        manifest.update(status="FAILED", error=repr(error), elapsed_s=time.monotonic() - started)
        write(); raise


if __name__ == "__main__":
    main()
