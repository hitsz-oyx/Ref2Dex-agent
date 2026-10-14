"""Compare continuous versus partial/resumed engineering runs on fixed dev inputs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch

TASK=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(TASK / "src"))
from ptv3_shared_wam_v2.paths import ROOT
from ptv3_shared_wam_v2.model import SharedWAMV2
from ptv3_shared_wam_v2.checkpoint import load_checkpoint, atomic_json
from ptv3_shared_wam.data import SharedWindows
from ptv3_shared_wam.loading import forecasting_collate, to_device


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--continuous",type=Path,required=True)
    p.add_argument("--resumed",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--gpu",type=int,required=True)
    args=p.parse_args()
    output=args.output.resolve()
    if ROOT / "outputs" / TASK.name not in output.parents or output.exists():
        raise ValueError("fresh owned report required")
    if subprocess.check_output(["nvidia-smi","-i",str(args.gpu),"--query-compute-apps=pid","--format=csv,noheader"],text=True).strip():
        raise RuntimeError("audit GPU occupied")
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    torch.set_num_threads(2)
    a,b=load_checkpoint(args.continuous),load_checkpoint(args.resumed)
    if a["config"]!=b["config"] or a["stats"]!=b["stats"] or a["step"]!=b["step"]:
        raise ValueError("different compared contracts")
    if not a["provenance"]["engineering_only"] or not b["provenance"]["engineering_only"]:
        raise ValueError("this audit compares engineering runs")
    rng_equal=all(torch.equal(a["rank_rng"][0][key],b["rank_rng"][0][key]) for key in ("torch","cuda"))
    maximum=max(float((a["model"][k]-b["model"][k]).abs().max()) for k in a["model"])
    ids_a=np.load(args.continuous.parent/"validation.npy")
    np.testing.assert_array_equal(ids_a,np.load(args.resumed.parent/"validation.npy"))
    dataset=SharedWindows(Path(a["provenance"]["data"]),"val",a["config"]["object_points"])
    batch=to_device(forecasting_collate([dataset[int(i)] for i in ids_a]),"cuda")
    m=SharedWAMV2(a["stats"],a["config"]).cuda().eval()
    with torch.no_grad():
        m.load_state_dict(a["model"],strict=True); pa=m(batch)
        m.load_state_dict(b["model"],strict=True); pb=m(batch)
    report=dict(rng_equal=rng_equal,max_parameter_difference=maximum,
        max_hand_difference_mm=float((pa["trajectory"]-pb["trajectory"]).abs().max())*1000,
        max_scene_difference_mm=float((pa["scene"]-pb["scene"]).abs().max())*1000,
        scope="engineering continuity; sparse GPU operations need not be bitwise deterministic")
    report["pass"]=rng_equal and maximum<.001 and max(report["max_hand_difference_mm"],report["max_scene_difference_mm"])<.1
    output.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(output,report)
    print(json.dumps(report),flush=True)
    if not report["pass"]:raise SystemExit(1)


if __name__=="__main__":main()
