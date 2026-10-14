"""GPU regression for CPU-mapped AdamW checkpoint continuation, including Torch2.0."""
import argparse
import io
import json
import os
from pathlib import Path
import subprocess
import sys

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy", action="store_true", help="exercise the uncorrected Torch loader")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("fresh engineering report required")
    occupied = subprocess.check_output(["nvidia-smi", "-i", str(args.gpu),
        "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    if occupied:
        raise RuntimeError("audit GPU occupied")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    from ptv3_shared_wam_v2.checkpoint import atomic_json

    def optimizer_for(parameter):
        return torch.optim.AdamW([parameter], lr=.001, fused=True)

    parameter = torch.nn.Parameter(torch.tensor([1., 2.], device="cuda"))
    original = optimizer_for(parameter)
    parameter.square().sum().backward()
    original.step()
    archive = io.BytesIO()
    torch.save(original.state_dict(), archive)
    archive.seek(0)
    saved = torch.load(archive, map_location="cpu", weights_only=False)
    continued = torch.nn.Parameter(parameter.detach().clone())
    optimizer = optimizer_for(continued)
    if args.legacy:
        optimizer.load_state_dict(saved)
    else:
        from ptv3_shared_wam_v2.checkpoint import restore_optimizer
        restore_optimizer(optimizer, saved)
    report = dict(torch=torch.__version__, legacy=args.legacy,
                  step_device=str(optimizer.state[continued]["step"].device))
    try:
        continued.square().sum().backward()
        optimizer.step()
        report.update(status="PASS", step=int(optimizer.state[continued]["step"]),
                      parameters_updated=not torch.equal(continued, parameter))
        if report["step"] != 2 or not report["parameters_updated"]:
            raise ValueError("optimizer did not continue its saved state")
    except RuntimeError as error:
        report.update(status="FAIL", error=str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, report)
    print(json.dumps(report), flush=True)
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
