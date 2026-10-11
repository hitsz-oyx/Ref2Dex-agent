"""Recover approximate six palm offsets from an author's public robot preview.

This is quantized visualization data, not an official unquantized parameter
release. Use only the observed t0, never fit offsets to forecast future hands.
The author viewer explicitly documents robot-pose quaternions as wxyz.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / "src"))
from oakink_wm.pointwam_data import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    meta = json.loads(args.metadata.read_text()); binary = args.binary.read_bytes()

    def array(name):
        block = meta["blocks"][name]
        return np.frombuffer(binary, dtype=block["dtype"], offset=block["offset"],
                             count=int(np.prod(block["shape"]))).reshape(block["shape"])

    hand = array("hand_pos").astype(np.float64)/meta["quant"]
    poses = array("rob_pose")
    offsets, body_offsets = [], []
    for side, lo, sign in (("right", 0, 1), ("left", 10, -1)):
        body_id = next(i for i, body in enumerate(meta["robot_bodies"])
                       if body["name"] == "allegro_palm_" + side)
        pose = poses[0, body_id]
        rotation = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
        local = (hand[0, lo:lo+6] - pose[:3]) @ rotation
        # Allegro distal direction is -body-z; index side is +/-body-y.
        # Both bases are proper rotations: canonical normal is +/-body-x.
        basis = np.array([[0, 0, sign], [0, sign, 0], [-1, 0, 0]], dtype=float)
        body_offsets.append(local.tolist());offsets.append((local @ basis).tolist())
    result = dict(offsets_m=offsets, hand_order=["right", "left"], exact_author_offsets=False,
                  source="author PointWAM public microwave preview, t0 only",
                  source_metadata_url="https://chrockey.github.io/PointWAM/assets/3d/microwave.json",
                  source_binary_url="https://chrockey.github.io/PointWAM/assets/3d/microwave.bin",
                  source_metadata_sha256=sha256(args.metadata), source_binary_sha256=sha256(args.binary),
                  quantization_step_m=1/meta["quant"], body_local_offsets_m=body_offsets,
                  human_basis="origin=middle knuckle; x=wrist->middle; y=little->index orthogonalized; z=x cross y",
                  axis_alignment="pilot convention: distal and index directions aligned; requires visual review",
                  limitation="Quantized preview offsets and declared frame convention; not an official parameter table.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, indent=2)+"\n")
    print(args.output)


if __name__ == "__main__":
    main()
