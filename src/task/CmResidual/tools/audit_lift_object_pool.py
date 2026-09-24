"""Read-only audit of retargeted S1 lift motion availability by object."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SOURCE = ROOT / "data/processed_data/inspire_rl_partial_filtered_20260905"
RAW_OBJECTS = ROOT / "data/raw_data/GRAB/objects"


def scan(path: Path) -> dict:
    object_name = path.parent.name[3:-5]
    row = {"sequence": path.parent.name, "object": object_name,
           "mesh_available": (RAW_OBJECTS / object_name / "mesh.obj").is_file()}
    try:
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2 or tensor.shape[1] != 598:
            raise ValueError(f"expected [T,598], got {getattr(tensor, 'shape', None)}")
        if not torch.isfinite(tensor).all():
            raise FloatingPointError("non-finite motion")
        contact = tensor[:, 205] >= .5
        separation = (tensor[:, 51:54] - tensor[:, 198:201]).norm(dim=-1)
        median_distance = float(separation[contact].median()) if contact.any() else None
        lift = float(tensor[:, 200].max() - tensor[0, 200])
        occupancy = float(contact.float().mean())
        row.update({"frames": len(tensor), "reference_lift_m": lift,
                    "contact_occupancy": occupancy,
                    "median_contact_wrist_object_distance_m": median_distance,
                    "passes_offline_gate": bool(row["mesh_available"] and lift >= .03 and
                        occupancy >= .05 and median_distance is not None and median_distance <= .25)})
    except (ValueError, RuntimeError, FloatingPointError) as error:
        row.update({"passes_offline_gate": False, "error": f"{type(error).__name__}: {error}"})
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("new output required")
    paths = sorted(args.source.glob("s1_*_lift/interaction_hand_inspire.pt"))
    rows = [scan(path) for path in paths]
    eligible = [row["object"] for row in rows if row["passes_offline_gate"]]
    additional = sorted(set(eligible) - {"airplane", "mug", "toothpaste", "apple"})
    report = {"schema": "ref2dex.lift_object_pool_audit.v1", "classification": "Probe",
              "source": str(args.source.resolve()), "rows": rows,
              "summary": {"source_sequences": len(rows), "eligible_objects": eligible,
                          "additional_unseen_object_count": len(additional),
                          "additional_unseen_objects": additional,
                          "expanded_split_data_gate_passed": len(additional) >= 8},
              "limits": ["Reference lift/contact do not establish simulator feasibility.",
                         "Penetration and coordinate alignment are unverified."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
