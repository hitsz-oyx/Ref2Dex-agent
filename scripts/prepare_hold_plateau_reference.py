#!/usr/bin/env python3
"""Create one predetermined synthetic hold task from immutable reference labels."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from scripts.run_contact_response_probe import sha,REFERENCE
from scripts.audit_reference_hold_protocol import runs


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    output=args.output.resolve()
    if output.exists() or ROOT not in output.parents:raise ValueError('unique independent output required')
    manifest=json.loads((REFERENCE/'run_manifest.json').read_text())
    inherited=manifest.get('input_sha256',manifest.get('source_sha256',{}))
    sources=sorted(Path(p) for p in inherited if p.endswith('interaction_hand_inspire.pt'))
    if len(sources)!=3:raise ValueError('exact reference pool required')
    inputs={str(p):inherited[str(p)] for p in sources}
    paths=[Path(__file__).resolve(),ROOT/'scripts/audit_reference_hold_protocol.py',
        ROOT/'third_party/DExplore/dexplore/env/tasks/base_dexplore_task.py',
        ROOT/'docs/experiments/probes/P-20261001-hold-plateau-substrate.md']
    inputs.update({str(p.resolve()):sha(p) for p in paths})
    if any(sha(Path(p))!=value for p,value in inputs.items()):raise ValueError('input drift')
    output.mkdir(parents=True);begin=time.monotonic();records=[]
    for path in sources:
        raw=torch.load(path,map_location='cpu',weights_only=False)
        if raw.ndim!=2 or raw.shape[1]!=598:raise ValueError('raw schema drift')
        z=raw[:,200]-raw[0,200];start,stop=runs(z>=.03)[0]
        peak=start+int(z[start:stop].argmax());copies=raw[peak:peak+1].expand(90,-1).clone()
        generated=torch.cat((raw[:peak+1],copies,raw[peak+1:]))
        if not torch.equal(generated[:peak+1],raw[:peak+1]) or not torch.equal(generated[peak+91:],raw[peak+1:]):raise ValueError('remainder drift')
        if not torch.equal(copies[1:],copies[:-1]) or not torch.equal(generated[0],raw[0]):raise ValueError('nonstationary plateau/initial drift')
        folder=output/'references'/path.parent.name;folder.mkdir(parents=True);filename=folder/path.name
        torch.save(generated,filename)
        records.append(dict(name=path.parent.name,source=str(path),source_sha256=sha(path),generated=str(filename),generated_sha256=sha(filename),
            original_frames=len(raw),generated_frames=len(generated),first_lift_interval=[start,stop],peak_frame=peak,
            plateau_reference_frames_inclusive=[peak+1,peak+90],after_physics_trace_slice_half_open=[peak,peak+90],
            reference_hold_height_mm=float(z[peak]*1000),all598columns_stationary=True,initial_and_original_remainder_preserved=True))
    size=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
    if time.monotonic()-begin>60 or size>10*(1<<20):raise RuntimeError('label generation budget')
    if any(sha(Path(p))!=value for p,value in inputs.items()):raise ValueError('post-generation drift')
    result=dict(experiment_id='P-20261001-hold-plateau-substrate',stage='SYNTHETIC_REFERENCE_GENERATION',run_status='COMPLETED',
        synthetic=True,no_physics_or_model_training=True,input_sha256=inputs,references=records,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),wall_seconds=time.monotonic()-begin,
        data_bytes=size,requires_native_loader_velocity_recomputation=True)
    (output/'run_manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
