#!/usr/bin/env python3
"""Retain wrong-axis primary; audit corrected tabletop labels from saved poses."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.run_contact_response_probe import sha
from scripts.analyze_static_hold_feasibility import rotation,longest
from src.task.CmResidual.static_hold_feasibility import mesh_vertices
from src.task.CmResidual.tabletop_clearance import clearance,tabletop_geometry


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);args=p.parse_args();torch.set_num_threads(2)
    root=args.directory;m=json.loads((root/'run_manifest.json').read_text())
    if m['run_status']!='COMPLETED' or m['evaluation_seeds']!=[502,503]:raise ValueError('nonterminal/cohort drift')
    if any(sha(Path(p))!=h for p,h in m['input_sha256'].items()):raise ValueError('frozen input drift')
    out=root/'tabletop_axis_correction_r1';out.mkdir(exist_ok=False)
    assets=ROOT/'third_party/DExplore/dexplore/data/assets/mjcf'
    ov=mesh_vertices(assets/'objects/airplane/airplane.obj');tv=mesh_vertices(assets/'objects/table/table.obj')
    rows=[];maximum_error=0.;panel_geometry=[];all_corrected=[]
    for seed in (502,503):
        d=root/f's{seed}';r=json.loads((d/'results.json').read_text())
        for name in ('initial','trace'):
            if sha(d/(name+'.pt'))!=r[name+'_sha256']:raise ValueError('trace drift')
        initial=torch.load(d/'initial.pt',map_location='cpu',weights_only=False)
        data=torch.load(d/'trace.pt',map_location='cpu',weights_only=False)
        table=initial['table_root'];normal,top,axis=tabletop_geometry(tv,table)
        if axis!=1:raise ValueError('audited actual tabletop axis changed')
        corrected=torch.stack([clearance(obj,table,ov,tv) for obj in data['object_root']])
        # Independent NumPy WORLD-UP support because native table normal is
        # checked horizontal: min world object vertex z - max world table z.
        tbl=table.numpy();tverts=tv.numpy().astype(np.float64);verts=ov.numpy().astype(np.float64)
        table_z=(rotation(tbl[:,3:7])[:,2,:]@tverts.T+tbl[:,2,None]).max(-1)
        numpy_clearance=[]
        for object_root in data['object_root'].numpy():
            z=rotation(object_root[:,3:7])[:,2,:]@verts.T+object_root[:,2,None]
            numpy_clearance.append(z.min(-1)-table_z)
        numpy_clearance=np.stack(numpy_clearance)
        maximum_error=max(maximum_error,float(np.abs(numpy_clearance-corrected.numpy()).max()))
        if maximum_error>1e-5:raise ValueError('independent corrected world/table-local support mismatch')
        h=data['object_root'][:,:,2].numpy();rest=initial['rest_height'].numpy();start=initial['initial_height'].numpy()
        pair=data['contact'].numpy().astype(bool).all(-1)
        necessary=(h-rest[None]>=np.float32(.03))&(h-start[None]>=np.float32(-.01))&pair
        valid=necessary&(corrected.numpy()>=np.float32(.02))
        for env in range(96):
            base=longest(necessary[:,env]);held=longest(valid[:,env])
            rows.append(dict(seed=seed,environment=env,motion=int(initial['motion'][env]),
                necessary_height_proxy_max_steps=base,corrected_max_steps=held,retained75=held>=75))
        initial_clearance=clearance(initial['object_root'],table,ov,tv)
        panel_geometry.append(dict(seed=seed,thin_axis=axis,local_normals=normal.unique(dim=0).tolist(),top_support_values=top.unique().tolist(),
            corrected_initial_clearance_mm=[float(initial_clearance[initial['motion']==motion].mean()*1000) for motion in range(3)]))
        all_corrected.append(corrected)
    count=sum(r['retained75'] for r in rows);necessary=sum(r['necessary_height_proxy_max_steps']>=75 for r in rows)
    torch.save(dict(rows=rows,corrected_clearance_m=torch.stack(all_corrected)),out/'corrected.pt')
    result=dict(run_status='COMPLETED',status='POSTHOC_GEOMETRY_CORRECTION',original_clearance_status='INVALID_TABLETOP_AXIS',
        original_primary_gate_results_retained=True,no_new_physics=True,trajectories=192,
        necessary_height_proxy75_count=necessary,corrected_retained75_count=count,
        independent_clearance_max_abs_error_m=maximum_error,geometry=panel_geometry,
        original_manifest_sha256=sha(root/'run_manifest.json'),
        source_sha256={str(p):sha(p) for p in (Path(__file__).resolve(),ROOT/'src/task/CmResidual/tabletop_clearance.py')},
        boundary='wrong-axis full criterion invalid; necessary height/proxy condition fails regardless of clearance; corrected labels are reused-data diagnostics, not pristine Validation')
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
