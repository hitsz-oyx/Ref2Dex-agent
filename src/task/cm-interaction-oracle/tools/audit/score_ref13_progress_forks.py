"""Score matched physical forks with old U and frozen causal delta-progress."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[5]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--folders',nargs=7,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',type=int,required=True)
    args=parser.parse_args(); root=args.run_dir.resolve()
    root.relative_to(ROOT/'outputs/cm-interaction-oracle')
    args.output.resolve().relative_to(root)
    if args.output.exists() or args.output.with_suffix('.pt').exists():
        raise FileExistsError(args.output)
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    import numpy as np
    import torch
    sys.path[:0]=[str(ROOT/'src/task/cm-interaction-oracle/src'),str(ROOT/'src/task/consequence-evaluator/src')]
    from rolling_control import candidate_scores,mixed_plan
    from consequence_evaluator.reference_bank import load_reference_features
    from consequence_evaluator.reference_progress import trajectory_features
    from consequence_evaluator.temporal_phase import LearnedReferenceProgress
    schedule=json.loads((root/'s3-schedule.json').read_text())
    rows=torch.tensor([i for i,g in enumerate(schedule['groups']) if g==0])
    folders=[root/name for name in args.folders]
    paths=[p/name for p in folders for name in ('panel.pt','trace.pt')]
    hashes={str(p):sha(p) for p in paths}
    panels=[torch.load(p/'panel.pt',map_location='cpu',weights_only=False) for p in folders]
    old_y,old=candidate_scores(panels,rows)
    if (panels[0]['motion_id'][rows]!=0).any():
        raise ValueError('s3-only physical reference bank')
    query=int(panels[0]['triggers'][rows][0])
    bank=ROOT/'outputs/consequence-evaluator/physical-reference-bank-20261008-r1'
    encoder=ROOT/'outputs/consequence-evaluator/physical-bank-tcc-20261008-r1'
    meta=json.loads((bank/'manifest.json').read_text())
    trained=json.loads((encoder/'manifest.json').read_text()); checkpoint=Path(trained['checkpoint'])
    if (trained['status']!='COMPLETED' or trained['reference_sha256']!=meta['reference_sha256']
            or sha(checkpoint)!=trained['checkpoint_sha256']):
        raise ValueError('frozen bank/encoder mismatch')
    refs,_=load_reference_features(bank,meta)
    matcher=LearnedReferenceProgress(refs,torch.load(checkpoint,map_location='cpu',weights_only=False),'cuda:0')
    values=np.empty((len(rows),7)); starts=np.empty_like(values)
    first_geometry=None
    for k,folder in enumerate(folders):
        trace=torch.load(folder/'trace.pt',map_location='cpu',weights_only=False)
        geometry=trace['progress_geometry']
        if first_geometry is None:
            first_geometry=geometry
        for key in ('object_pose','hand_keypoints'):
            if not torch.equal(geometry[key][:query+1,rows],first_geometry[key][:query+1,rows]):
                raise ValueError('progress geometry prefix differs before candidate')
        for j,row in enumerate(rows.tolist()):
            features=trajectory_features(geometry['object_pose'][:query+25,row],
                geometry['hand_keypoints'][:query+25,row],geometry['timestamps'][:query+25])
            progress=matcher.align(features)['progress']
            starts[j,k]=float(progress[query]); values[j,k]=float(progress[query+24]-progress[query])
        print(json.dumps(dict(candidate=k,rows=len(rows),mean_delta_progress=float(values[:,k].mean()))),flush=True)
    if not np.isfinite(values).all() or not np.allclose(starts,starts[:,:1],rtol=0,atol=1e-12):
        raise ValueError('causal start progress mismatch')
    new=torch.from_numpy(values)
    old_plan=mixed_plan(old,rows,96); new_plan=mixed_plan(new,rows,96)
    result=dict(scope='matched same-state GT scoring Probe; not executed policy benefit',query=query,
        rows=rows.tolist(),old_scores=old.tolist(),new_scores=values.tolist(),start_progress=starts.tolist(),
        old_choices=old_plan[rows].tolist(),new_choices=new_plan[rows].tolist(),
        differing_choices=int((old_plan[rows]!=new_plan[rows]).sum()),
        old_selection_counts=torch.bincount(old_plan[rows],minlength=7).tolist(),
        new_selection_counts=torch.bincount(new_plan[rows],minlength=7).tolist(),
        new_label='P[t+24]-P[t]; causal prefix truncated at t+24; fixed eight-member train230 bank',
        input_sha256={**hashes,**{str(p):sha(p) for p in (bank/'manifest.json',bank/'reference.npz',encoder/'manifest.json',checkpoint,Path(__file__).resolve())}})
    torch.save(dict(rows=rows,old_y=old_y,old_scores=old,new_scores=new,
                    old_plan=old_plan,new_plan=new_plan),args.output.with_suffix('.pt'))
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('query','differing_choices','old_selection_counts','new_selection_counts')}),flush=True)


if __name__=='__main__':
    main()
