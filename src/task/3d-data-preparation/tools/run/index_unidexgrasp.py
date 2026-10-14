#!/usr/bin/env python3
"""Reference existing actual trajectories/CAD without duplicating windows or assets."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    source=args.run.resolve();dest=args.output.resolve()
    if dest.exists():raise FileExistsError('fresh corpus directory required')
    old=json.loads((source/'trajectory-manifest.json').read_text())
    if old['profile']!='unidexgrasp-shadow-complete-cad-provisional' or old['training_allowed']:
        raise ValueError('audited provisional same-execution source required')
    objects=[];identities={}
    for i,row in enumerate(old['objects']):
        key=row['object_code']+'@'+str(row['scale'])
        identities[(row['object_code'],row['scale'])]=key
        path=source/'geometry'/('object-%03d.npz'%i)
        objects.append(dict(key=key,path=os.path.relpath(path,dest),mesh_id=row['object_code'],scale=row['scale']))
    sequences=[]
    for row in old['trajectories']:
        sequences.append(dict(path=os.path.relpath(source/row['path'],dest),format='unidexgrasp-compact',
            source_id=4,profile=old['profile'],group_id='unidexgrasp:'+row['object_code'],
            objects=[identities[(row['object_code'],row['scale'])]],target_slot=0,
            object_code=row['object_code'],scale=row['scale'],split=row['split'],
            source_episode_id=row['path'],official_success=row['official_success'],
            endpoint_within_5cm=row['endpoint_within_5cm']))
    meta=dict(schema='ref2dex.pose-mesh.v1',fps=30,units='m',history=4,horizon=24,object_points=32,
        training_allowed=False,objects=objects,sequences=sequences,source_manifest=str(source/'trajectory-manifest.json'),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_clock='actual60Hz frames selected every2; includes reset warm step',
        storage='read-only source references; no copied CAD/trajectory/window',
        limitations=['full-CAD ShadowHand palm-root profile','not Inspire action supervision',
                     'target distance labels are not stable grasp evidence'])
    dest.mkdir(parents=True)
    (dest/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(dict(objects=len(objects),trajectories=len(sequences),manifest_bytes=(dest/'manifest.json').stat().st_size)))


if __name__=='__main__':main()
