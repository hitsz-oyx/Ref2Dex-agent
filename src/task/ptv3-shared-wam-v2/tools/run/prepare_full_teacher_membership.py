"""Export exact existing native-current workspace-qualified membership, on CPU."""
import argparse
import json
from pathlib import Path
import sys
TASK=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(TASK/'src'))
from ptv3_shared_wam_v2.robot_data import V2RLRobotWindows
from ptv3_shared_wam.action_targets import GEOMETRY_LIFTED
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--initial-R',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    config=torch.load(a.initial_R,map_location='cpu',weights_only=False)['config']
    workspace=dict(schema=config['encoder_workspace_admission'],origin_m=config['voxel_origin_m'],voxel_m=config['voxel_m'])
    datasets={s:V2RLRobotWindows(a.source,s,32,GEOMETRY_LIFTED,workspace) for s in ['train','val']}
    result=dict(source=str(a.source.resolve()),workspace=workspace,splits={s:dict(records=d.records,ticks=[x.tolist() for x in d.ticks_by_sequence],windows=len(d)) for s,d in datasets.items()})
    a.output.write_text(json.dumps(result)+'\n');print({s:len(d) for s,d in datasets.items()})


if __name__=='__main__':main()
