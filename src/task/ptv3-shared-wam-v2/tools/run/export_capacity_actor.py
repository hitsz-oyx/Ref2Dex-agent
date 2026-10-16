"""Bind a fixed capacity-probe head to its exact frozen WAM for native execution."""
import argparse
from pathlib import Path
import sys

TASK=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(TASK/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    import torch
    from ptv3_shared_wam_v2.capacity_actor import CapacityActor
    from ptv3_shared_wam_v2.robot_model import SharedWAMV2Robot
    saved=torch.load(a.run/'final.pt',map_location='cpu',weights_only=False)
    assert saved['schema']=='ref2dex.query-capacity-actor.v1' and saved['step']==6000
    config=saved['config'];actor=CapacityActor(config['width'],config['direct_goal'],config['bypass'])
    actor.load_state_dict(saved['model'],strict=True)
    packet=torch.load(saved['feature_checkpoint'],map_location='cpu',weights_only=False)
    packet={k:v for k,v in packet.items() if k not in ('optimizer','rank_rng','rng')}
    packet['config']=dict(packet['config'],current_observation_input='delta0',query_capacity_probe=dict(
        width=config['width'],direct_goal=config['direct_goal'],bypass=config['bypass']))
    weights={k:v for k,v in packet['model'].items() if k.startswith('forecaster.')}
    weights.update({'retargeter.'+k:v for k,v in actor.state_dict().items()})
    packet.update(model=weights,step=6000,capacity_probe_run=str(a.run.resolve()),
        capacity_probe_manifest=config,frozen_wam_source=saved['feature_checkpoint'])
    check=SharedWAMV2Robot(packet['stats'],packet['config'],packet['config']['action_representation'])
    check.load_state_dict(packet['model'],strict=True)
    a.output.parent.mkdir(parents=True,exist_ok=True);torch.save(packet,a.output)
    print(str(a.output),a.output.stat().st_size)


if __name__=='__main__':main()
