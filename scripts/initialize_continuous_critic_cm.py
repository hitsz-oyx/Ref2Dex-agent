"""Tiny isolated CPU weight initialization; all real model computation is GPU."""
import argparse
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.task.CmResidual.continuous_critic_cm import initialized_models,SCHEMA
from src.task.CmResidual.paired_evaluation import fingerprint
from scripts.run_contact_response_probe import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve()
    if out.exists() or ROOT not in out.parents:raise ValueError('new independent own weight packet')
    variants={}
    for name in ('cm','state_only','none'):
        actor,critic=initialized_models()
        variants[name]=dict(actor=actor.state_dict(),critic=critic.state_dict(),actor_fingerprint=fingerprint(actor.state_dict()),critic_fingerprint=fingerprint(critic.state_dict()))
    assert len({p['actor_fingerprint'] for p in variants.values()})==1
    assert len({p['critic_fingerprint'] for p in variants.values()})==1
    out.mkdir()
    path=out/'policy_heads.pt';torch.save(dict(schema=SCHEMA,seed=762,updates=0,variants=variants,official_actor_weights_used=False),path)
    (out/'results.json').write_text(json.dumps(dict(run_status='COMPLETED',updates=0,checkpoint_sha256=sha(path),
        device_reason='CPU tiny deterministic weight initialization; no training or repeated model inference',schema=SCHEMA),indent=2)+'\n')
    print(sha(path))


if __name__=='__main__':main()
