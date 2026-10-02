"""Fixed models on reused actual/learned successors; no learning or deployment."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.run_contact_response_probe import sha
from src.task.CmResidual.option_model_policy import read_options,state_features,initialized_network,scores


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args();root=a.directory.resolve();begin=time.monotonic();torch.set_num_threads(2);assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    m=json.loads((root/'run_manifest.json').read_text());assert m['experiment_id']=='P-20261002-successor-value-compatibility'
    cp=torch.load(m['source_models'],map_location='cpu',weights_only=False);base=torch.load(m['base_checkpoint'],map_location='cpu',weights_only=False)
    data=read_options(Path(m['source_run'])/'s618',base);assert len(data['reward'])==768
    current=state_features(data['current'],data['extra'],cp['extra_mean'],cp['extra_std']);future=state_features(data['future'],data['future_extra'],cp['extra_mean'],cp['extra_std'])
    gpu=lambda v:torch.from_numpy(v).cuda();x=gpu(current);xf=gpu(future);raw=gpu(data['raw']);known=gpu(data['known'])
    models={name:initialized_network('dynamics' if name in ('cm','dynamics_off') else 'value').cuda().eval() for name in cp['models']}
    for name,model in models.items():model.load_state_dict(cp['models'][name])
    with torch.no_grad():
        predictions={name:scores(x,raw,known,models,gpu(cp['delta_mean']),gpu(cp['delta_std']),name).cpu().numpy() for name in ('cm','dynamics_off','direct_q')}
        predictions['oracle']=models['value'](torch.cat((xf,torch.tanh(raw)),-1)).flatten().cpu().numpy()
    assert all(np.isfinite(v).all() and ((v>=0)&(v<=1)).all() for v in predictions.values())
    losses={name:(pred.astype(np.float64)-data['reward'])**2 for name,pred in predictions.items()};brier={name:float(loss.mean()) for name,loss in losses.items()}
    generator=torch.Generator(device='cpu').manual_seed(3558);indices=np.concatenate([np.flatnonzero(data['motion']==j)[torch.randint(256,(1000,256),generator=generator).numpy()] for j in range(3)],1)
    intervals={name:np.quantile(delta[indices].mean(-1),(.025,.975)).tolist() for name,delta in {'oracle_minus_direct_q':losses['oracle']-losses['direct_q'],'cm_minus_oracle':losses['cm']-losses['oracle']}.items()}
    gates=dict(oracle_gain10percent=brier['oracle']<=.9*brier['direct_q'],oracle_interval_upper_negative=intervals['oracle_minus_direct_q'][1]<0,cm_excess10percent_over_oracle=brier['cm']>=1.1*brier['oracle'],cm_minus_oracle_interval_lower_positive=intervals['cm_minus_oracle'][0]>0)
    useful=gates['oracle_gain10percent'] and gates['oracle_interval_upper_negative'];gap=gates['cm_excess10percent_over_oracle'] and gates['cm_minus_oracle_interval_lower_positive']
    label='UNPROMISING' if not useful else ('PROMISING' if gap else 'UNCLEAR')
    torch.save(dict(data=data,current=current,future=future,predictions=predictions,bootstrap_indices=indices),root/'predictions.pt')
    result=dict(run_status='COMPLETED',label=label,brier=brier,paired_bootstrap_intervals=intervals,gates=gates,oracle_useful=useful,mean_gap_present=gap,episodes=768,reused_dataset=True,new_native_trajectories=0,new_optimizer_steps=0,actual_future_only_oracle_diagnosis=True,no_jensen_or_conditional_mean_claim=True,no_policy_utility_claim=True,formal_validation=False,predictions_sha256=sha(root/'predictions.pt'),per_motion_brier={str(j):{name:float(loss[data['motion']==j].mean()) for name,loss in losses.items()} for j in range(3)},per_arm_brier={str(j):{name:float(loss[data['arm']==j].mean()) for name,loss in losses.items()} for j in range(4)},wall_seconds=time.monotonic()-begin)
    (root/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
