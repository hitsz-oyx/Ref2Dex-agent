"""Frozen Q choice from short one-subject truths; no query terminal labels exist."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--models',type=Path,required=True);a=p.parse_args()
    import numpy as np,torch
    from src.task.CmResidual.oracle_features import descriptor,load,OracleQ,masked,ARMS
    from scripts.run_contact_response_probe import sha
    out=a.source;destination=out/'selection';destination.mkdir();cp=load(a.models);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    zero=descriptor(out/'baseline',out/'baseline',np.zeros((12,12),np.float32),include_labels=False)
    features=[];hashes={str(a.models):sha(a.models)}
    for target in range(12):
        rows=[zero['features'][target]]
        for index in range(1,8):
            name='e%02d_c%02d'%(target,index);path=out/'queries'/name
            packet=descriptor(path,out/'baseline',np.load(out/'requests'/(name+'.npy')),include_labels=False)
            if packet['labels'] is not None or len(load(path/'trace.pt')['object_root'])!=68:raise ValueError('short-only oracle; no terminal label')
            rows.append(packet['features'][target]);hashes[str(path/'trace.pt')]=sha(path/'trace.pt')
        features.append(np.stack(rows))
        print(json.dumps(dict(subject=target,short_options=8,prefix_exact=True)),flush=True)
    features=np.stack(features);torch.save(dict(features=features,motion=zero['motion'],no_terminal_query_labels=True),destination/'feature_packet.pt')
    x=torch.tensor(features.reshape(-1,cp['input_dim']),device='cuda');x=((x-cp['mean'].to('cuda'))/cp['std'].to('cuda')).clamp(-10,10);choices={};scores={}
    with torch.no_grad():
        for arm in ARMS:
            network=OracleQ(cp['input_dim']).to('cuda').eval();network.load_state_dict(cp['models'][arm]['model'])
            scores[arm]=network(masked(x,arm,cp['common_dim'],cp['effect_dim'])).sigmoid().cpu().numpy().reshape(12,8)
            choices[arm]=scores[arm].argmax(1).tolist();del network
    (destination/'choices.json').write_text(json.dumps(choices,indent=2)+'\n');torch.save(scores,destination/'scores.pt')
    result=dict(run_status='COMPLETED',new_optimizer_steps=0,inherited_optimizer_steps=6000,short_truth_only=True,no_terminal_query_labels=True,input_sha256=hashes,choices=choices)
    (destination/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(run_status='COMPLETED',choices=choices)),flush=True)

if __name__=='__main__':main()
