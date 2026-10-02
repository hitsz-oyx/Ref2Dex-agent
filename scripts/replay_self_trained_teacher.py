"""Independent full expert forward on GPU from saved current native observations."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_contact_response_probe import sha
import torch

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--panel',type=int,required=True);a=p.parse_args();start=time.monotonic();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    root=a.directory.resolve();m=json.loads((root/'run_manifest.json').read_text());assert a.panel in (587,588) and m['experiment_id']=='P-20261002-self-trained-teacher-qualification'
    cp=Path(m['teacher_checkpoint']);assert sha(cp)==m['teacher_sha256'];c=torch.load(cp,map_location='cpu',weights_only=False);rms=c['running_mean_std'];state={(k[10:] if k.startswith('_orig_mod.') else k):v for k,v in c['model'].items()}
    assert c['epoch']==260 and c['frame']==163840 and not any('rnn' in k for k in state)
    teacher={k:v.cuda() for k,v in state.items() if 'actor_mlp' in k or k.startswith('a2c_network.mu.')};mean=rms['running_mean'].float().cuda();var=rms['running_var'].float().cuda()
    d=root/f's{a.panel}';r=json.loads((d/'results.json').read_text());assert sha(d/'trace.pt')==r['trace_sha256'];data=torch.load(d/'trace.pt',map_location='cpu',weights_only=False);x=data['teacher_observation'][1:].reshape(-1,1442);expected=data['teacher_action'][1:].reshape(-1,18);maximum=0.
    with torch.no_grad():
        for b in range(0,len(x),1024):
            z=((x[b:b+1024].cuda()-mean)/torch.sqrt(var+1e-5)).clamp(-5,5)
            for index in (0,2,4,6):
                key='a2c_network.actor_mlp.'+str(index);z=torch.nn.functional.linear(z,teacher[key+'.weight'],teacher[key+'.bias']).relu()
            z=torch.nn.functional.linear(z,teacher['a2c_network.mu.weight'],teacher['a2c_network.mu.bias']).clamp(-1,1);z[:,[7,9,11,13,16,17]]=0
            maximum=max(maximum,float((z.cpu()-expected[b:b+1024]).abs().max()))
    if maximum>1e-5:raise ValueError(('full independent expert action replay',maximum))
    result=dict(run_status='COMPLETED',source_checkpoint_sha256=sha(cp),rows=len(x),full_saved_native_obs_forward=True,legacy_external_rms_replayed=True,maximum_action_error=maximum,tolerance=1e-5,device='cuda',native1442_feature_derivation_not_independent=True,clock_scope='native audit checks preprogress/valid mask and independently reconstructs 70state context',wall_seconds=time.monotonic()-start)
    (d/'teacher_replay.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
