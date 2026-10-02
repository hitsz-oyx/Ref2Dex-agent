"""Training-only fixed-history outcome and behavior-gradient energy diagnostic."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_contact_response_probe import sha
import torch

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();source=a.source.resolve();out=a.output.resolve();assert not out.exists() and ROOT in out.parents
    torch.set_num_threads(2);start=time.monotonic();old=json.loads((source/'run_manifest.json').read_text());assert old['run_status']=='COMPLETED'
    hashes={str(Path(__file__).resolve()):sha(Path(__file__)),str((ROOT/'docs/experiments/probes/P-20261002-task-outcome-determinacy.md').resolve()):sha(ROOT/'docs/experiments/probes/P-20261002-task-outcome-determinacy.md'),str((source/'run_manifest.json').resolve()):sha(source/'run_manifest.json')};details=[];maximum_packet_advantage_error=0.
    sums={name:dict(rows=0,determined=0,known_failed=0,known_success=0,proxy_energy=0.,determined_proxy_energy=0.,currently_valid_after_failure=0) for name in ('cm','state_only','none')}
    out.mkdir();manifest=dict(run_status='RUNNING',experiment_id='P-20261002-task-outcome-determinacy',source=str(source),input_sha256=hashes,cpu_reason='file/current-history label and arithmetic statistics; no neural forward/training',no_physics_or_model_updates=True)
    def save():manifest['wall_seconds']=time.monotonic()-start;(out/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    save()
    try:
        for seed in range(547,567):
            if time.monotonic()-start>180:raise TimeoutError('TRAIN-only diagnostic time budget')
            d=source/f's{seed}';r=json.loads((d/'results.json').read_text());audit=json.loads((d/'panel_audit.json').read_text());assert r['run_status']==audit['run_status']=='COMPLETED'
            files=[d/'initial.pt',d/'trace.pt',d/'rows.json',d/'results.json',d/'panel_audit.json',source/f'u{seed-546:02d}/update_packet.pt']
            record=old['panel_checkpoints'][str(seed)];cp_path=Path(record['path']);files.append(cp_path)
            for f in files:hashes[str(f.resolve())]=sha(f)
            assert sha(d/'initial.pt')==r['initial_sha256'] and sha(d/'trace.pt')==r['trace_sha256'] and sha(cp_path)==record['sha256']
            initial=torch.load(d/'initial.pt',map_location='cpu',weights_only=False);data=torch.load(d/'trace.pt',map_location='cpu',weights_only=False);rows=json.loads((d/'rows.json').read_text());cp=torch.load(cp_path,map_location='cpu',weights_only=False);packet=torch.load(files[-2],map_location='cpu',weights_only=False)
            # files[-2] is update packet; checkpoint is the final appended item.
            assert len(rows)==768 and [x['environment'] for x in rows]==list(range(768)) and all(x['seed']==seed for x in rows)
            progress=data['progress'];assert torch.equal(progress,torch.arange(1,203)[:,None].expand(202,768))
            stops=initial['phase_stop'][initial['motion']];window=(progress>=stops[None]-74)&(progress<=stops[None]+30);assert (window.sum(0)==105).all()
            valid=(data['object_root'][...,2]-initial['initial_height'][None]>=.03)&(data['clearance']>=.02);bad=window&~valid
            reward=torch.tensor([x['physical105'] for x in rows],dtype=torch.float32);assert torch.equal(~bad.any(0),reward.bool())
            post_failure=bad.cumsum(0)>0;pre_failure=torch.cat((torch.zeros_like(post_failure[:1]),post_failure[:-1]),0)
            finished=torch.arange(202)[:,None]>=stops[None]+30;pre_success=finished&~pre_failure;determined=pre_failure|pre_success
            assert not (pre_success&pre_failure).any()
            for group,name in enumerate(('cm','state_only','none'),1):
                ids=(initial['policy_group']==group).nonzero().flatten();assert len(ids)==192
                rec=packet['records'][name];ret=reward[ids][None].expand(202,192).reshape(-1);behavior=data['critic_value'][:,ids].reshape(-1)
                advantage=(ret-behavior-rec['advantage_mean'])/(rec['advantage_std']+1e-8)
                error=float((advantage[rec['indices']]-rec['advantage']).abs().max());maximum_packet_advantage_error=max(maximum_packet_advantage_error,error)
                if error>2e-5:raise ValueError('saved firstbatch normalizedadvantage mismatch')
                assert torch.equal(ret[rec['indices']],rec['return_target'])
                std=cp['variants'][name]['actor']['log_std'].clamp(-5,0).exp().double();noise=data['request_noise'][:,ids].reshape(-1,12).double();score=(noise/std).square().sum(-1)
                energy=advantage.double().square()*score;dead=determined[:,ids].reshape(-1);fail=pre_failure[:,ids].reshape(-1);success=pre_success[:,ids].reshape(-1)
                pre_valid=torch.cat((torch.zeros_like(valid[:1]),valid[:-1]),0)[:,ids].reshape(-1)
                row=dict(seed=seed,variant=name,rows=len(ret),determined=int(dead.sum()),known_failed=int(fail.sum()),known_success=int(success.sum()),proxy_energy=float(energy.sum()),determined_proxy_energy=float(energy[dead].sum()),currently_valid_after_failure=int((fail&pre_valid).sum()))
                details.append(row)
                for key in sums[name]:sums[name][key]+=row[key]
            save()
            del data,initial,packet,cp
        for name,s in sums.items():
            assert s['rows']==20*202*192
            s['determined_fraction']=s['determined']/s['rows'];s['determined_energy_fraction']=s['determined_proxy_energy']/s['proxy_energy'] if s['proxy_energy'] else 0.
        gate=any(s['determined_fraction']>=.5 or s['determined_energy_fraction']>=.5 for s in sums.values())
        for f,h in hashes.items():
            if sha(Path(f))!=h:raise ValueError('TRAIN source drift '+f)
        result=dict(run_status='COMPLETED',decision='PRIORITIZE_ABSORBING_TASK_STATE' if gate else 'AFTER_CERTAINTY_NOT_DOMINANT',variants=sums,panels=20,training_trajectories=15360,all_physical105_labels_rebuilt=True,terminal_reward_correct=True,first_minibatch_all_return_targets_verified=True,maximum_packet_normalized_advantage_error=maximum_packet_advantage_error,proxy_not_actual_ppo_or_adam_gradient=True,no_reward_corruption_or_mc_gradient_bias_claim=True,no_cm_utility_claim=True,details=details,source_sha256=hashes,wall_seconds=time.monotonic()-start)
        (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');manifest.update(run_status='COMPLETED',decision=result['decision'],bytes=sum(f.stat().st_size for f in out.iterdir() if f.is_file()),inputs_unchanged=True);print(json.dumps({k:v for k,v in result.items() if k not in ('details','source_sha256')}))
    except BaseException as e:manifest.update(run_status='FAILED',error=repr(e));raise
    finally:save()
if __name__=='__main__':main()
