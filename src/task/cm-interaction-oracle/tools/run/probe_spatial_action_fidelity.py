#!/usr/bin/env python3
"""Frozen nominal/forecast spatial layers: decode known physical finger motion."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[5];TASK=ROOT/'src/task/cm-interaction-oracle'
sys.path[:0]=[str(ROOT),str(TASK/'src'),str(TASK/'tools/run'),str(TASK/'tools/audit')]
from spatial_action_fidelity import intrinsic_finger_flow,current_finger_support,extract_stages,fit_decoder,replay_decoder
from spatial_consequence import SpatialConsequence,SEED
from geometric_consequence import NominalSurfaceActions
from execution_geometry import endpoint_flows
from probe_execution_geometry import source_state,to_device
from probe_geometric_consequence import cpu_tree
from probe_interventions import sha,tensor_hash
from consequence_sufficiency import cluster_gain


def metrics(prediction,target,test,clusters,support):
    error=(prediction[:,1:]-target[:,1:]).square()
    baseline=target[:,1:].square()
    e=error[test];b=baseline[test];t=target[test,1:]
    signed=t.abs()>=.02
    mask=support[test,None].expand_as(e)
    denominator=float((b*mask).sum())
    mse=float(e.mean());zero=float(b.mean())
    return dict(mse=mse,zero_mse=zero,gain_vs_zero=1-mse/zero,
        bootstrap=cluster_gain(b.mean((1,2)).cpu().numpy(),e.mean((1,2)).cpu().numpy(),clusters,249),
        per_axis_mse=e.mean((0,1)).cpu().tolist(),per_arm_mse=e.mean((0,2)).cpu().tolist(),
        signed_agreement=float((torch.sign(prediction[test,1:])[signed]==torch.sign(t)[signed]).float().mean()),
        signed_entries=int(signed.sum()),current_support_gain=1-float((e*mask).sum())/denominator if denominator>1e-12 else None,
        current_support_per_axis_fraction=support[test].float().mean(0).cpu().tolist())


def decision(values):
    raw=values['RawObject']['gain_vs_zero'];intrinsic=values['RawHandBase']['gain_vs_zero']
    local=values['Trained_LocalFlow']['gain_vs_zero'];fused=values['Trained_Fused']['gain_vs_zero']
    if raw<.8 and intrinsic>=.8 and intrinsic-raw>=.1:return 'INTRINSIC_FRAME', 'PROMISING'
    if raw>=.8 and local<.5:return 'LOCAL_SELECTION', 'UNPROMISING'
    if raw>=.8 and local>=.8 and fused<.5 and raw-fused>=.2:return 'SPATIAL_COMPRESSION', 'UNPROMISING'
    if fused>=.8 and raw>=.8 and fused/raw>=.8:return 'RETAINED_MOTION_CHECK_CONSEQUENCE_OBJECTIVE', 'PROMISING'
    return 'INSUFFICIENT_LINEAR_DISCRIMINATION', 'UNCLEAR'


def load_inputs(args,device):
    fm=json.loads((args.forecast_run/'manifest.json').read_text())
    assert fm['run_status']=='COMPLETED' and not fm['smoke']
    assert fm['projection']=='bounded_only_preserve_continuous'
    assert all(sha(ROOT/k)==v for k,v in fm['code_sha256'].items())
    assert sha(args.dataset)==fm['dataset_sha256']
    forecast=torch.load(args.forecast_run/'diagnostic.pt',map_location='cpu',weights_only=False)
    original_run=Path(forecast['original_run']);source_run=Path(forecast['source_run'])
    assert sha(original_run/'diagnostic.pt')==fm['original_diagnostic_sha256']
    assert sha(source_run/'diagnostic.pt')==fm['source_diagnostic_sha256']
    sm=json.loads((source_run/'manifest.json').read_text())
    assert sm['run_status']=='COMPLETED' and not sm['smoke']
    assert all(sha(ROOT/k)==v for k,v in sm['code_sha256'].items())
    assert all(sha(Path(k))==v for k,v in sm['hand_visual_sha256'].items())
    source=torch.load(source_run/'diagnostic.pt',map_location='cpu',weights_only=False)
    original=torch.load(original_run/'diagnostic.pt',map_location='cpu',weights_only=False)
    p=torch.load(args.dataset,map_location='cpu',weights_only=False)
    bridge=NominalSurfaceActions(ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf',device)
    g=bridge.build(p)
    assert all(torch.equal(g[k].cpu(),original['source_geometry'][k]) for k in ('points','normals','nominal_flow','targets','joint','geometry'))
    g['obj_points']=source['geometry']['obj_points'].to(device);g['obj_normals']=source['geometry']['obj_normals'].to(device)
    h=source_state(p,source,g,'full',device)
    assert all(np.array_equal(source[k],forecast[k]) for k in ('train','test','clusters','arms'))
    assert not set(source['clusters'][source['train']])&set(source['clusters'][source['test']])
    return p,bridge,g,h,source,forecast,source_run,original_run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--forecast-run',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();args.run_dir.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();torch.set_num_threads(2)
    def save(name,value):(args.run_dir/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    paths=[Path(__file__).resolve(),TASK/'src/spatial_action_fidelity.py',TASK/'src/spatial_consequence.py',
        TASK/'src/geometric_consequence.py',TASK/'src/execution_geometry.py',TASK/'src/execution_projection.py',
        TASK/'tools/run/probe_execution_geometry.py',TASK/'tools/run/probe_conditional_consequence.py',
        TASK/'tools/run/probe_geometric_consequence.py',TASK/'tools/run/probe_interventions.py',
        TASK/'src/consequence_sufficiency.py',ROOT/'src/task/ObjectInteractionCmv2/model.py']
    manifest=dict(run_status='STARTED',git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        created_at=datetime.now(timezone.utc).isoformat(),physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        seeds=[245,249],smoke=args.smoke,command=sys.argv,dataset_sha256=sha(args.dataset),
        forecast_manifest_sha256=sha(args.forecast_run/'manifest.json'),forecast_diagnostic_sha256=sha(args.forecast_run/'diagnostic.pt'),
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},new_consequence_fits=0,new_policy_fits=0)
    save('manifest.json',manifest)
    try:
        dev=torch.device('cuda:0');torch.cuda.set_device(dev);torch.cuda.reset_peak_memory_stats(dev)
        p,bridge,g,h,source,forecast,source_run,original_run=load_inputs(args,dev)
        manifest.update(source_run=str(source_run),original_run=str(original_run),
            source_diagnostic_sha256=sha(source_run/'diagnostic.pt'),original_diagnostic_sha256=sha(original_run/'diagnostic.pt'))
        train=source['train'][:8] if args.smoke else source['train']
        test=source['test'][:4] if args.smoke else source['test']
        ids=torch.as_tensor(np.concatenate((train,test)),device=dev)
        fit=torch.arange(len(train),device=dev);hold=torch.arange(len(train),len(ids),device=dev)
        clusters=source['clusters'][test]
        support=current_finger_support(g)[ids]
        source_arms=torch.as_tensor(source['arms'],device=dev)
        flows={'Nominal':g['nominal_flow']}
        qs={'Nominal':g['targets'],'Forecast':forecast['oof_q']['Ha'].to(dev)}
        flows['Forecast']=endpoint_flows(p,bridge,g,qs['Forecast'])[0]
        targets={name:(q[ids,:,6:]-q[ids,:1,6:])/.32 for name,q in qs.items()}
        stats,decoder_states,predictions,hashes,engineering={}, {}, {}, {}, {}
        feature_rms={}
        for input_name in ('Nominal','Forecast'):
            stats[input_name]={};predictions[input_name]={};decoder_states[input_name]={};hashes[input_name]={};feature_rms[input_name]={}
            raw={'RawObject':(flows[input_name][ids]-flows[input_name][ids,:1]).flatten(2)/.02,
                 'RawHandBase':intrinsic_finger_flow(p,bridge,g,qs[input_name])[ids].flatten(2)/.02}
            for name,value in raw.items():
                pred,state=fit_decoder(value,targets[input_name],fit)
                stats[input_name][name]=metrics(pred,targets[input_name],hold,clusters,support)
                decoder_states[input_name][name]=cpu_tree(state);predictions[input_name][name]=pred.cpu()
                hashes[input_name][name]=tensor_hash(value)
                engineering[input_name+'_'+name]=float((replay_decoder(value,state)-pred).abs().max())
            for training in ('Initialized','Trained'):
                torch.manual_seed(SEED);model=SpatialConsequence(h.shape[1],26).to(dev)
                if training=='Trained':
                    state=source['states']['full_Flow'] if input_name=='Nominal' else forecast['states']['PredictedSurface']
                    model.load_state_dict(state)
                model.eval();before=tensor_hash(torch.cat([v.detach().flatten() for v in model.parameters()]))
                stages=extract_stages(model,h,g,flows[input_name],ids)
                if training=='Trained':
                    raw_output=stages['Output']*source['target_scale'].to(dev)+source['base'].to(dev)[ids,None]
                    test_slots=[np.flatnonzero(source['test']==v)[0] for v in test]
                    expected=source['candidate_arrays']['Flow'][test_slots] if input_name=='Nominal' else forecast['candidates']['PredictedSurface'][test_slots]
                    error=float(np.max(np.abs(raw_output[hold].cpu().numpy()-expected)))
                    engineering[input_name+'_original_candidate_replay']=error
                    actual=raw_output[torch.arange(len(ids),device=dev),source_arms[ids]].cpu().numpy()
                    expected=source['full_predictions']['Flow'][ids.cpu().numpy()] if input_name=='Nominal' else forecast['predictions']['PredictedSurface'][ids.cpu().numpy()]
                    engineering[input_name+'_original_factual_replay']=float(np.max(np.abs(actual-expected)))
                    assert max(engineering.values())<1e-5
                for stage,value in stages.items():
                    name=training+'_'+stage;value=value-value[:,:1]
                    pred,state=fit_decoder(value,targets[input_name],fit)
                    stats[input_name][name]=metrics(pred,targets[input_name],hold,clusters,support)
                    decoder_states[input_name][name]=cpu_tree(state);predictions[input_name][name]=pred.cpu()
                    hashes[input_name][name]=tensor_hash(value)
                    feature_rms[input_name][name]=float(value[hold,1:].square().mean().sqrt())
                    engineering[input_name+'_'+name]=float((replay_decoder(value,state)-pred).abs().max())
                    print(json.dumps(dict(input=input_name,stage=name,gain=stats[input_name][name]['gain_vs_zero'],elapsed=time.monotonic()-started)),flush=True)
                assert before==tensor_hash(torch.cat([v.detach().flatten() for v in model.parameters()]))
                del stages,model
        choices={name:dict(zip(('decision','status'),decision(value))) for name,value in stats.items()}
        status='UNCLEAR' if args.smoke else ('PROMISING' if any(c['status']=='PROMISING' for c in choices.values()) else 'UNPROMISING' if all(c['status']=='UNPROMISING' for c in choices.values()) else 'UNCLEAR')
        result=dict(status=status,engineering_only=args.smoke,choices=choices,metrics=stats,
            engineering_replay=engineering,feature_rms=feature_rms,elapsed_seconds=time.monotonic()-started,
            peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(dev),
            limits='Known generated input-motion decoding on exposed14arm grid; linear recoverability only. No measured future target, E/I re-fit, new simulation, downstream task, selector or policy utility claim.')
        save('result.json',result)
        torch.save(dict(train=train,test=test,clusters=source['clusters'],ids=ids.cpu(),
            targets=cpu_tree(targets),support=support.cpu(),decoder_states=decoder_states,
            predictions=predictions,feature_hashes=hashes),args.run_dir/'diagnostic.pt')
        assert sha(args.dataset)==manifest['dataset_sha256']
        assert sha(args.forecast_run/'diagnostic.pt')==manifest['forecast_diagnostic_sha256']
        assert all(sha(ROOT/k)==v for k,v in manifest['code_sha256'].items())
        assert sum(f.stat().st_size for f in args.run_dir.rglob('*') if f.is_file())<80*1024**2
        manifest['run_status']='COMPLETED';print(json.dumps(dict(status=status,choices=choices,elapsed=result['elapsed_seconds'])),flush=True)
    except BaseException as error:
        manifest.update(run_status='FAILED',failure=f'{type(error).__name__}: {error}');raise
    finally:
        manifest.update(completed_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started);save('manifest.json',manifest)


if __name__=='__main__':main()
