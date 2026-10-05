#!/usr/bin/env python3
"""No-fit frozen-feature/PCA/decoder replay of physical action fidelity."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[5];TASK=ROOT/'src/task/cm-interaction-oracle'
sys.path[:0]=[str(ROOT),str(TASK/'src'),str(TASK/'tools/run')]
from probe_spatial_action_fidelity import load_inputs,metrics,decision
from spatial_action_fidelity import intrinsic_finger_flow,current_finger_support,extract_stages,decoder_features,replay_decoder
from spatial_consequence import SpatialConsequence,SEED
from execution_geometry import endpoint_flows
from probe_execution_geometry import to_device
from probe_interventions import sha,tensor_hash


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--forecast-run',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args();out=args.run_dir/'engineering_replay.json'
    if out.exists():raise FileExistsError(out)
    started=time.monotonic();torch.set_num_threads(2)
    m=json.loads((args.run_dir/'manifest.json').read_text());assert m['run_status']=='COMPLETED' and not m['smoke']
    assert all(sha(ROOT/k)==v for k,v in m['code_sha256'].items())
    assert sha(args.forecast_run/'diagnostic.pt')==m['forecast_diagnostic_sha256']
    d=torch.load(args.run_dir/'diagnostic.pt',map_location='cpu',weights_only=False)
    result=json.loads((args.run_dir/'result.json').read_text())
    dev=torch.device('cuda:0');torch.cuda.set_device(dev)
    p,bridge,g,h,source,forecast,_,_=load_inputs(args,dev)
    ids=d['ids'].to(dev);fit=torch.arange(len(d['train']),device=dev)
    hold=torch.arange(len(d['train']),len(ids),device=dev)
    assert np.array_equal(ids.cpu().numpy(),np.concatenate((source['train'],source['test'])))
    assert not set(source['clusters'][d['train']])&set(source['clusters'][d['test']])
    rows=(fit[:,None]*15+torch.arange(15,device=dev)).flatten()
    support=current_finger_support(g)[ids];assert torch.equal(support.cpu(),d['support'])
    qs=dict(Nominal=g['targets'],Forecast=forecast['oof_q']['Ha'].to(dev))
    flows=dict(Nominal=g['nominal_flow'],Forecast=endpoint_flows(p,bridge,g,qs['Forecast'])[0])
    replay={};normal_equations={};norm_errors={};metric_errors={};stats={}
    for input_name,q in qs.items():
        target=(q[ids,:,6:]-q[ids,:1,6:])/.32
        assert torch.equal(target.cpu(),d['targets'][input_name])
        stats[input_name]={}
        stages=dict(RawObject=(flows[input_name][ids]-flows[input_name][ids,:1]).flatten(2)/.02,
                    RawHandBase=intrinsic_finger_flow(p,bridge,g,q)[ids].flatten(2)/.02)
        for training in ('Initialized','Trained'):
            torch.manual_seed(SEED);model=SpatialConsequence(h.shape[1],26).to(dev)
            if training=='Trained':model.load_state_dict(source['states']['full_Flow'] if input_name=='Nominal' else forecast['states']['PredictedSurface'])
            model.eval()
            values=extract_stages(model,h,g,flows[input_name],ids)
            stages.update({training+'_'+k:v-v[:,:1] for k,v in values.items()})
        for name,value in stages.items():
            assert tensor_hash(value)==d['feature_hashes'][input_name][name]
            state=to_device(d['decoder_states'][input_name][name],dev);norm=state['norm'];model=state['model']
            x=decoder_features(value,norm)
            pred=replay_decoder(value,state)
            key=input_name+'_'+name
            replay[key]=float((pred.cpu()-d['predictions'][input_name][name]).abs().max())
            replay[key+'_scores']=float((x.cpu()-d['decoder_inputs'][input_name][name]).abs().max())
            raw=value.flatten(0,1);score=(raw-norm['mean'])@norm['projection']
            norm_errors[key]=max(float((norm['mean']-raw[rows].mean(0)).abs().max()),
                float((norm['score_mean']-score[rows].mean(0)).abs().max()),
                float((norm['score_scale']-score[rows].std(0,unbiased=False).clamp_min(1e-6)).abs().max()))
            xc=x[rows].double()-model['x_mean'];yc=target.flatten(0,1)[rows].double()-model['y_mean']
            rhs=xc.T@yc
            residual=(xc.T@xc+model['alpha']*torch.eye(x.shape[-1],device=dev,dtype=torch.float64))@model['weight']-rhs
            normal_equations[key]=float(residual.norm()/rhs.norm().clamp_min(1e-12))
            values=metrics(pred,target,hold,source['clusters'][d['test']],support)
            original=result['metrics'][input_name][name]
            metric_errors[key]=max(abs(values[k]-original[k]) for k in ('mse','zero_mse','gain_vs_zero','signed_agreement','current_support_gain'))
            stats[input_name][name]=values
        choice,status=decision(stats[input_name]);assert result['choices'][input_name]==dict(decision=choice,status=status)
        # LocalFlow intentionally omits geometry and averages selected motion;
        # its weakness cannot by itself identify a radius/selection defect.
        assert torch.equal(stages['Initialized_LocalFlow'],stages['Trained_LocalFlow'])
    assert max(replay.values())<1e-5 and max(norm_errors.values())<1e-6
    assert max(normal_equations.values())<1e-7 and max(metric_errors.values())<1e-6
    report=dict(status='PASS',replay_errors=replay,source_normalizer_errors=norm_errors,
        normal_equation_relative_residuals=normal_equations,metric_errors=metric_errors,
        diagnostic_sha256=sha(args.run_dir/'diagnostic.pt'),audit_code_sha256=sha(Path(__file__).resolve()),
        elapsed_seconds=time.monotonic()-started,
        interpretation_limit='LocalFlow is a uniform pooled-motion proxy, not the complete network edge input. Linear/PCA recoverability failure does not prove information loss; all targets are known generated input motion on the14arm grid.')
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='PASS',replay_max=max(replay.values()),source_norm_max=max(norm_errors.values()),ridge_residual_max=max(normal_equations.values()),elapsed=report['elapsed_seconds'])))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(14,6))
    names=['RawObject','RawHandBase','Trained_LocalFlow','Trained_Contact','Trained_Tokens','Trained_Fused','Trained_Output']
    for ax,input_name in zip(axes,('Nominal','Forecast')):
        values=result['metrics'][input_name];gain=[values[n]['gain_vs_zero'] for n in names]
        low=[v-values[n]['bootstrap']['lower95'] for n,v in zip(names,gain)]
        high=[values[n]['bootstrap']['upper95']-v for n,v in zip(names,gain)]
        # Bootstrap percentile intervals need not bracket the point estimate.
        y=np.arange(len(names));ax.barh(y,gain,alpha=.7)
        for i,n in enumerate(names):ax.plot([values[n]['bootstrap']['lower95'],values[n]['bootstrap']['upper95']],[i,i],color='black',linewidth=1.2)
        ax.set_yticks(y);ax.set_yticklabels(names);ax.invert_yaxis();ax.set_xlim(-.2,1.05)
        ax.set_xlabel('Physical input-motion decoding gain over zero');ax.set_title(input_name)
    fig.suptitle('Frozen spatial action fidelity / exposed14arm grid / linear decoder')
    fig.tight_layout();fig.savefig(args.run_dir/'action_fidelity.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
