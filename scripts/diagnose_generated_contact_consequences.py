#!/usr/bin/env python3
"""Cal-only, current-input replay of frozen H10 consequence forecasts."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
from run_paired_evaluator_resolution import sha, gpu_admission

BASE = ROOT/'src/task/CmResidual/research/contact_consequence/output'
CARD = ROOT/'docs/experiments/probes/P-20261002-generated-contact-consequence-diagnostic.md'
PRIMARY = ROOT/'docs/experiments/probes/P-20261002-optimized-contact-opportunity-results-r1.json'
PRIMARY_AUDIT = ROOT/'docs/experiments/probes/P-20261002-optimized-contact-opportunity-audit-r1.json'
MODES = ('cm', 'state_only', 'shuffled', 'direct_score')


def run(args):
    started = time.monotonic()
    if args.output.parent.resolve() != BASE.resolve() or args.output.is_symlink():
        raise ValueError('unique owned output directory required')
    source_manifest = args.source/'run_manifest.json'
    source = json.loads(source_manifest.read_text())
    primary = json.loads(PRIMARY.read_text())
    audit = json.loads(PRIMARY_AUDIT.read_text())
    if (source['run_status'] != 'COMPLETED' or source['child_exit_code'] != 0
            or [p['seed'] for p in source['phases']] != list(range(591, 603))):
        raise ValueError('complete fixed source panel required')
    if not audit['audit_passed'] or audit['result_sha256'] != sha(PRIMARY) or primary['label'] != 'UNCLEAR':
        raise ValueError('preserved audited HF20 primary required')
    hashes = dict(source['input_sha256'])
    hashes.update(primary['input_sha256'])
    for path in (source_manifest, PRIMARY, PRIMARY_AUDIT, CARD, Path(__file__), args.checkpoint,
                 ROOT/'scripts/run_paired_evaluator_resolution.py',
                 ROOT/'src/task/CmResidual/paired_evaluation.py'):
        hashes[str(path.resolve())] = sha(path)
    if any(sha(Path(k)) != v for k, v in hashes.items()):
        raise ValueError('frozen input drift')
    admission = gpu_admission(args.gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = admission['uuid']
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    args.output.mkdir(exist_ok=False)
    manifest = dict(experiment_id='P-20261002-generated-contact-consequence-diagnostic',
        family='HD03', probe_index_in_family=1, run_id=args.output.name, run_status='RUNNING',
        pid=os.getpid(), command=sys.argv, input_sha256=hashes, admission=admission,
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        wall_limit_seconds=600, output_limit_bytes=32 << 20, model_training=False,
        scope='cal-only model/decision diagnostic; no new rollout or held tuning; HF20 gates unchanged')
    def save():
        (args.output/'run_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    def bound():
        if time.monotonic()-started > 600:
            raise TimeoutError('diagnostic wall bound')
        if sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file()) > 32 << 20:
            raise ValueError('diagnostic storage bound')
    save()
    try:
        import torch
        from src.task.CmResidual.contact_geometry_consequence import GeometryConsequenceModel, node_inputs, normalize, FEATURES
        from src.task.CmResidual.optimized_contact_actions import live_inputs
        from src.task.CmResidual.native_pd_selector import native_pd_targets
        from src.task.CmResidual.paired_evaluation import fingerprint
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if not torch.cuda.is_available():
            raise RuntimeError('GPU required for model replay')
        device = torch.device('cuda:0')
        bundle = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if bundle['schema'] != 'ref2dex.contact_geometry_consequence.v1':
            raise ValueError('frozen consequence bundle required')
        norm = {k:v.to(device) for k,v in bundle['normalization'].items()}
        models = {}
        for mode in MODES:
            models[mode] = []
            for weights in bundle['models'][mode]:
                model = GeometryConsequenceModel(bundle['physical_dim'], mode).to(device)
                model.load_state_dict(weights, strict=True)
                model.eval().requires_grad_(False)
                models[mode].append(model)
        parameter_before = fingerprint({k:[m.state_dict() for m in v] for k,v in models.items()})
        forecasts, labels, assignments, propensities, changed, identities = [], [], [], [], [], []
        replay_errors = {'joint_height_mm':0., 'geometric_loss_probability':0.}
        covered_rows = 0
        for phase in source['phases']:
            bound()
            if phase['run_status'] != 'COMPLETED' or phase['native_exit_code'] != 0 or phase['audit_exit_code'] != 0:
                raise ValueError('source phase/audit not terminal')
            directory = Path(phase['directory'])
            record_path, planning_path = directory/'records.pt', directory/'planning.pt'
            if sha(record_path) != phase['result']['record_sha256'] or sha(planning_path) != phase['result']['planning_sha256']:
                raise ValueError('record/planning drift')
            b = torch.load(record_path, map_location='cpu', weights_only=False)
            planning = torch.load(planning_path, map_location='cpu', weights_only=False)
            if b['schema'] != 'ref2dex.optimized_contact_source.v1' or b['generator_checkpoint_sha256'] != sha(args.checkpoint):
                raise ValueError('source/checkpoint mismatch')
            take = torch.where((b['split_group_bucket'] >= 50) & (b['split_group_bucket'] < 70))[0]
            n = len(take)
            current = {k:b[k][take].to(device) for k in
                ('history', 'state', 'initial_hand_force', 'initial_object_force', 'mass_kg',
                 'initial_clearance', 'rest_z', 'candidate_actions', 'candidate_weights')}
            native = b['native_observation'][take, 0].to(device)
            inputs = live_inputs(current['history'], native, current['initial_hand_force'],
                current['initial_object_force'], current['mass_kg'], b['gravity_magnitude'],
                current['initial_clearance'], current['rest_z'])
            q = current['state'][:,None,:18].expand(-1,8,-1)
            pd = native_pd_targets(current['candidate_actions'], q, b['pd_offset'].to(device), b['pd_scale'].to(device))
            nodes, actions = node_inputs(current['state'][:,None].expand(-1,8,-1),
                native[:,None].expand(-1,8,-1), pd, current['candidate_weights'])
            data = {k:inputs[k][:,None].expand(-1,8,*inputs[k].shape[1:]).flatten(0,1)
                    for k in ('history','physical','prior')}
            data.update(node_state=nodes.flatten(0,1), node_action=actions.flatten(0,1),
                law=torch.tensor([0,1,1,1,1,1,1,1],device=device,dtype=q.dtype)[None,:,None].expand(n,-1,-1).flatten(0,1))
            normalized = normalize(data, norm)
            with torch.no_grad():
                prediction = torch.stack([torch.stack([torch.cat([
                    model(**{k:normalized[k][s:s+128] for k in FEATURES}, prior=data['prior'][s:s+128])
                    for s in range(0,n*8,128)]).reshape(n,8,52) for model in models[mode]]) for mode in MODES]).cpu()
            if not torch.isfinite(prediction).all():
                raise ValueError('nonfinite full-network forecast')
            # Link only calibration forecasts to the already audited planning log.
            lookup = {(int(b['env_id'][i]),int(b['trigger'][i])):j for j,i in enumerate(take)}
            covered = set()
            for batch in planning['batches']:
                for j, env in enumerate(batch['env_id']):
                    key = (int(env),batch['tick'])
                    if key not in lookup:
                        continue
                    row = lookup[key]
                    if row in covered:
                        raise ValueError('duplicate cal planning row')
                    covered.add(row)
                    for arm, mode in ((2,'cm'),(3,'direct_score'),(4,'shuffled')):
                        forecast = prediction[MODES.index(mode),:,row,arm]
                        report = batch['modes'][mode]
                        for name, actual, expected in (
                            ('joint_height_mm',forecast[:,51]*10,report['predicted_score_mm'][:,j]),
                            ('geometric_loss_probability',forecast[:,49].sigmoid(),report['predicted_loss'][:,j])):
                            replay_errors[name] = max(replay_errors[name],float((actual-expected).abs().max()))
                            if not torch.allclose(actual,expected,atol=1e-4 if name=='joint_height_mm' else 2e-6,rtol=2e-6):
                                raise ValueError('full NN cached forecast mismatch '+name)
            if len(covered) != n:
                raise ValueError('incomplete cal forecast/planning coverage')
            covered_rows += len(covered)
            forecasts.append(prediction)
            # Futures enter factual labels only, after all pre-only predictions.
            bits = b['future_contact'][take,-3:]
            clear = b['future_clearance'][take] >= .002
            hand, obj = bits[:,:,0].all(-1), bits[:,:,1].all(-1)
            joint = hand & obj
            support = joint & clear[:,-3:].all(-1)
            height = (b['future_state'][take,-3:,38].amin(-1)-b['rest_z'][take]).clamp_min(0)
            past = torch.cat(((b['initial_clearance'][take]>=.002)[:,None],clear[:,:-1]),-1).cummax(-1).values
            loss = (past & ~clear).any(-1)
            labels.append(torch.stack((hand.float(),obj.float(),joint.float(),loss.float(),
                (support & (height>=.03)).float(),height*support*1000),-1))
            assignments.append(b['assignment'][take]); propensities.append(b['propensity'][take])
            changed.append((pd[:,2]-pd[:,1]).abs().amax(-1).cpu() > 1e-5)
            identities.extend([dict(seed=phase['seed'],env_id=int(b['env_id'][i]),tick=int(b['trigger'][i]),
                bucket=int(b['split_group_bucket'][i])) for i in take])
        pred = torch.cat(forecasts,2); actual = torch.cat(labels); chosen = torch.cat(assignments)
        propensity = torch.cat(propensities).double(); changed = torch.cat(changed)
        n = len(actual)
        if n != 306 or covered_rows != n or any(not 50 <= v['bucket'] < 70 for v in identities):
            raise ValueError('fixed calibration subset required')
        rows = torch.arange(n)
        member_factual = pred[:,:,rows,chosen]
        factual = member_factual.mean(1)
        probabilities = member_factual[:,:,:,[47,48,49,50]].sigmoid().mean(1)
        truth = actual[:,[0,1,3,4]]
        def metrics(mode_index, mask):
            count = int(mask.sum())
            if not count:
                return dict(rows=0)
            p = probabilities[mode_index,mask]; y = truth[mask]
            h = factual[mode_index,mask,51]*10; target = actual[mask,5]
            return dict(rows=count,
                brier=dict(zip(('hand_presence','object_presence','geometric_loss','supported_lift'),
                               ((p-y).square().mean(0)).tolist())),
                observed_rates=dict(zip(('hand_presence','object_presence','geometric_loss','supported_lift'),y.mean(0).tolist())),
                predicted_rates=dict(zip(('hand_presence','object_presence','geometric_loss','supported_lift'),p.mean(0).tolist())),
                joint_presence_observed=float(actual[mask,2].mean()),
                supported_height_mae_mm=float((h-target).abs().mean()),
                supported_height_bias_mm=float((h-target).mean()),
                observed_supported_height_mm=float(target.mean()),predicted_supported_height_mm=float(h.mean()))
        factual_metrics = {mode:dict(all=metrics(i,torch.ones(n,dtype=torch.bool)),
            by_arm={str(a):metrics(i,chosen==a) for a in range(8)},
            generated_pool=metrics(i,(chosen>=2)&(chosen<=4)),random_pool=metrics(i,chosen>=5)) for i,mode in enumerate(MODES)}
        consistency = {}
        for i,mode in enumerate(MODES):
            probs = pred[i,:,:, :, [47,48,49,50]].sigmoid()
            mean = probs.mean(0)
            lower = (mean[...,0]+mean[...,1]-1).clamp_min(0)
            upper = mean[...,:2].amin(-1)
            violation = mean[...,3]-upper
            cmp = {}
            for name,mask in (('all',torch.ones(n,dtype=torch.bool)),('changed_cup_pd',changed)):
                cmp[name] = dict(rows=int(mask.sum()),
                    hand_probability_difference=float((mean[mask,2,0]-mean[mask,1,0]).mean()),
                    object_probability_difference=float((mean[mask,2,1]-mean[mask,1,1]).mean()),
                    loss_probability_difference=float((mean[mask,2,2]-mean[mask,1,2]).mean()),
                    supported_lift_probability_difference=float((mean[mask,2,3]-mean[mask,1,3]).mean()),
                    predicted_joint_height_gain_mm=float(((pred[i,:,:,2,51]-pred[i,:,:,1,51])*10).mean(0)[mask].mean()),
                    frechet_definite_decline_rows=int((upper[mask,2] < lower[mask,1]).sum()),
                    frechet_decline_over_5pp_rows=int((upper[mask,2]+.05 < lower[mask,1]).sum()))
            consistency[mode] = dict(cm_vs_cup=cmp,
                lift_subset_violation_by_arm={str(a):dict(rows=int((violation[:,a]>1e-6).sum()),
                    over_5pp_rows=int((violation[:,a]>.05).sum()),maximum_excess=float(violation[:,a].max())) for a in range(8)},
                scope='predicted marginals/internal consistency; no counterfactual outcome; direct-score non-height heads are prior-only')
        allocation = {str(a):dict(rows=int((chosen==a).sum()),
            ht_weight_mass=float(((chosen==a).double()/propensity).mean()),
            conditional_observed_joint_presence=float(actual[chosen==a,2].mean())) for a in range(8)}
        parameter_after = fingerprint({k:[m.state_dict() for m in v] for k,v in models.items()})
        if parameter_after != parameter_before or any(sha(Path(k)) != v for k,v in hashes.items()):
            raise ValueError('parameters or pinned inputs changed')
        torch.cuda.synchronize()
        torch.save(dict(schema='ref2dex.generated_contact_forecast_diagnostic.v1', modes=MODES,
            prediction=pred, factual_labels=actual, assignment=chosen, propensity=propensity,
            changed_cup_pd=changed, identities=identities, future_used_as_model_input=False),args.output/'forecasts.pt')
        result = dict(experiment_id=manifest['experiment_id'],run_status='COMPLETED',rows=n,
            source_hf20_label=primary['label'],device='cuda:0',physical_gpu=args.gpu,
            planning_forecast_max_errors=replay_errors,planning_cal_rows=covered_rows,
            factual_metrics=factual_metrics,consistency=consistency,allocation=allocation,
            parameter_fingerprint_before=parameter_before,parameter_fingerprint_after=parameter_after,
            parameter_frozen=True,input_hashes_unchanged=True,elapsed_seconds=time.monotonic()-started,
            scope=manifest['scope'],source_cost_separate=True)
        (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
        manifest.update(run_status='COMPLETED',elapsed_seconds=time.monotonic()-started,
            result_sha256=sha(args.output/'results.json'),forecast_sha256=sha(args.output/'forecasts.pt'),
            input_hashes_unchanged=True,parameters_frozen=True)
        save();bound()
        print(json.dumps(dict(run_status='COMPLETED',rows=n,elapsed_seconds=manifest['elapsed_seconds'],
            forecast_errors=replay_errors)),flush=True)
    except BaseException as exc:
        manifest.update(run_status='FAILED',elapsed_seconds=time.monotonic()-started,error=repr(exc))
        save()
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=BASE/'P-20261002-optimized-contact-opportunity-source-r1')
    parser.add_argument('--checkpoint',type=Path,default=BASE/'P-20261002-contact-geometry-fit-r1/contact_geometry_consequence.pt')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',type=int,default=1)
    run(parser.parse_args())
