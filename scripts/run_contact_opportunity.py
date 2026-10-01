#!/usr/bin/env python3
"""One frozen six-candidate, two-repeat local opportunity Probe."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
from run_paired_evaluator_resolution import gpu_admission, sha, PYTHON, MOTIONS, R7


def run(args):
    base = (ROOT/'src/task/CmResidual/research/contact_consequence/output').resolve()
    output = args.output.absolute()
    if output.parent.resolve() != base or output.is_symlink():
        raise ValueError('output must be unique owned task directory')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    prior = []
    for path in args.prior:
        previous = json.loads((path/'run_manifest.json').read_text())
        if previous['run_status'] not in ('FAILED', 'STOPPED'):
            raise ValueError('prior attempt must be terminal')
        prior += previous.get('prior_attempts', [])
        prior.append(dict(path=str(path.resolve()), seconds=previous['elapsed_seconds'],
                          bytes=sum(p.stat().st_size for p in path.rglob('*') if p.is_file())))
    if len({p['path'] for p in prior}) != len(prior): raise ValueError('duplicate prior accounting')
    prior_seconds = sum(p['seconds'] for p in prior)
    prior_bytes = sum(p['bytes'] for p in prior)
    route_path = ROOT/'src/task/CmResidual/configs/hf02_temporal_canonical_route.json'
    route = json.loads(route_path.read_text())
    inputs = [route_path, R7/'environment.yaml', R7/'training.yaml']
    for name in route['candidate_experts']:
        spec = route['experts'][name]; path = (ROOT/spec['checkpoint']).resolve()
        if sha(path) != spec['sha256']: raise ValueError('expert provenance drift: '+name)
        inputs.append(path)
    for motion in route['motions']:
        path = MOTIONS/motion['name']/'interaction_hand_inspire.pt'
        if sha(path) != motion['interaction_hand_sha256']: raise ValueError('motion hash drift')
        inputs.append(path)
    sources = [Path(__file__), ROOT/'scripts/collect_contact_consequences.py',
               ROOT/'src/task/CmResidual/contact_consequence.py', ROOT/'src/task/CmResidual/paired_evaluation.py',
               ROOT/'src/task/CmResidual/physical_value_live.py', ROOT/'third_party/DExplore/dexplore/evaluate.py',
               ROOT/'third_party/DExplore/dexplore/env/tasks/dexplore_inspire.py']
    hashes = {str(p.resolve()): sha(p) for p in inputs+sources}
    manifest = dict(experiment_id='P-20261001-contact-consequence-opportunity', family='HF09',
                    probe_index_in_family=1, run_id=output.name, run_status='RUNNING', pid=os.getpid(),
                    git_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
                    command=sys.argv, input_sha256=hashes, prior_attempts=prior,
                    actor_training=False, cm_training=False, phases=[],
                    wall_limit_seconds=3600, output_limit_bytes=8<<30)

    def save(): (output/'run_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    def check():
        if prior_seconds+time.monotonic()-started > 3540: raise TimeoutError('whole Probe time budget')
        if prior_bytes+sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) > 8<<30:
            raise RuntimeError('Probe storage budget')
        if any(sha(p) != hashes[str(p.resolve())] for p in sources): raise ValueError('executing source changed')
    def execute(arm, repeat, reference=None):
        check(); admission = gpu_admission(args.gpu)
        name = f'arm{arm}_repeat{repeat}'
        directory = output/name
        source = (ROOT/route['experts']['source_e260']['checkpoint']).resolve()
        command = [PYTHON, '-u', str(ROOT/'scripts/collect_contact_consequences.py'),
                   '--output-dir', str(directory), '--arm', str(arm), '--repeat', str(repeat),
                   '--panel-seed', str(args.seed), '--max-states', str(args.max_states),
                   '--wall-seconds', '240', '--task', 'Dexplore_Inspire',
                   '--cfg_env', str(R7/'environment.yaml'), '--cfg_train', str(R7/'training.yaml'),
                   '--checkpoint', str(source), '--motion_file', str(MOTIONS), '--headless',
                   '--num_envs', '96', '--seed', str(args.seed), '--sim_device', 'cuda:0',
                   '--rl_device', 'cuda:0', '--graphics_device_id', '0', '--disable-early-termination',
                   '--output', str(directory/'unused.json'), '--output_path', str(directory/'player')]
        if reference: command += ['--reference', str(reference)]
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=admission['uuid'], LOCAL_RANK='0',
                           RANK='0', WORLD_SIZE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                           PYTHONDONTWRITEBYTECODE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                           PYTHONHASHSEED=str(args.seed), TORCH_EXTENSIONS_DIR=str(output/'cache/torch_extensions'),
                           XDG_CACHE_HOME=str(output/'cache'))
        environment['LD_LIBRARY_PATH']='/home2/wyy/miniconda3/envs/graspenv/lib:'+environment.get('LD_LIBRARY_PATH','')
        phase = dict(name=name, arm=arm, repeat=repeat, command=command, gpu=admission,
                     directory=str(directory), run_status='STARTED')
        manifest['phases'].append(phase); save()
        process = None; begin = time.monotonic()
        print(json.dumps(dict(phase=name, run_status='STARTED')), flush=True)
        try:
            with (output/(name+'.log')).open('x') as log:
                process = subprocess.Popen(command, cwd=ROOT/'third_party/DExplore', env=environment,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                phase.update(pid=process.pid, pgid=process.pid); save()
                code = process.wait(timeout=min(290, 3540-prior_seconds-(time.monotonic()-started)))
            if code: raise RuntimeError('native exit '+str(code)+': '+name)
            result = json.loads((directory/'results.json').read_text())
            if result['run_status'] != 'COMPLETED': raise ValueError('incomplete native phase')
            phase.update(run_status='COMPLETED', result=result, record_sha256=sha(directory/'records.pt'))
            print(json.dumps(dict(phase=name, result=result)), flush=True)
            return directory/'records.pt'
        except BaseException as error:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL); process.wait(timeout=5)
            phase.update(run_status='FAILED', error=repr(error)); raise
        finally:
            phase['elapsed_seconds'] = time.monotonic()-begin; save()
    save()
    try:
        # Repeat0 base also captures the cold snapshot and pretrigger trace.
        files = {(4,0): execute(4,0)}
        reference = files[(4,0)]
        for arm in range(6):
            for repeat in range(2):
                if (arm,repeat) != (4,0): files[(arm,repeat)] = execute(arm,repeat,reference)
        import torch
        from src.task.CmResidual.contact_consequence import opportunity_gate, BASE_INDEX
        ref = torch.load(reference, map_location='cpu', weights_only=False)
        n = int(ref['selected'].sum())
        lift = torch.zeros(n,6,2); contact = torch.zeros_like(lift); drop = torch.zeros_like(lift, dtype=torch.bool)
        paired = torch.ones(n, dtype=torch.bool)
        for key,path in files.items():
            data = torch.load(path, map_location='cpu', weights_only=False)
            if data['initial_fingerprint'] != ref['initial_fingerprint'] or not torch.equal(data['selected'], ref['selected']):
                raise ValueError('cold state or contact state identity mismatch')
            if not torch.equal(data['trigger'], ref['trigger']): raise ValueError('trigger identity mismatch')
            if not torch.equal(data['candidate_actions'][ref['selected']], ref['candidate_actions'][ref['selected']]):
                raise ValueError('candidate proposal changed')
            arm,repeat=key
            lift[:,arm,repeat]=data['outcome']['supported_lift_mm']
            contact[:,arm,repeat]=data['outcome']['contact_fraction']
            drop[:,arm,repeat]=data['outcome']['drop']
            paired &= data['paired_valid']
        gate = opportunity_gate(lift,contact,drop,paired_valid=bool(paired.all()))
        panel = dict(schema=ref['schema'], seed=args.seed, reference=str(reference),
                     files={f'{a}/{r}':str(p) for (a,r),p in files.items()},
                     supported_lift_mm=lift, contact_fraction=contact, drop=drop,
                     paired_valid=paired, selected_envs=ref['selected'].nonzero().flatten())
        torch.save(panel, output/'panel.pt')
        report = dict(run_status='COMPLETED', opportunity=gate,
                      paired_valid_states=int(paired.sum()), drop_eligible=int(ref['outcome']['drop_eligible'].sum()),
                      decision='PROCEED_TO_CONSEQUENCE_RANKING' if gate['passed'] else 'REVIEW_CANDIDATE_OPPORTUNITY',
                      baseline_status='PARTIAL', cm_utility='UNPROVEN')
        if any(sha(Path(p)) != digest for p,digest in hashes.items()): raise ValueError('input changed after execution')
        report.update(input_hashes_unchanged=True, elapsed_seconds=time.monotonic()-started,
                      cumulative_seconds=prior_seconds+time.monotonic()-started,
                      output_bytes=prior_bytes+sum(p.stat().st_size for p in output.rglob('*') if p.is_file()))
        (output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
        manifest.update(run_status='COMPLETED', decision=report['decision'], label=gate['label'], input_hashes_unchanged=True)
        check(); print(json.dumps(report),flush=True)
    except BaseException as error:
        manifest.update(run_status='FAILED', error=repr(error)); raise
    finally:
        manifest['elapsed_seconds']=time.monotonic()-started; save()


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=4)
    p.add_argument('--seed',type=int,default=330)
    p.add_argument('--max-states',type=int,default=32)
    p.add_argument('--prior',type=Path,action='append',default=[])
    run(p.parse_args())
