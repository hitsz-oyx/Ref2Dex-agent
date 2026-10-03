"""Bounded fixed-density task screen, with engineering prerequisite and audit."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_contact_response_probe import sha, admission, PYTHON
from scripts.resume_continuous_critic_policy import run_owned_child, bytes_in


def aggregate(rows):
    counts = np.zeros((3, 4, 2), dtype=np.int64)
    totals = np.zeros_like(counts)
    for row in rows:
        cell = row['motion'], row['arm'], row['load']
        counts[cell] += int(row['physical105'])
        totals[cell] += 1
    if len(rows) != 768 or not np.all(totals == 32):
        raise ValueError('all motion/policy/load cells included')
    pooled = counts.sum(axis=0)
    nominal, heavy = pooled[3] / 96.
    gates = dict(nominal_cold_q_at_least25pct=bool(nominal >= .25),
                 heavy_cold_q_between10_and80pct=bool(.10 <= heavy <= .80),
                 nominal_minus_heavy_at_least15pp=bool(nominal - heavy >= .15))
    return dict(run_status='COMPLETED', label='PROMISING' if all(gates.values()) else 'UNPROMISING',
                gates=gates, success_per_motion_arm_load_per32=counts.tolist(),
                success_pooled_arm_load_per96=pooled.tolist(),
                arm_names=['p0', 'cm', 'dynamics_off', 'cold_q'],
                load_names=['density20', 'density1000'],
                cold_q_nominal_minus_heavy_pp=float((nominal-heavy)*100),
                new_native_episodes=768, env_control_ticks=155136,
                actual_optimizer_steps=0, fixed_actor_sensitivity_only=True,
                cm_training_benefit_not_tested=True, exact_paired_counterfactual=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    source, out = args.source.resolve(), args.output.resolve()
    if ROOT not in out.parents or out.exists():
        raise ValueError('unique owned output required')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True):
        raise ValueError('fixed clean code commit required')
    begin = time.monotonic()
    old = json.loads((source/'run_manifest.json').read_text())
    if old['run_status'] != 'COMPLETED' or old['experiment_id'] != 'P-20261003-inspire-filter-impact':
        raise ValueError('fixed corrected-physics source')
    command = list(next(phase['command'] for phase in old['phases'] if phase['name'] == 's655'))
    hashes = dict(old['input_sha256'])
    # Prior hashes record historical code versions. Protect current code and source
    # manifests independently; retain original asset/checkpoint/reference identities.
    hashes = {name: digest for name, digest in hashes.items()
              if '/data/assets/' in name or '/P-20261001-hold-plateau-reference-r1/' in name}
    for flag in ('--policy-checkpoint', '--continuous-checkpoint', '--checkpoint', '--cfg_env',
                 '--cfg_train', '--references-manifest', '--option-actors'):
        path = Path(command[command.index(flag)+1]).resolve()
        hashes[str(path)] = sha(path)
    hashes[str(source/'run_manifest.json')] = sha(source/'run_manifest.json')
    for name in subprocess.check_output(['git', 'ls-files', '*.py'], cwd=ROOT, text=True).splitlines():
        hashes[str(ROOT/name)] = sha(ROOT/name)
    for name in ('docs/decisions/D-20261003-latent-load-feasibility.md',
                 'docs/experiments/probes/P-20261003-latent-load-feasibility.md'):
        hashes[str(ROOT/name)] = sha(ROOT/name)
    base = Path(old['base_checkpoint'])
    actors = Path(command[command.index('--option-actors')+1])
    if sha(base) != old['policy_sha256'] or sha(actors) != old['trained_actor_sha256']:
        raise ValueError('fixed own actor identities')

    def verify():
        for name, digest in hashes.items():
            if sha(Path(name)) != digest:
                raise ValueError('protected input drift '+name)

    verify()
    out.mkdir()
    (out/'fit').mkdir()
    (out/'fit/actors.pt').symlink_to(actors)
    manifest = dict(experiment_id='P-20261003-latent-load-feasibility', run_id=out.name,
                    run_status='RUNNING', pid=os.getpid(), source_run=str(source),
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    input_sha256=hashes, base_checkpoint=str(base), policy_sha256=old['policy_sha256'],
                    trained_actor_sha256=sha(actors), panel_checkpoints={'701': old['panel_checkpoints']['655']},
                    phases=[], wall_limit_seconds=600, storage_limit_bytes=512<<20,
                    actual_optimizer_steps=0, prospective_evaluation_seed=701,
                    physics_density_kg_m3=[20, 1000], force_normalization_uses_nominal_mass=True)

    def save():
        manifest['wall_seconds'] = time.monotonic()-begin
        (out/'run_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')

    def execute(name, cmd, native):
        verify()
        gpu = None
        if native:
            for index in (6, 4, 5, 3, 7):
                try:
                    gpu = admission(index)
                    break
                except RuntimeError:
                    pass
            if gpu is None:
                raise RuntimeError('no fresh idle GPU')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu['uuid'] if gpu else '',
                   LOCAL_RANK='0', RANK='0', WORLD_SIZE='1', OMP_NUM_THREADS='2',
                   MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1',
                   TORCH_EXTENSIONS_DIR=str(out/'cache/torch_extensions'), XDG_CACHE_HOME=str(out/'cache'),
                   CUBLAS_WORKSPACE_CONFIG=':4096:8')
        env['LD_LIBRARY_PATH']='/home2/wyy/miniconda3/envs/graspenv/lib:'+env.get('LD_LIBRARY_PATH', '')
        phase = dict(name=name, command=cmd, run_status='RUNNING', admission=gpu,
                     execution_device='cuda' if gpu else 'cpu')
        manifest['phases'].append(phase)
        save()

        def spawned(child):
            phase.update(pid=child.pid, pgid=child.pid)
            save()

        def guard():
            if time.monotonic()-begin > 600 or bytes_in(out) > 512<<20:
                raise RuntimeError('bounded load-feasibility budget')
            if gpu:
                apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                                '--format=csv,noheader'], text=True)
                for line in apps.splitlines():
                    fields = [field.strip() for field in line.split(',')]
                    if len(fields)==2 and fields[0]==gpu['uuid'] and fields[1]!=str(phase.get('pid')):
                        raise RuntimeError('GPU contention; stop owned child only')
        print(json.dumps(dict(phase=name, status='STARTED', admission=gpu)), flush=True)
        start = time.monotonic()
        try:
            run_owned_child(cmd, ROOT/'third_party/DExplore', env, out/(name+'.log'), guard,
                            min(240, 600-(time.monotonic()-begin)), spawned)
            verify()
            phase['run_status'] = 'COMPLETED'
        except BaseException as error:
            phase.update(run_status='FAILED', error=repr(error))
            raise
        finally:
            phase['wall_seconds']=time.monotonic()-start
            save()
        print(json.dumps(dict(phase=name, status='COMPLETED')), flush=True)

    save()
    try:
        execute('engineering', [PYTHON, '-u', str(ROOT/'scripts/smoke_latent_load_physics.py'),
                                '--output', str(out/'engineering.json')], True)
        if not json.loads((out/'engineering.json').read_text())['passed']:
            raise ValueError('native mass/inertia prerequisite')
        directory = out/'s701'
        command[2] = str(ROOT/'scripts/run_latent_load_environment.py')
        for flag, value in (('--run-dir', directory), ('--output', directory/'unused.json'),
                            ('--output_path', directory/'player'), ('--eval-seed', 701), ('--seed', 701)):
            command[command.index(flag)+1] = str(value)
        execute('s701', command, True)
        execute('s701_audit', [PYTHON, '-u', str(ROOT/'scripts/audit_latent_load_panel.py'),
                              '--directory', str(out), '--panel', '701'], False)
        result = aggregate(json.loads((directory/'rows.json').read_text()))
        (out/'results.json').write_text(json.dumps(result, indent=2)+'\n')
        verify()
        manifest.update(run_status='COMPLETED', label=result['label'], bytes=bytes_in(out), inputs_unchanged=True)
        print(json.dumps(result), flush=True)
    except BaseException as error:
        manifest.update(run_status='FAILED', error=repr(error))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
