"""Owned, bounded, sequential native contact API capability checks."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_contact_response_probe import sha, admission, PYTHON
from scripts.resume_continuous_critic_policy import run_owned_child, bytes_in


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() or ROOT not in out.parents:
        raise ValueError('unique owned output')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True):
        raise ValueError('clean fixed commit')
    begin = time.monotonic()
    paths = [ROOT/'scripts/smoke_oracle_contact_api.py', Path(__file__),
             ROOT/'docs/decisions/D-20261003-effect-interaction-oracle.md',
             Path('/home2/wyy/isaac-gym/isaacgym/docs/api/python/struct_py.html')]
    hashes = {str(path): sha(path) for path in paths}
    out.mkdir()
    manifest = dict(run_status='RUNNING', experiment_id='engineering-oracle-contact-api',
                    run_id=out.name, engineering_only=True, pid=os.getpid(), phases=[],
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    input_sha256=hashes, wall_limit_seconds=300, storage_limit_bytes=128<<20,
                    actual_optimizer_steps=0)

    def save():
        manifest['wall_seconds'] = time.monotonic()-begin
        (out/'run_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')

    def verify():
        for name, digest in hashes.items():
            if sha(Path(name)) != digest:
                raise ValueError('engineering input drift')

    save()
    try:
        for pipeline, physics in (('gpu', 'gpu'), ('cpu', 'gpu'), ('cpu', 'cpu')):
            verify()
            gpu = None
            for index in (6, 4, 5, 2, 7):
                try:
                    gpu = admission(index)
                    break
                except RuntimeError:
                    pass
            if gpu is None:
                raise RuntimeError('no idle GPU')
            name = pipeline+'_pipeline_'+physics+'_physics'
            command = [PYTHON, '-u', str(ROOT/'scripts/smoke_oracle_contact_api.py'),
                       '--pipeline', pipeline, '--physics', physics, '--output', str(out/(name+'.json'))]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu['uuid'], OMP_NUM_THREADS='2',
                       MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1',
                       TORCH_EXTENSIONS_DIR=str(out/'cache/torch_extensions'), XDG_CACHE_HOME=str(out/'cache'))
            env['LD_LIBRARY_PATH']='/home2/wyy/miniconda3/envs/graspenv/lib:'+env.get('LD_LIBRARY_PATH', '')
            phase = dict(name=name, command=command, admission=gpu, run_status='RUNNING')
            manifest['phases'].append(phase)

            def spawned(child):
                phase.update(pid=child.pid, pgid=child.pid)
                save()

            def guard():
                if time.monotonic()-begin > 300 or bytes_in(out) > 128<<20:
                    raise RuntimeError('engineering budget')
                rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                                '--format=csv,noheader'], text=True)
                for row in rows.splitlines():
                    fields = [x.strip() for x in row.split(',')]
                    if len(fields) == 2 and fields[0] == gpu['uuid'] and fields[1] != str(phase.get('pid')):
                        raise RuntimeError('GPU contention; owned process only')
            save()
            print(json.dumps(dict(name=name, status='STARTED')), flush=True)
            try:
                run_owned_child(command, ROOT, env, out/(name+'.log'), guard, 90, spawned)
                verify()
                phase.update(run_status='COMPLETED', result_sha256=sha(out/(name+'.json')))
            except BaseException as error:
                phase.update(run_status='FAILED', error=repr(error))
                raise
            finally:
                save()
            print(json.dumps(dict(name=name, status='COMPLETED')), flush=True)
        summary = {phase['name']: json.loads((out/(phase['name']+'.json')).read_text())
                   for phase in manifest['phases']}
        result = dict(run_status='COMPLETED', engineering_only=True,
                      capabilities={name: {key: value.get(key) for key in
                                           ('attributed_contact_api_available', 'native_contact_api_error',
                                            'basic_contact_contract_pass', 'lambda_to_weight_ratio')}
                                    for name, value in summary.items()})
        (out/'results.json').write_text(json.dumps(result, indent=2)+'\n')
        manifest.update(run_status='COMPLETED', inputs_unchanged=True, bytes=bytes_in(out))
        print(json.dumps(result), flush=True)
    except BaseException as error:
        manifest.update(run_status='FAILED', error=repr(error))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
