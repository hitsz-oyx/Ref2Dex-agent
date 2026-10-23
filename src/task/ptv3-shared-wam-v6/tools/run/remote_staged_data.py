"""NAS-only source/split-stratified conversion with immutable batch receipts.

Plans use mother identities, never object identity. The existing geometric
admission function is unchanged. A plan is not a training admission receipt.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path[:0] = [str(TASK/'src'), str(TASK/'tools/audit')]
NAS = Path('/mnt/ugreen_nas/storage/oyx_storage')
PROBE = NAS/'remote-rigid-throughput-20261023-r1'
SEED = 20261023
GIB = 2**30


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value, immutable=False):
    path = Path(path)
    if immutable and path.exists():
        raise FileExistsError(path)
    pending = path.with_name(path.name+'.part')
    with pending.open('w') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    pending.replace(path);path.chmod(0o664)


def stratified_keys(rows):
    groups = defaultdict(list)
    seen = {}
    for row in rows:
        mother = (row['source_id'], row.get('mother_identity', row['source_sequence']))
        if mother in seen:
            raise ValueError('duplicate mother or cross-split mother: '+str(mother))
        seen[mother] = row['split']
        groups[(row['source_id'], row['split'])].append(row)
    first = set()
    for group in groups.values():
        ordered = sorted(group, key=lambda r: hashlib.sha256(
            ('%s/%s/%s' % (SEED, r['source_id'], r.get('mother_identity', r['source_sequence']))).encode()).hexdigest())
        first.update(r['key'] for r in ordered[:(len(ordered)+1)//2])
    return first


def paths_environment(out):
    if NAS not in out.resolve().parents:
        raise ValueError('all artifacts must be on NAS')
    short = ROOT/'tmp/rh'
    if short.is_symlink():
        if short.resolve() != out:
            raise ValueError('short NAS link belongs to another run')
    else:
        short.parent.mkdir(exist_ok=True);short.symlink_to(out, target_is_directory=True)
    scratch = short/'scratch';scratch.mkdir(exist_ok=True)
    if len(os.fsencode(str(scratch/'pymp-12345678/listener-12345678'))) >= 108:
        raise ValueError('short checkout required for AF_UNIX')
    os.environ.update(TMPDIR=str(scratch), PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        XDG_CACHE_HOME=str(out/'cache'), TORCH_HOME=str(out/'cache/torch'),
        CUDA_CACHE_PATH=str(out/'cache/cuda'), MPLCONFIGDIR=str(out/'cache/mpl'))


def plan(out, workers):
    import remote_rigid_throughput as remote
    baseline = json.loads(gzip.decompress((out/'baseline.json.gz').read_bytes()))
    refs = {r['key']: r for r in baseline['rows']}
    rows = [json.loads(line) for line in (PROBE/'input-rows.jsonl').open()]
    patches = read(PROBE/'input-overrides.json')['rows']
    for patch in patches:
        if sha(patch['path']) != patch['sha256']:
            raise ValueError('explicit input alias drift')
        row = next(r for r in rows if r['key'] == patch['key'])
        row[patch['role']] = patch['path'];row['missing'] = []
    if len(rows) != len(refs) or set(refs) != {r['key'] for r in rows}:
        raise ValueError('baseline census differs')
    for row in rows:
        ref = refs[row['key']]
        for key in ('source_id', 'sequence_id', 'split'):
            if row[key] != ref[key]:
                raise ValueError('baseline identity drift: '+row['key'])
        if row['missing']:
            raise ValueError('input unavailable: '+row['key'])
        row['source_sequence'] = ref['source_sequence']
        # Relative episode names repeat under different source roots. The
        # resolved native directory/payload is the actual mother identity.
        row['mother_identity'] = str(Path(row['job'].get('sequence_folder')
            if row['source_id'] < 3 else row['job']['payload']).resolve())
        row['baseline'] = ref
    first = stratified_keys(rows)
    by_source = []
    for source in sorted({r['source_id'] for r in rows}):
        entry = dict(source_id=source)
        for stage in ('half', 'remaining'):
            chosen = [r for r in rows if r['source_id'] == source and (r['key'] in first) == (stage == 'half')]
            entry[stage] = {split: dict(mothers=sum(r['split'] == split for r in chosen),
                baseline_windows=sum(r['baseline']['admitted_windows'] for r in chosen if r['split'] == split))
                for split in ('train', 'dev')}
            entry[stage]['conversion_exact_bytes'] = sum(r['conversion_bytes'] for r in chosen)
        by_source.append(entry)
    exact = {s: sum(r['conversion_bytes'] for r in rows if (r['key'] in first) == (s == 'half')) for s in ('half','remaining')}
    free = shutil.disk_usage(out).free
    # Metadata, indices, receipts, audit/panel, plus retained checkpoints.
    peak = sum(exact.values()) + 64*GIB + 136*GIB + workers*max(r['conversion_bytes'] for r in rows)
    if free-peak < 30*GIB:
        raise RuntimeError('full campaign NAS budget does not fit')
    result = dict(schema='ref2dex.remote-staged-plan.v1', seed=SEED,
        definition='ceil(50%) mothers independently within every source x inherited split, SHA256(seed/source/resolved-native-mother-URI) order',
        baseline_sha256=sha(out/'baseline.json.gz'), baseline_manifest_sha256=baseline['manifest_sha256'],
        baseline_plan_sha256=read(PROBE/'input-audit.json')['frozen_plan_sha256'],
        registry_sha256=read(PROBE/'input-audit.json')['registry_sha256'],
        half_keys=sorted(first), rows=rows, by_source=by_source, conversion_exact_bytes=exact,
        conversion_estimate_seconds_at_12_2_MB_s={s:n/12.2e6 for s,n in exact.items()},
        workers=workers, device='cuda:0', batch_mothers=64, floor_bytes=30*GIB,
        artifact_budget_bytes=peak, free_before_bytes=free,
        provenance='baseline windows are frozen local counts, not measured remote conversion results',
        remote_input_version='NAS-native packs; HOT3D segment version differs, native continuity must be rechecked',
        source8_dev_missing=True, training_allowed=False)
    write(out/'plan.json', result, immutable=True)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','half_keys')}, indent=2), flush=True)


def worker_init(dest, device):
    import remote_rigid_throughput as remote
    remote.configure_worker(dest, device)


def convert_one(arg):
    import remote_rigid_throughput as remote
    return remote.convert(arg)


def execute(out, stage):
    plan = read(out/'plan.json');first = set(plan['half_keys'])
    rows = [r for r in plan['rows'] if (r['key'] in first) == (stage == 'half')]
    # Interleave sources within every bounded batch; selection already frozen.
    rows.sort(key=lambda r: hashlib.sha256((str(SEED)+r['key']).encode()).hexdigest())
    dest = out/stage;dest.mkdir(exist_ok=True)
    for name in ('hands/sequences','contacts','events','receipts','windows','batches'):
        (dest/name).mkdir(parents=True, exist_ok=True)
    for name, value in [('inventory.json',dict(plan_sha256=plan['baseline_plan_sha256'])),
            ('contract.json',dict(identity=hashlib.sha256((plan['baseline_plan_sha256']+'full-rigid-v1').encode()).hexdigest(), training_allowed=False))]:
        if not (dest/name).exists():write(dest/name, value, immutable=True)
    attempt = str(time.time_ns())
    import torch
    started = time.monotonic()
    write(dest/('run-'+attempt+'.json'), dict(command=sys.argv,git_commit=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(), plan_sha256=sha(out/'plan.json'),
        stage=stage,workers=plan['workers'],python=sys.version,executable=sys.executable,
        torch=torch.__version__,cuda=torch.version.cuda,host=os.uname().nodename,
        environment={k:os.environ.get(k) for k in ('TMPDIR','XDG_CACHE_HOME','CUDA_CACHE_PATH','CONDA_PREFIX')},
        automatic_restart=False,output=str(dest),budget_bytes=plan['artifact_budget_bytes']),immutable=True)
    total_bytes=0;total_windows=0
    pool = ProcessPoolExecutor(plan['workers'], mp_context=mp.get_context('spawn'),
        initializer=worker_init,initargs=(str(dest),'cuda:0'))
    try:
        for offset in range(0,len(rows),64):
            if (out/'STOP').exists() or (dest/'STOP').exists():raise InterruptedError('owner STOP')
            free=shutil.disk_usage(out).free
            if free < plan['floor_bytes']+sum(r['conversion_bytes'] for r in rows[offset:offset+64]):
                raise RuntimeError('NAS space floor')
            results=list(pool.map(convert_one,[(str(dest),r) for r in rows[offset:offset+64]],chunksize=1))
            actual=[r['result'] for r in results]
            failures=[r for r in actual if r['status']!='ADMITTED']
            total_bytes+=sum(r.get('conversion_actual_bytes',0) for r in actual)
            total_windows+=sum(r['admitted_windows'] for r in actual)
            elapsed=time.monotonic()-started
            progress=dict(status='RUNNING',stage=stage,processed=offset+len(actual),total=len(rows),
                conversion_bytes=total_bytes,admitted_windows=total_windows,elapsed_s=elapsed,
                eta_seconds=elapsed*(len(rows)-offset-len(actual))/(offset+len(actual)),
                free_bytes=shutil.disk_usage(out).free,failures=failures)
            write(dest/'batches'/('batch-%05d-%s.json'%(offset//64,attempt)),
                dict(progress,receipts=[dict(key=r['key'],sha256=sha(dest/'receipts'/(r['key']+'.json')),
                    window_shard_sha256=r.get('window_shard_sha256')) for r in actual]),immutable=True)
            write(dest/'progress.json',progress);print(json.dumps(progress),flush=True)
            if failures:raise RuntimeError('admission/conversion failed; no automatic restart')
            if time.monotonic()-started > 6*3600:raise TimeoutError('6h conversion stage budget')
        pool.shutdown(wait=True)
        write(dest/'progress.json',dict(progress,status='CONVERSION_COMPLETED'))
    except BaseException as exc:
        write(dest/('failure-'+attempt+'.json'),dict(error=repr(exc),automatic_restart=False,at=time.time()),immutable=True)
        for child in list(pool._processes.values()):child.terminate()
        pool.shutdown(wait=True);raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=('plan','half','remaining'),required=True)
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args();out=args.output.resolve()
    if not 1<=args.workers<=12:raise ValueError('1..12 workers')
    paths_environment(out)
    if args.mode=='plan':plan(out,args.workers)
    else:execute(out,args.mode)


if __name__=='__main__':main()
