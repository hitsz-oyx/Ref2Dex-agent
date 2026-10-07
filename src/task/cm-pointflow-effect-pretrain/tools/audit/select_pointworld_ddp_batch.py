#!/usr/bin/env python3
"""Bounded three-rank real-data throughput selection; never opens the test split."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

TASK = Path(__file__).resolve().parents[2]
REPO = TASK.parents[2]


def memory_snapshot():
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu',
                                    '--format=csv,noheader,nounits'], text=True)
    return {int(v[0]): dict(used_mib=int(v[1]), total_mib=int(v[2]), utilization=int(v[3]))
            for row in rows.splitlines() if (v := row.split(','))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--stats', type=Path, required=True)
    p.add_argument('--init-weights', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpus', default='0,1,2')
    p.add_argument('--batches', default='16,32,64')
    p.add_argument('--updates', type=int, default=45)
    a = p.parse_args()
    gpus = [int(x) for x in a.gpus.split(',')]
    if len(gpus)!=3 or len(set(gpus))!=3 or a.updates<20:
        p.error('requires three unique assigned GPUs and at least20updates')
    a.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    results = []
    for batch in map(int, a.batches.split(',')):
        snapshot = memory_snapshot()
        if any(snapshot[gpu]['used_mib']>64 for gpu in gpus):
            raise RuntimeError('assigned GPU is occupied; do not preempt another process')
        config = json.loads((TASK/'configs/pointworld_temporal_wm24.json').read_text())
        config.update(seed=219, microbatch=batch, accumulation=3, updates=a.updates,
                      learning_rate=1e-4, warmup_updates=5, workers=4, validation_microbatch=2,
                      group_seconds=300)
        config_path = a.output/f'batch{batch}.json'
        config_path.write_text(json.dumps(config,indent=2)+'\n')
        out = a.output/f'batch{batch}'
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpus, TMPDIR=str(REPO/'tmp'),
                   TRITON_CACHE_DIR=str(REPO/'tmp/triton-pointworld-action-ddp-20261007'),
                   OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='2')
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=3',
                   str(TASK/'tools/run/train_oakink2_pointworld_ddp.py'), '--data', str(a.data.resolve()),
                   '--stats', str(a.stats.resolve()), '--config', str(config_path.resolve()),
                   '--output', str(out.resolve()), '--arm', 'action', '--init-weights', str(a.init_weights.resolve()),
                   '--fused-hilbert', '--benchmark', '--deadline', str(time.time()+300)]
        print(json.dumps(dict(status='BENCHMARK', batch_per_rank=batch, command=command)),flush=True)
        samples = []
        with (a.output/f'batch{batch}.log').open('w') as log:
            proc = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
            while proc.poll() is None:
                samples.append(dict(time=time.time(), gpus=memory_snapshot()))
                if time.time()-started>900:
                    proc.terminate()
                    proc.wait(timeout=30)
                    raise TimeoutError('selection exceeded15minute engineering cap')
                time.sleep(2)
        result = dict(batch_per_rank=batch, returncode=proc.returncode, utilization_samples=samples)
        if proc.returncode==0 and (out/'train.jsonl').exists() and len((out/'train.jsonl').read_text().splitlines())>15:
            rows = [json.loads(x) for x in (out/'train.jsonl').read_text().splitlines()]
            measured = rows[15:]
            result.update(updates=len(rows), median_seconds=statistics.median(x['seconds'] for x in measured),
                          peak_reserved_mib=max(x['peak_reserved_mib'] for x in rows),
                          peak_allocated_mib=max(x['peak_allocated_mib'] for x in rows),
                          median_input_seconds=statistics.median(x['input_seconds'] for x in measured),
                          rank_parameters_identical=json.loads((out/'result.json').read_text())['rank_parameters_identical'])
            result['windows_per_second']=batch*3/result['median_seconds']
            result['eligible']=(len(rows)==a.updates and result['rank_parameters_identical'] and
                                result['peak_reserved_mib']<.75*min(snapshot[g]['total_mib'] for g in gpus))
            # Utilization is retained as sampled evidence, including startup;
            # throughput and allocator peak are the selection criteria.
        else:
            result['eligible']=False
        results.append(result)
        (a.output/'selection.json').write_text(json.dumps(dict(results=results),indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='utilization_samples'}),flush=True)
    eligible = [x for x in results if x['eligible']]
    if not eligible:
        raise RuntimeError('no safe finite three-rank configuration completed')
    fastest = max(x['windows_per_second'] for x in eligible)
    selected = min((x for x in eligible if x['windows_per_second']>=.95*fastest), key=lambda x:x['batch_per_rank'])
    summary = dict(results=results, selected_batch_per_rank=selected['batch_per_rank'],
                   selection_rule='smallest batch within5%of fastest safe throughput; peak reserve<75%VRAM',
                   elapsed_seconds=time.time()-started, engineering_only=True)
    (a.output/'selection.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='results'}),flush=True)


if __name__=='__main__':
    main()
