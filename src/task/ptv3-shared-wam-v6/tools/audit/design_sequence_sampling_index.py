#!/usr/bin/env python3
"""Offline composition option; NEVER wired into training or curated_sampling.py."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np

TASK=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(TASK/'src'))
from ptv3_shared_wam_v6.curated_sampling import CuratedTauSampler,load_manifest

PHASES=('approach','close','contact','hold','unknown')
TARGET=dict(zip(PHASES,(.2475,.2475,.2475,.2475,.01)))


class OfflineIndexOption:
    """One shared global mother queue, per-mother/phase time cursors and deficits.

    A fixed seeded circular mother permutation makes even wraparound batches
    duplicate-free. Distributed ranks would slice one global batch, not create
    independent rank-local mother queues. This is an audit-only prototype.
    """
    def __init__(self,rows,seed=170):
        self.rows=rows
        self.pools=defaultdict(lambda:defaultdict(list))
        for i,row in enumerate(rows):self.pools[int(row['sequence_id'])][row['phase']].append(i)
        self.order=np.random.default_rng(seed).permutation(sorted(self.pools))
        self.cursor=0;self.drawn=0;self.phase_counts=Counter();self.time_cursors=Counter()
        for phases in self.pools.values():
            for pool in phases.values():pool.sort(key=lambda i:(rows[i]['tick'],rows[i]['slot']))

    def draw(self,count):
        if count>len(self.order):raise ValueError('not enough mothers for a distinct global batch')
        chosen=self.order[(np.arange(count)+self.cursor)%len(self.order)]
        self.cursor=(self.cursor+count)%len(self.order)
        result=[]
        for seq in chosen:
            available=self.pools[int(seq)]
            phase=max(available,key=lambda p:(TARGET[p]*(self.drawn+1)-self.phase_counts[p],-PHASES.index(p)))
            pool=available[phase];key=(int(seq),phase)
            position=self.time_cursors[key]%len(pool)
            result.append(self.rows[pool[position]])
            self.time_cursors[key]+=1;self.phase_counts[phase]+=1;self.drawn+=1
        return result


def batch_stats(rows):
    grouped=defaultdict(list)
    for row in rows:grouped[int(row['sequence_id']),int(row['slot'])].append(int(row['tick']))
    near=0
    for ticks in grouped.values():
        ticks=np.sort(ticks)
        adjacent=np.diff(ticks)<24
        marked=np.zeros(len(ticks),bool);marked[:-1]|=adjacent;marked[1:]|=adjacent
        near+=int(marked.sum())
    n=len(rows)
    keys={(r['sequence_id'],r['slot'],r['tick']) for r in rows}
    return dict(unique_mothers=len({r['sequence_id'] for r in rows}),
        repeated_mother_fraction=1-len({r['sequence_id'] for r in rows})/n,
        exact_repeat_fraction=1-len(keys)/n, near_temporal_duplicate_fraction=near/n)


def simulate(rows,draw,steps,batch):
    exposure=Counter();phase=Counter();source=Counter();mothers=set();objects=set();stats=[]
    # Trace the most populated mother, which illustrates actual time rotation.
    counts=Counter(r['sequence_id'] for r in rows)
    trace_seq=counts.most_common(1)[0][0];trace=[]
    for step in range(steps):
        chosen=draw(step);assert len(chosen)==batch
        stats.append(batch_stats(chosen))
        for row in chosen:
            exposure[row['sequence_id'],row['slot'],row['tick']]+=1
            phase[row['phase']]+=1;source[str(row['source_id'])]+=1
            mothers.add(row['sequence_id']);objects.add((row['source_id'],row['object']))
            if row['sequence_id']==trace_seq and len(trace)<16:
                trace.append(dict(step=step,tick=row['tick'],slot=row['slot'],phase=row['phase']))
    values=np.array([exposure[r['sequence_id'],r['slot'],r['tick']] for r in rows])
    return dict(global_batches=steps,global_batch=batch,total_draws=steps*batch,
        batch_expectation_estimates={k:float(np.mean([s[k] for s in stats])) for k in stats[0]},
        exposure_per_original_window=dict(mean=float(values.mean()),minimum=int(values.min()),
            p50=float(np.median(values)),p95=float(np.percentile(values,95)),maximum=int(values.max()),
            unexposed=int((values==0).sum())),
        phase_draw_fraction={p:phase[p]/(steps*batch) for p in PHASES},
        source_draw_fraction={s:n/(steps*batch) for s,n in sorted(source.items())},
        sampled_unique_mothers=len(mothers),sampled_unique_objects=len(objects),
        time_rotation_example=dict(sequence_id=int(trace_seq),first_occurrences=trace))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batches',type=int,default=256)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    manifest=load_manifest(a.manifest)
    rows=[r for r in manifest['windows'] if r.get('admitted') and r['split']=='train']
    old=CuratedTauSampler(manifest,seed=170,max_per_sequence=8)
    option=OfflineIndexOption(rows)
    # The current planned sampler is rank-local. Compare a world3/local96
    # global288 reference, matching the historical exposure accounting.
    def old_draw(step):
        return [r for rank in range(3) for r in old.draw(96,step=step,rank=rank)]
    before=simulate(rows,old_draw,a.batches,288)
    after=simulate(rows,lambda _:option.draw(288),a.batches,288)
    assert after['batch_expectation_estimates']['unique_mothers']==288
    assert before['exposure_per_original_window']['mean']==after['exposure_per_original_window']['mean']
    result=dict(schema='ref2dex.tau-sequence-index-design.v1',run_id='tau-contract-20261023-r1',
        classification='DATA_COMPOSITION_CHANGE_OPTION_ONLY',applied=False,training_updates=0,
        before_definition='existing unconnected CuratedTauSampler, independent rank draws, world3/local96; not the historical V6Windows sampler',
        after_definition='global distinct mother circular queue; per-mother/phase sorted time rotation; phase deficits',
        seed=170,phase_target=TARGET,reference_budget='256 global batches is an offline design statistic, not an authorized optimizer budget',
        composition=dict(windows=len(rows),unique_mothers=len(set(r['sequence_id'] for r in rows)),
            unique_objects=len(set((r['source_id'],r['object']) for r in rows)),
            phase_window_fraction={p:sum(r['phase']==p for r in rows)/len(rows) for p in PHASES},
            structural_near_duplicate_percent=manifest['effective_coverage']['curated_train']['near_duplicate_window_fraction']),
        before=before,after=after,
        independent_coverage_added=dict(mothers=0,objects=0),
        decision='DO_NOT_REORDER_OR_WIRE; keep as owner choice because independent mother/object pool is unchanged',
        tradeoffs=['removes source-uniform weighting; source proportions change substantially',
            'cycles pre-existing rows; cannot create new independent evidence',
            'fixed circular mother order is periodic, not new independent draws',
            'unknown rows remain eligible; stage names are teacher-derived composition labels, not causal runtime inputs'],
        index_design=dict(window_ids='int32; each existing row once',
            sequence_phase_offsets='int32 prefix offsets',time_cursor='one integer per sequence/phase',
            global_queue='one mother permutation/cursor shared across ranks; slice after global selection',
            expected_window_id_bytes=4*len(rows),
            writes='statistics only, no materialized training order or production sampler change'))
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(before=before,after=after,decision=result['decision']),indent=2))


if __name__=='__main__':main()
