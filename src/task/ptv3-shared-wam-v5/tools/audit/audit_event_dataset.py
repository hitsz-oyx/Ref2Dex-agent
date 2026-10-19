"""Read real candidate events with spawn workers and preserve base sampling."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

TASK=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(TASK/'src'))
from ptv3_shared_wam_v5 import paths
from ptv3_shared_wam_v5.data import V5Windows, collate
from ptv3_shared_wam_v5.event_data import EventV5Windows
from ptv3_shared_wam_v5.events import event_features, EVENT_ENCODING
from ptv3_shared_wam_v5.cache_contract import metadata_identity
from ptv3_shared_wam_v5.storage import read_json, load_array


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','events','stats','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--git-commit',required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);started=time.time()
    data=EventV5Windows(a.manifest,'train',a.events)
    base=V5Windows(a.manifest,'train')
    sampled=data.draw(32768,177,10050,0)
    assert [i for i,_ in sampled]==base.draw(32768,177,10050,0)
    sample_index=np.asarray([i for i,_ in sampled],np.int64)
    drops=np.asarray([d for _,d in sampled],bool)
    ids=data.index[sample_index]
    order=np.argsort(ids[:,0],kind='stable');counts=np.bincount(ids[:,0],minlength=len(data.meta['sequences']))
    starts=np.r_[0,np.cumsum(counts)];sources={}
    for seq_id in np.flatnonzero(counts):
        row=data.meta['sequences'][seq_id];folder=Path(data.event_rows[row['key']]['path'])
        at=order[starts[seq_id]:starts[seq_id+1]];_,slots,ticks=ids[at].T
        kind=load_array(folder/'event_type.npy',mmap_mode='r')[ticks,slots]
        delta=load_array(folder/'event_delta_steps.npy',mmap_mode='r')[ticks,slots]
        valid=load_array(folder/'event_valid.npy',mmap_mode='r')[ticks,slots]
        view=sources.setdefault(str(row['source_id']),Counter())
        view.update(samples=len(at),unknown=int((~valid).sum()),none=int((valid&(kind==0)).sum()),
                    contact=int((valid&(kind==1)).sum()),release=int((valid&(kind==2)).sum()),
                    far=int((valid&(delta>24)).sum()),available_after_dropout=int((valid&~drops[at]).sum()))
    # Actual payload reads, including one parent-equal sample/source.
    pairs=[]
    for sid,pool in data.pools.items():
        index=int(pool[0]); old=base[index]; new=data[index]
        assert all(np.array_equal(v,new[k]) for k,v in old.items())
        pairs.append(dict(source_id=int(sid),sample_id=new['sample_id'].tolist(),
                          event_type=int(new['event_type']),event_delta_steps=float(new['event_delta_steps']),
                          event_valid=bool(new['event_valid'])))
    loader=DataLoader(data,batch_sampler=[sampled[:128]],num_workers=2,multiprocessing_context='spawn',
                      timeout=120,collate_fn=collate)
    batch=next(iter(loader)); features,valid=event_features(batch)
    assert features.shape==(128,5) and torch.isfinite(features).all()
    # A reviewable candidate recipe, intentionally blocked by the production
    # launcher's existing admission guard until the continuation is designed.
    recipe=dict(data.meta)
    recipe.update(training_allowed=False,event_candidate_only=True,
        parent_manifest=str(a.manifest.resolve()),stats=str(a.stats.resolve()),
        event_annotations=dict(manifest=str(a.events.resolve()),identity=metadata_identity(a.events),
                               encoding=EVENT_ENCODING,event_dropout=.15,
                               reader='EventV5Windows(parent_manifest,split,event_manifest)'),
        limitation='candidate only; generic training admission intentionally disabled')
    (a.output/'candidate-recipe.json').write_text(json.dumps(recipe,indent=2)+'\n')
    (a.output/'stats.json').symlink_to(a.stats.resolve())
    result=dict(status='COMPLETED',git_commit=a.git_commit,python=sys.version,torch=torch.__version__,
        device='CPU annotation/data reader only; no model inference',elapsed_s=time.time()-started,
        event_sequences=len(data.event_rows),train_windows=len(data),
        dev_windows=int(len(load_array(data.meta['indices']['dev'],mmap_mode='r'))),
        original_source_stratum_and_window_choices_exact=True,original_payload_exact_on_nine_source_panel=True,
        spawn_workers=2,real_payload_batch=128,valid_conditions_in_batch=int(valid.sum()),
        weighted_draw_seed=177,weighted_draw_step=10050,weighted_draw_rank=0,weighted_draw_count=len(sampled),
        weighted_draw_sources=sources,source_examples=pairs,training_started=False,
        normalization='original TRAIN-only K24 stats unchanged and symlinked',
        limitation='32768 inherited-mixture TRAIN draws estimate exposure, not population accuracy; no event effectiveness test')
    (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    for path in a.output.iterdir():
        if not path.is_symlink():path.chmod(0o664)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
