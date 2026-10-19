"""Check real checkpoint expansion; never update or overwrite a training run."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import torch

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK/'src'))
from ptv3_shared_wam_v5 import paths
from ptv3_shared_wam_v5.model import SharedWAMV5
from ptv3_shared_wam_v5.events import EVENT_ENCODING, initialize_event_model
from ptv3_shared_wam_v5.event_migration import expand_event_optimizer
from ptv3_shared_wam_v5.data import V5Windows, collate
from ptv3_shared_wam_v5.storage import read_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'manifest', 'stats', 'output'): p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--git-commit', required=True)
    p.add_argument('--device', default='cuda:1'); p.add_argument('--write-candidate', action='store_true')
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    if not a.device.startswith('cuda') or shutil.disk_usage(a.output).free < 20*2**30:
        raise ValueError('GPU inference and >=20GiB free disk required')
    torch.set_num_threads(2); torch.manual_seed(177); started = time.time()
    device = torch.device(a.device); torch.cuda.set_device(device)
    free, total = torch.cuda.mem_get_info(device)
    cap = 6*2**30
    if free < cap+8*2**30: raise ValueError('need6GiB allocator cap plus8GiB free headroom')
    torch.cuda.set_per_process_memory_fraction(cap/total, device)
    torch.backends.cuda.matmul.allow_tf32 = True
    if not a.checkpoint.is_file(): raise ValueError('published checkpoint required')
    a.checkpoint.chmod(0o664)  # owned derived checkpoint, never raw data
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    c = saved['config']; stats = read_json(a.stats)
    if c.get('event_conditioning'): raise ValueError('original unconditioned checkpoint required')
    base = SharedWAMV5(stats, c)
    base.load_state_dict(saved['model'], strict=True)
    target_config = dict(c, event_conditioning=True, event_encoding=EVENT_ENCODING)
    target = SharedWAMV5(stats, target_config)
    weights = initialize_event_model(target, saved['model'])
    expanded, optimizer = expand_event_optimizer(base, target, saved['optimizer'], c)
    old_names = dict(base.named_parameters()); new_names = dict(target.named_parameters())
    assert all(torch.equal(v, target.state_dict()[k]) for k, v in saved['model'].items())
    assert not target.event_goal.encoder[-1].weight.count_nonzero()
    assert not target.event_goal.encoder[-1].bias.count_nonzero()
    # States alias the original tensors, so neither values nor step placement
    # has been transformed during positional-ID remapping.
    assert {id(v) for v in expanded['state'].values()} == {id(v) for v in saved['optimizer']['state'].values()}
    original_steps = sorted({float(s['step']) for s in expanded['state'].values() if 'step' in s})
    if a.write_candidate:
        candidate = dict(saved)
        candidate.update(schema='ref2dex.wam-v5-event-migration-candidate.v1',
            config=target_config, model=target.state_dict(), optimizer=expanded,
            git_commit=a.git_commit, parent_git_commit=saved.get('git_commit'),
            parent_checkpoint=str(a.checkpoint.resolve()), selection={},
            event_migration=dict(weights=weights, optimizer=optimizer),
            training_allowed=False,
            limitation='requires new event recipe/data identity and explicit migration validation before training')
        pending = a.output/'event-expanded-candidate.pt.part'
        torch.save(candidate, pending); pending.rename(a.output/'event-expanded-candidate.pt')
    base.to(device).eval(); target.to(device).eval()
    data = V5Windows(a.manifest, 'train'); rows = []
    with torch.inference_mode():
        for sid, pool in data.pools.items():
            sample = data[int(pool[0])]
            b = {k: v.to(device) for k, v in collate([sample]).items()}
            b.update(event_type=torch.tensor([1], device=device),
                     event_delta_steps=torch.tensor([6.], device=device),
                     event_valid=torch.tensor([True], device=device), event_hz=torch.tensor([30], device=device))
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=c.get('amp', True)):
                before = base(b); repeat = base(b); after = target(b)
            delta = lambda a, z: float((a.float()-z.float()).norm(dim=-1).mean())*1000
            rows.append(dict(source_id=int(sid), sample_id=sample['sample_id'].tolist(),
                repeat_tau_mean_mm=delta(before['trajectory'], repeat['trajectory']),
                expansion_tau_mean_mm=delta(before['trajectory'], after['trajectory']),
                expansion_tau_max_mm=float((before['trajectory'].float()-after['trajectory'].float()).norm(dim=-1).max())*1000,
                contact_max_abs=float((before['interaction'][..., 0].float().sigmoid()-after['interaction'][..., 0].float().sigmoid()).abs().max())))
            del before, repeat, after, b
    report = dict(status='COMPLETED', checkpoint=str(a.checkpoint.resolve()), step=saved['step'],
        git_commit=a.git_commit, python=sys.version, torch=torch.__version__, device=str(device),
        elapsed_s=time.time()-started, weights=weights, optimizer=optimizer,
        original_weights_exact=True, original_optimizer_states_exact=True, original_optimizer_steps=original_steps,
        new_parameter_count=sum(new_names[n].numel() for n in set(new_names)-set(old_names)),
        rows=rows, peak_gpu_gib=torch.cuda.max_memory_allocated(device)/2**30,
        conditioning_training_started=False,
        limitation='GPU sparse reductions can vary on repeated baseline; zero-residual equality is exact at weights/residual level, CPU full-model tests and named next-update tests. This is not an event effectiveness test.')
    path=a.output/'report.json'; path.write_text(json.dumps(report, indent=2)+'\n'); path.chmod(0o664)
    print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
