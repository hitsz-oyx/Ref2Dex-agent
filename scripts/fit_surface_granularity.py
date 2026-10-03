"""Fixed factorial probe; identical 64-query supervised targets in every arm."""
import argparse
import copy
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.task.CmResidual.surface_granularity_prior import GranularityPrior, VARIANTS, encode_granularity, granularity_gates
from src.task.CmResidual.surface_motion_prior import metrics
from scripts.run_contact_response_probe import sha


def main():
    p = argparse.ArgumentParser(); p.add_argument('--data', type=Path, required=True); p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    out = a.output.resolve(); assert ROOT in out.parents and not out.exists() and torch.cuda.is_available()
    torch.set_num_threads(2); torch.manual_seed(3903)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0'); template = copy.deepcopy(GranularityPrior().state_dict())
    provenance = json.loads((a.data/'provenance.json').read_text()); data = {}; parents = {}
    for group in provenance['selection']:
        assert sha(a.data/(group+'.npz')) == provenance['selected_packet_sha256'][group]
        with np.load(a.data/(group+'.npz')) as f: raw = {key: f[key] for key in f.files}
        for aggregation in ('mean', 'detail'):
            x, y = encode_granularity(raw, aggregation)
            data[group, aggregation] = (torch.from_numpy(x).to(device), torch.from_numpy(y[:, :64]).to(device))
        parents[group] = [row['parent'] for row in provenance['selection'][group]]
    out.mkdir(); reports = {}; predictions = {}; training = {}; baselines = {}
    schedule = torch.randint(2048, (1500, 32), generator=torch.Generator().manual_seed(3904))
    for hand in ('mano', 'inspire'):
        for queries, aggregation in VARIANTS:
            name = '{}_{}_{}'.format(hand, queries, aggregation); start = time.monotonic()
            model = GranularityPrior().to(device); model.load_state_dict(template)
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
            x, target = data[hand+'_train', aggregation]; losses = []
            for step, indices in enumerate(schedule):
                ids = indices.to(device); optimizer.zero_grad(set_to_none=True)
                prediction = model(x[ids, :queries])[:, :64]
                loss = (prediction-target[ids]).square().mean(); assert torch.isfinite(loss)
                loss.backward(); grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 10); assert torch.isfinite(grad)
                optimizer.step(); losses.append(float(loss.detach()))
                if (step+1)%500 == 0: print(json.dumps(dict(arm=name, step=step+1, loss=losses[-1])), flush=True)
            torch.cuda.synchronize(); fit_seconds = time.monotonic()-start
            state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            change = max(float((state[key]-template[key]).abs().max()) for key in state); assert change > 0
            torch.save(dict(state=state, initial=template, optimizer=optimizer.state_dict(), batch_schedule=schedule,
                            steps=1500, windows=2048, queries=queries, aggregation=aggregation,
                            hand=hand, losses=losses, maximum_parameter_change=change), out/(name+'.pt'))
            reports[name] = {}
            with torch.no_grad():
                for group in ('mano_eval', 'inspire_eval'):
                    vx, vy = data[group, aggregation]; chunks = []
                    for begin in range(0, len(vx), 32): chunks.append(model(vx[begin:begin+32, :queries])[:, :64].cpu())
                    pred = torch.cat(chunks).numpy(); predictions[name+'__'+group] = pred
                    reports[name][group] = metrics(pred, vy.cpu().numpy(), parents[group])
            training[name] = dict(steps=1500, windows=2048, window_draws=48000, supervised_point_targets=3072000,
                                  context_point_draws=48000*queries, parameters=sum(p.numel() for p in model.parameters()),
                                  fit_seconds=fit_seconds, maximum_parameter_change=change)
            print(json.dumps(dict(arm=name, complete=True, epe_mm=reports[name][hand+'_eval']['parent_epe_mm'], fit_seconds=fit_seconds)), flush=True)
    for group in ('mano_eval', 'inspire_eval'):
        x, y = data[group, 'mean']; y = y.cpu().numpy()
        baselines[group] = dict(zero=metrics(np.zeros_like(y), y, parents[group]),
                                persistence=metrics(x[:, :64, -3:].cpu().numpy(), y, parents[group]))
    gains, gates, label = granularity_gates(reports)
    np.savez(out/'predictions.npz', **predictions)
    result = dict(run_status='COMPLETED', experiment_id='P-20261003-cm-granularity', label=label,
                  gains=gains, gates=gates, reports=reports, baselines=baselines, training=training,
                  actual_optimizer_updates=12000, groups=provenance['groups'], eval_checkpoint_selection=False,
                  equal_update_budget=True, equal_target_budget=True, flops_matched=False,
                  realized_future_hand_input=True, policy_utility_unmeasured=True, pure_hand_effect_unidentified=True)
    (out/'results.json').write_text(json.dumps(result, indent=2)+'\n'); print(json.dumps(dict(label=label, gains=gains)), flush=True)


if __name__ == '__main__': main()
