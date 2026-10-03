"""Fixed scale/hand matrix, GPU training and inference, no policy calls."""
import argparse
import copy
import json
import sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.task.CmResidual.surface_motion_prior import SurfaceMotionPrior, SIZES, FLOW_COLUMNS, encode_raw, metrics, classify
from scripts.run_contact_response_probe import sha


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--data', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); out = args.output.resolve(); assert ROOT in out.parents and not out.exists() and torch.cuda.is_available()
    torch.set_num_threads(2); torch.manual_seed(3803)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0'); template = SurfaceMotionPrior().state_dict()
    provenance = json.loads((args.data/'provenance.json').read_text()); data = {}; features_np = {}; parents = {}
    for group in ('mano_train', 'inspire_train', 'mano_eval', 'inspire_eval'):
        assert sha(args.data/(group+'.npz')) == provenance['selected_packet_sha256'][group]
        with np.load(args.data/(group+'.npz')) as payload: raw = {key: payload[key] for key in payload.files}
        x, y = encode_raw(raw); features_np[group] = x
        data[group] = (torch.from_numpy(x).to(device), torch.from_numpy(y).to(device))
        parents[group] = [row['parent'] for row in provenance['selection'][group]]
    out.mkdir(); predictions = {}; reports = {}; records = {}; checkpoints = {}
    permutation = torch.from_numpy(np.random.default_rng(3805).permutation(7168)).to(device)
    template_cpu = {key: value.clone() for key, value in template.items()}

    @torch.no_grad()
    def evaluate(name, model, motion_off):
        model.eval(); report = {}
        for group in ('mano_eval', 'inspire_eval'):
            x, y = data[group]; chunks = []
            for start in range(0, len(x), 32):
                features = x[start:start+32].clone()
                if motion_off: features[..., list(FLOW_COLUMNS)] = 0
                chunks.append(model(features).cpu())
            predicted = torch.cat(chunks).numpy(); predictions[name+'__'+group] = predicted
            report[group] = metrics(predicted, y.cpu().numpy(), parents[group])
        reports[name] = report

    def train(name, group, size, steps, state, motion_off=False, shuffled=False):
        model = SurfaceMotionPrior().to(device); model.load_state_dict(state)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        generator = torch.Generator().manual_seed(3804)
        x, y = data[group]; losses = []; initial = copy.deepcopy(model.state_dict())
        schedule = torch.randint(size, (steps, 32), generator=generator)
        for step in range(steps):
            ids = schedule[step].to(device)
            features = x[ids].clone()
            if motion_off: features[..., list(FLOW_COLUMNS)] = 0
            target = y[permutation[ids]] if shuffled else y[ids]
            model.train(); optimizer.zero_grad(set_to_none=True)
            loss = (model(features)-target).square().mean()
            assert torch.isfinite(loss)
            loss.backward(); gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            assert torch.isfinite(gradient)
            optimizer.step(); losses.append(float(loss.detach()))
            if (step+1) % 500 == 0: print(json.dumps(dict(arm=name, step=step+1, loss=losses[-1])), flush=True)
        maximum_change = max(float((value-initial[key]).abs().max()) for key, value in model.state_dict().items())
        assert maximum_change > 0
        checkpoint = dict(state={key: value.cpu() for key, value in model.state_dict().items()},
                          initial={key: value.cpu() for key, value in initial.items()}, optimizer=optimizer.state_dict(),
                          generator_state=generator.get_state(), steps=steps, group=group, size=size,
                          motion_off=motion_off, shuffled=shuffled, target_permutation=permutation.cpu() if shuffled else None,
                          common_initial=template_cpu, maximum_parameter_change=maximum_change,
                          batch_schedule=schedule, losses=losses)
        torch.save(checkpoint, out/(name+'.pt')); checkpoints[name] = checkpoint['state']
        records[name] = {key: checkpoint[key] for key in ('steps', 'group', 'size', 'motion_off', 'shuffled', 'maximum_parameter_change')}
        bank = provenance['selection'][group][:size]
        records[name].update(window_draws=steps*32, mean_draws_per_bank_window=steps*32/size,
                             bank_parents=len(set(row['parent'] for row in bank)),bank_objects=len(set(row['object'] for row in bank)),
                             parameters=sum(parameter.numel() for parameter in model.parameters()))
        evaluate(name, model, motion_off)
        print(json.dumps(dict(arm=name, complete=True, inspire_epe_mm=reports[name]['inspire_eval']['parent_epe_mm'])), flush=True)

    for hand in ('mano', 'inspire'):
        for size in SIZES: train(hand+'_'+str(size), hand+'_train', size, 1500, template_cpu)
    train('mano_motion_off', 'mano_train', 7168, 1500, template_cpu, motion_off=True)
    train('mano_shuffled', 'mano_train', 7168, 1500, template_cpu, shuffled=True)
    for size in SIZES: train('adapt_mano_'+str(size), 'inspire_train', 256, 600, checkpoints['mano_'+str(size)])
    train('adapt_mano_motion_off', 'inspire_train', 256, 600, checkpoints['mano_motion_off'], motion_off=True)
    train('adapt_mano_shuffled', 'inspire_train', 256, 600, checkpoints['mano_shuffled'])
    train('adapt_scratch', 'inspire_train', 256, 600, template_cpu)
    baselines = {}
    for group in ('mano_eval', 'inspire_eval'):
        x, y = data[group]
        baselines[group] = dict(zero=metrics(np.zeros_like(y.cpu().numpy()), y.cpu().numpy(), parents[group]),
                                persistence=metrics(x[..., 19:22].cpu().numpy(), y.cpu().numpy(), parents[group]))
    selected = {name: values['mano_eval' if name.startswith('mano_') and name not in ('mano_motion_off', 'mano_shuffled') else 'inspire_eval'] for name, values in reports.items()}
    gates, label = classify(selected, baselines['inspire_eval']['persistence']['parent_epe_mm'])
    np.savez(out/'predictions.npz', **predictions)
    result = dict(run_status='COMPLETED', label=label, gates=gates, reports=reports, baselines=baselines, training=records,
                  actual_optimizer_updates=sum(r['steps'] for r in records.values()), data_groups=provenance['groups'],
                  experiment_id='P-20261003-cm-scale-cross-hand', real_motion_offline=True,
                  command_conditioned_policy_utility_unmeasured=True, pure_hand_morphology_effect_unidentified=True,
                  source_inspire_physics_legacy=True, eval_used_for_checkpoint_selection=False)
    (out/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(label=label, gates=gates, updates=result['actual_optimizer_updates'])), flush=True)


if __name__ == '__main__': main()
