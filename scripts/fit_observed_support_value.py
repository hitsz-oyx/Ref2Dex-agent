"""Matched current-observation value fits on fresh fixed-policy cohorts."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts.run_contact_response_probe import sha
from src.task.CmResidual.observed_support_value import (
    SCHEMA, VARIANTS, read_panel, features, initialized_model, phase_template, analyze)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.directory.resolve(), args.output.resolve()
    assert ROOT in out.parents and not out.exists() and torch.cuda.is_available()
    start = time.monotonic()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    manifest = json.loads((root / 'run_manifest.json').read_text())
    assert manifest['experiment_id'] == 'P-20261002-observed-support-task-value'
    cp = torch.load(manifest['panel_checkpoints']['597']['path'], map_location='cpu', weights_only=False)
    states = [cp['variants'][k]['actor'] for k in ('cm', 'state_only', 'none')]
    assert cp['updates'] == 0 and all(all(torch.equal(states[0][k], s[k]) for k in states[0]) for s in states[1:])
    fit, test = read_panel(root / 's597'), read_panel(root / 's598')
    assert fit['eligible_rows'] >= 10000 and test['eligible_rows'] >= 1024 and test['episodes'] >= 32
    mean = fit['extra'].numpy().mean(0)
    std = fit['extra'].numpy().std(0).clip(.001)
    models = {k: initialized_model().cuda().train() for k in VARIANTS}
    optimizers = {k: torch.optim.Adam(models[k].parameters(), lr=3e-4, foreach=False) for k in VARIANTS}
    xs = {k: torch.from_numpy(features(fit, mean, std, k)).cuda() for k in VARIANTS}
    labels = fit['target'].cuda()
    generator = torch.Generator(device='cpu').manual_seed(3542)
    last = {}
    for step in range(1500):
        indices = torch.randint(len(labels), (4096,), generator=generator).cuda()
        for k in VARIANTS:
            loss = torch.nn.functional.binary_cross_entropy(models[k](xs[k][indices]).flatten(), labels[indices])
            optimizers[k].zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(models[k].parameters(), 10)
            assert torch.isfinite(norm)
            optimizers[k].step()
            last[k] = float(loss.detach())
        if step % 500 == 499:
            print(json.dumps(dict(updates_each=step + 1, loss=last)), flush=True)
    parameters, predictions, errors = {}, {}, {}
    with torch.no_grad():
        for k in VARIANTS:
            model = models[k].eval()
            x = features(test, mean, std, k)
            predictions[k] = model(torch.from_numpy(x).cuda()).flatten().cpu().numpy()
            state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
            parameters[k] = state
            z = x.astype(np.float64)
            for layer in ('0', '2'):
                z = np.maximum(z @ state[layer + '.weight'].numpy().astype(np.float64).T + state[layer + '.bias'].numpy(), 0)
            z = z @ state['4.weight'].numpy().astype(np.float64).T + state['4.bias'].numpy()
            pred = (1 / (1 + np.exp(-z.clip(-700, 700)))).ravel()
            errors[k] = float(np.abs(pred - predictions[k]).max())
            assert errors[k] <= 2e-5
    table = phase_template(fit)
    predictions['motion_time'] = table[test['motion'].numpy(), test['timebin'].numpy()]
    report = analyze(test, predictions)
    out.mkdir()
    torch.save(dict(schema=SCHEMA, parameters=parameters, extra_mean=mean, extra_std=std,
                    phase_template=table, updates_each=1500, init_seed=3541, batch_seed=3542), out / 'models.pt')
    torch.save(dict(predictions=predictions,
                    rows={k: test[k] for k in ('target', 'cluster', 'tick', 'environment')},
                    extra=test['extra'], current=test['current']), out / 'predictions.pt')
    report.update(run_status='COMPLETED', fit_rows=fit['eligible_rows'], test_rows=test['eligible_rows'],
                  test_episodes=test['episodes'], actual_optimizer_steps=3000,
                  full_numpy_forward_maximum=errors, last_loss=last,
                  model_sha256=sha(out / 'models.pt'), prediction_sha256=sha(out / 'predictions.pt'),
                  current_observed_information_only=True, no_actor_or_cm_updates=True,
                  wall_seconds=time.monotonic() - start)
    (out / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
