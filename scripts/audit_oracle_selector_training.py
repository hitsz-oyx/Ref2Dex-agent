"""Read-only frozen-selector fit audit; no optimization or simulator runs."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    from src.task.CmResidual.oracle_features import ARMS, OracleQ, masked

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    packet_path = args.fit / 'feature_packet.pt'
    model_path = args.fit / 'selector.pt'
    packets = torch.load(packet_path, map_location='cpu', weights_only=False)
    saved = torch.load(model_path, map_location='cpu', weights_only=False)
    mean, std = saved['mean'].cuda(), saved['std'].cuda()
    report = dict(
        kind='saved_artifact_diagnostic', new_optimizer_updates=0,
        device=torch.cuda.get_device_name(0),
        source_sha256={str(p): digest(p) for p in (packet_path, model_path)},
        evaluation_scope='r1 global-option heldout collection, NOT r2 deployed utility',
        scenes={}, arms={})

    for scene, packet in packets.items():
        labels = packet['labels'].astype(np.float64)
        best = labels.max(axis=0)
        counts = labels.sum(axis=0)
        motion = packet['motion']
        report['scenes'][scene] = dict(
            samples=int(labels.size), distinct_contexts=int(labels.shape[1]),
            positives=int(labels.sum()), positive_fraction=float(labels.mean()),
            per_option_positives=labels.sum(axis=1).astype(int).tolist(),
            per_motion={str(int(m)): dict(
                samples=int(labels[:, motion == m].size),
                positives=int(labels[:, motion == m].sum()),
                reachable_contexts=int(best[motion == m].sum()),
                mixed_outcome_contexts=int(((counts[motion == m] > 0) &
                                           (counts[motion == m] < labels.shape[0])).sum()))
                for m in np.unique(motion)},
            best_candidate_successes=int(best.sum()),
            baseline_successes=int(labels[0].sum()),
            mixed_outcome_contexts=int(((counts > 0) & (counts < labels.shape[0])).sum()))

    for arm in ARMS:
        net = OracleQ(saved['input_dim']).cuda()
        net.load_state_dict(saved['models'][arm]['model'])
        net.eval()
        history = np.asarray(saved['losses'][arm], dtype=np.float64)
        arm_report = dict(
            updates=int(saved['models'][arm]['updates']),
            minibatch_bce_means={
                'first100': float(history[:100].mean()),
                'steps701_800': float(history[700:800].mean()),
                'steps1301_1400': float(history[1300:1400].mean()),
                'last100': float(history[-100:].mean()),
                'last': float(history[-1])})
        for scene, packet in packets.items():
            raw = torch.from_numpy(packet['features'].reshape(-1, saved['input_dim'])).cuda()
            inputs = masked(((raw - mean) / std).clamp(-10, 10), arm,
                            saved['common_dim'], saved['effect_dim'])
            y = packet['labels'].reshape(-1).astype(np.float64)
            with torch.no_grad():
                logits = torch.cat([net(batch) for batch in inputs.split(128)])
                bce = float(torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, torch.from_numpy(y.astype(np.float32)).cuda()))
                values = logits.cpu().numpy().astype(np.float64)
                probs = logits.sigmoid().cpu().numpy().astype(np.float64)
            positive, negative = values[y == 1], values[y == 0]
            auc = None
            if len(positive) and len(negative):
                diff = positive[:, None] - negative[None, :]
                auc = float(((diff > 0) + 0.5 * (diff == 0)).mean())
            labels = packet['labels']
            choices = values.reshape(labels.shape).argmax(axis=0)
            selected = labels[choices, np.arange(labels.shape[1])]
            arm_report[scene] = dict(
                full_dataset_bce=bce, brier=float(((probs - y) ** 2).mean()),
                accuracy_at_half=float(((values >= 0) == (y == 1)).mean()),
                positive_recall=float((positive >= 0).mean()) if len(positive) else None,
                negative_recall=float((negative < 0).mean()) if len(negative) else None,
                auc=auc, selected_query_successes=int(selected.sum()),
                per_motion_selected_query_successes={str(int(m)): int(selected[packet['motion'] == m].sum())
                    for m in np.unique(packet['motion'])})
        report['arms'][arm] = arm_report
        del net

    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
