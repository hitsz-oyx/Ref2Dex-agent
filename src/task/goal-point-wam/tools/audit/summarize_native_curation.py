#!/usr/bin/env python3
"""Summarize an existing curation census and frozen inference; no model calls."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def summarize(root):
    census = json.loads((root / 'result.json').read_text())
    names = ('oakink2', 'grab', 'arctic')
    hands, transitions = {}, {}
    with np.load(root / 'model-hand-r1/val.npz') as data:
        val = {key: data[key] for key in data.files}
    valid = val['valid'].any(axis=(1, 3))
    distance = np.minimum(val['target_now_per_hand_mm'], val['target_future_per_hand_mm'])
    for sid, name in enumerate(names):
        selected = (val['source'] == sid) & (val['metrics'][:, 0] < 2)
        groups = {}
        for label, condition in (
            ('all', valid), ('near', valid & (distance <= 50)),
            ('far', valid & (distance > 100)),
            ('middle', valid & (distance > 50) & (distance <= 100)),
        ):
            mask = selected[:, None] & condition
            groups[label] = dict(
                hands=int(mask.sum()),
                real_mm=float(val['real_per_hand_mm'][mask].mean()) if mask.any() else None,
                zero_mm=float(val['zero_per_hand_mm'][mask].mean()) if mask.any() else None,
                error_share=float(val['real_per_hand_mm'][mask].sum()
                                  / val['real_per_hand_mm'][selected[:, None] & valid].sum()),
            )
        hands[name] = dict(windows=int(selected.sum()), groups=groups)
        with np.load(root / (name + '.npz')) as data:
            metric, category = data['metrics'], data['categories']
        counts = np.bincount(category, minlength=3)
        probabilities = np.array([.6, .2, .2]) * (counts > 0)
        probabilities /= probabilities.sum()
        weights = probabilities[category] / counts[category]
        static = metric[:, 0] < 2
        flags = dict(
            current_far_50=metric[:, 4] > 50,
            static_release_like=static & (metric[:, 4] <= 50) & (metric[:, 6] > 100),
            static_near_both_ends=static & (metric[:, 4] <= 50) & (metric[:, 6] <= 50),
            far_to_near=(metric[:, 4] > 50) & (metric[:, 5] <= 50),
        )
        transitions[name] = {
            key: dict(raw_count=int(mask.sum()), sampled_fraction=float(weights[mask].sum()),
                      fraction_of_sampled_static=float(weights[mask & static].sum() / weights[static].sum()))
            for key, mask in flags.items()
        }
        metric = val['metrics']
        release = selected & (metric[:, 4] <= 50) & (metric[:, 6] > 100)
        transitions[name]['val_static_release_like'] = dict(
            count=int(release.sum()), static_count=int(selected.sum()),
            real_mm=float(val['real_mm'][release].mean()) if release.any() else None,
            zero_mm=float(val['zero_mm'][release].mean()) if release.any() else None,
        )
    with np.load(root / 'model-r1/val.npz') as original:
        difference = abs(val['real_mm'] - original['real_mm'])
    transitions['rerun'] = dict(max_window_error_difference_mm=float(difference.max()),
                                mean_window_error_difference_mm=float(difference.mean()))
    (root / 'per_hand_summary.json').write_text(json.dumps(hands, indent=2))
    (root / 'transition_summary.json').write_text(json.dumps(transitions, indent=2))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    labels = ['OakInk2', 'GRAB', 'ARCTIC']
    x = np.arange(3)
    ax = axes[0, 0]
    for offset, key, label in [(-.18, 'raw', 'Raw TRAIN windows'), (.18, 'sampled', 'Expected training draws')]:
        values = [100 * census['sources'][name][key]['goal_lt2'] for name in names]
        bars = ax.bar(x + offset, values, .36, label=label)
        ax.bar_label(bars, fmt='%.1f%%', padding=3)
    ax.set(xticks=x, xticklabels=labels, ylim=(0, 100), ylabel='Percent within source',
           title='Endpoint goal < 2 mm: sampling changes the picture')
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    source_weights = [census['sources'][name]['source_weight'] for name in names]
    keys = ['static_goal_near', 'static_goal_far_moving_other_near', 'static_goal_idle_away']
    values = [100 * sum(w * census['sources'][name]['sampled'][key]
                       for w, name in zip(source_weights, names)) for key in keys]
    total = 100 * sum(w * census['sources'][name]['sampled']['goal_lt2']
                      for w, name in zip(source_weights, names))
    values.append(total - sum(values))
    bars = ax.barh(['Target near', 'Target far; another object near', 'Idle away', 'Other static-goal cases'],
                   values, color=['#4c956c', '#e09f3e', '#9e9e9e', '#577590'])
    ax.bar_label(bars, fmt='%.2f%%', padding=3)
    ax.set(xlim=(0, 32), xlabel='Percent of ALL expected mixed training draws',
           title=f'Static goals total {total:.2f}% (exclusive groups)')
    ax.invert_yaxis()

    ax = axes[1, 0]
    values = [100 * transitions[name]['static_release_like']['fraction_of_sampled_static'] for name in names]
    bars = ax.bar(x, values, color='#577590')
    ax.bar_label(bars, fmt='%.1f%%', padding=3)
    ax.set(xticks=x, xticklabels=labels, ylim=(0, 100), ylabel='Percent of sampled static-goal windows',
           title='Static goal, but hands depart: near now -> far at end')
    ax.text(.02, .94, 'Distance <= 50 mm now; > 100 mm at step 24\nGeometric proxy, not a semantic release label',
            transform=ax.transAxes, va='top', fontsize=8)

    ax = axes[1, 1]
    for offset, key, label in [(-.18, 'near', 'Hand near target sometime'), (.18, 'far', 'Hand always > 100 mm away')]:
        values = [hands[name]['groups'][key]['real_mm'] for name in names]
        bars = ax.bar(x + offset, values, .36, label=label)
        ax.bar_label(bars, fmt='%.1f', padding=3)
    ax.set(xticks=x, xticklabels=labels, ylim=(0, 240), ylabel='Per-hand trajectory EPE (mm)',
           title='Frozen VAL static goals: error by hand proximity')
    ax.legend(fontsize=8)
    fig.suptitle('Local three-domain curation audit | 4,378,478 TRAIN windows; frozen 1,536-window VAL', fontsize=13)
    fig.savefig(root / 'curation_summary.png', dpi=160)
    plt.close(fig)
    print(json.dumps(dict(status='COMPLETED', output=str(root / 'curation_summary.png'))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    summarize(parser.parse_args().root)
