#!/usr/bin/env python3
"""Plot an existing horizon audit without re-running model inference."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    data = json.loads(a.audit.read_text())
    x = np.array(data['future_seconds'])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout='constrained')
    for key, label in [('real', 'WAM, true goal'), ('zero', 'Stationary hand'), ('velocity', 'Constant velocity')]:
        y = np.mean([s['per_step_mm'][key] for s in data['sources'].values()], axis=0)
        axes[0].plot(x, y, label=label)
    for name, source in data['sources'].items():
        axes[1].plot(x, source['per_step_mm']['real'], label=name)
    for ax in axes:
        ax.axvline(8/30, color='gray', linestyle='--', alpha=.6)
        ax.set(xlabel='Future time (s)', ylabel='Hand point EPE (mm)', xlim=(0, .8), ylim=(0, None))
        ax.grid(alpha=.2)
        ax.legend(frameon=False)
    axes[0].set_title('Source macro average, fixed1536 windows')
    axes[1].set_title('WAM by source, final10000')
    fig.savefig(a.output, dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
