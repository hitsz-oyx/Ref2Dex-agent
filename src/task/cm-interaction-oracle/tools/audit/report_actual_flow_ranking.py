#!/usr/bin/env python3
"""Export parameter costs, prediction ties and a standalone offline figure."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT),str(ROOT/'src/task/cm-interaction-oracle/src'),str(ROOT/'src/task/cm-interaction-oracle/tools/run'),str(Path(__file__).parent)]
from package_rolling_oracle_asset import Snapshot, sha
from probe_actual_flow_ranking import load_data
from actual_flow_ranking import ARMS, utility


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--asset-run',type=Path,required=True)
    ap.add_argument('--dense-run',type=Path,required=True)
    ap.add_argument('--run-dir',type=Path,required=True)
    args = ap.parse_args(); torch.set_num_threads(2)
    if (args.run_dir/'report.json').exists() or (args.run_dir/'ranking.png').exists():
        raise ValueError('refusing existing report artifacts')
    snapshot = Snapshot(); data = load_data(args.asset_run,args.dense_run,snapshot)
    manifest = snapshot.read(args.run_dir/'manifest.json')
    saved = snapshot.read(args.run_dir/'models.pt',manifest['models_sha256'])
    result = snapshot.read(args.run_dir/'result.json',manifest['result_sha256'])
    gt = utility(data['y']).reshape(-1,7).numpy()
    ties = {}
    for arm in ARMS:
        pr = utility(saved['predictions'][arm]).reshape(-1,7).numpy()
        all_ties = strict_ties = 0
        for i in range(7):
            for j in range(i+1,7):
                tied = pr[:,i] == pr[:,j]
                all_ties += int(tied.sum())
                strict_ties += int((tied & (gt[:,i] != gt[:,j])).sum())
        ties[arm] = dict(prediction_tied_pairs=all_ties,prediction_ties_on_GT_strict_pairs=strict_ties)
    params = {}
    for arm in ARMS:
        head = saved['folds'][0]['heads'][arm]['stats']['parameters']
        physical = saved['folds'][0]['physical']['stats']['parameters'] if arm in ('Bottleneck','Hybrid') else 0
        params[arm] = dict(Y_head=head,consequence_head=physical,inference_pipeline=head+physical,
                           Y_training_fits=4,consequence_training_fits=16 if physical else 0,
                           note='Shared E/I fits counted for each dependent pipeline; not independent retraining.')
    out = dict(status='COMPLETED',parameters=params,tie_counts=ties,
               input_sha256=snapshot.hashes,report_tool_sha256=sha(Path(__file__).resolve()),
               scope='CPU saved-artifact reporting; no training/physics')
    (args.run_dir/'report.json').write_text(json.dumps(out,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    accuracy = [result['metrics'][a]['pairwise_accuracy'] for a in ARMS]
    shuffled = [result['shuffle_metrics'][a]['pairwise_accuracy'] for a in ARMS]
    x = np.arange(len(ARMS)); fig,ax = plt.subplots(figsize=(8,4))
    ax.bar(x-.18,accuracy,.36,label='Held-anchor OOF')
    ax.bar(x+.18,shuffled,.36,label='Within-panel flow shuffle')
    ax.axhline(.70,color='black',ls='--',lw=1,label='Frozen screening gate')
    ax.set_xticks(x);ax.set_xticklabels(ARMS);ax.set_ylim(0,1)
    ax.set_ylabel('GT-strict pair accuracy');ax.set_title('Ref14_3 actual-flow oracle: offline ranking')
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(args.run_dir/'ranking.png',dpi=180);plt.close(fig)
    print(json.dumps(dict(parameters=params,tie_counts=ties),indent=2))


if __name__ == '__main__':
    main()
