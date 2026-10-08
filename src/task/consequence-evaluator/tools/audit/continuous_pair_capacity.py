"""Bounded read-only audit of all eligible episode pairs in labeled rollouts.

Compare event, physical-state and H-matched capacity with selected coverage.
No threshold changes, labels, dataset promotion or model computation.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import subprocess

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(TASK/'src'))
from consequence_evaluator.contracts import EPISODE_SCHEMA, K, is_within
from consequence_evaluator.data import sha
from consequence_evaluator.supervision import RULE, local_event, physical_trace, states_match


def capacity(windows, check=lambda: None):
    buckets = {}
    for window in windows:
        key = tuple(window[k] for k in ('split','task','expert','motion','phase'))
        buckets.setdefault(key, []).append(window)
    totals = {name: {split: set() for split in ('train','val','test')}
              for name in ('event','physical','history')}
    strata = []
    for key, group in sorted(buckets.items()):
        found = {name: set() for name in totals}
        for good_event, bad_event in (('maintained_hold','unrecovered_drop'),('lift_achieved','grasp_lost')):
            good = [w for w in group if w['event'] == good_event]
            bad = [w for w in group if w['event'] == bad_event]
            heights = np.asarray([w['initial_relative_height'] for w in bad])
            for a in good:
                check()
                found['event'].update(tuple(sorted((a['episode'],b['episode'])))
                                      for b in bad if a['episode'] != b['episode'])
                for index in np.flatnonzero(np.abs(heights-a['initial_relative_height']) <= .01):
                    b = bad[index]
                    check()
                    if a['episode'] == b['episode']:
                        continue
                    pair = tuple(sorted((a['episode'],b['episode'])))
                    if pair in found['history']:
                        continue
                    # states_match checks H whenever it is present, even with
                    # require_history=False. Omit it only for this diagnostic.
                    physical_a = {k:a[k] for k in ('object_pose','hand_keypoints')}
                    physical_b = {k:b[k] for k in physical_a}
                    if not states_match(physical_a,physical_b):
                        continue
                    found['physical'].add(pair)
                    if states_match(a,b,require_history=True):
                        found['history'].add(pair)
        for name in totals:
            totals[name][key[0]].update(found[name])
        strata.append(dict(zip(('split','task','expert','motion','phase'),key),
                           **{name+'_episode_pairs':len(v) for name,v in found.items()}))
    return dict(totals={name:{split:len(v) for split,v in groups.items()}
                        for name,groups in totals.items()},strata=strata)


def audit(source, output, seconds=600, *, all_ticks=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    owned = ROOT/'outputs/consequence-evaluator'
    if not is_within(source,owned) or not is_within(output,owned) or output.exists():
        raise ValueError('owned input and fresh owned output required')
    if not 1 <= seconds <= 600:
        raise ValueError('audit budget must be in1..600 seconds')
    started = time.monotonic()
    frozen = {}
    def pin(path, expected=None):
        path = Path(path).resolve()
        actual = sha(path)
        if expected is not None and actual != expected:
            raise ValueError('audit input identity mismatch: '+str(path))
        frozen[str(path)] = actual
    def check():
        if time.monotonic()-started >= seconds:
            raise TimeoutError('fixed capacity audit deadline')
    pin(source/'manifest.json')
    manifest = json.loads((source/'manifest.json').read_text())
    if (manifest.get('schema') != EPISODE_SCHEMA or manifest.get('status') != 'COMPLETED'
            or manifest.get('rollout_kind') != 'continuous' or manifest.get('label_rule') != RULE
            or not manifest.get('label_report_sha256')):
        raise ValueError('completed current-rule labeled continuous input required')
    pin(source/'label_report.json',manifest['label_report_sha256'])
    selected = json.loads((source/'label_report.json').read_text())['unique_episode_pair_groups']
    for path in (Path(__file__),TASK/'src/consequence_evaluator/contracts.py',
                 TASK/'src/consequence_evaluator/supervision.py'):
        pin(path)
    output.mkdir()
    report = dict(status='RUNNING',diagnostic_only=True,training_allowed=False,
                  cpu_reason='existing file/label statistics; no model computation',
                  git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  window_selection='all_known_control_ticks' if all_ticks else 'labeled_window_ticks',
                  rule=RULE,sources=frozen,observed_selected_unique_episode_pairs=selected)
    try:
        windows = []
        for record in manifest['episodes']:
            check()
            path, sidecar = (source/record['path']).resolve(), Path(record['diagnostics']).resolve()
            if not is_within(path,owned) or not is_within(sidecar,owned):
                raise ValueError('owned episode and physical diagnostic inputs required')
            pin(path,record['sha256']); pin(sidecar,record['diagnostics_sha256'])
            with np.load(path,allow_pickle=False) as packet, np.load(sidecar,allow_pickle=False) as diagnostic:
                trace = physical_trace(packet,diagnostic)
                h, pose, hand, phase = (packet[k] for k in ('history','object_pose','hand_keypoints','phase'))
                known = packet['plan_known']
                ticks = range(len(packet['action'])-K+1) if all_ticks else record['window_ticks']
                for tick in ticks:
                    check()
                    if not known[tick]:
                        if all_ticks:
                            continue
                        raise ValueError('selected window has unknown residual plan')
                    event = local_event(trace,str(phase[tick]),tick)
                    if event:
                        windows.append(dict(episode=record['episode'],tick=tick,event=event,
                            split=record['split'],task=record['task'],expert=record['expert'],motion=record['motion'],
                            phase=str(phase[tick]),initial_relative_height=float(trace['height'][tick]),
                            history=h[tick].copy(),object_pose=pose[tick].copy(),hand_keypoints=hand[tick].copy()))
                    if len(windows) > 20000:
                        raise ValueError('at most20000 event windows per bounded audit')
        report.update(capacity(windows,check),event_windows=len(windows))
        for path, expected in frozen.items():
            if sha(path) != expected:
                raise RuntimeError('audit source/input drift: '+path)
        report['status'] = 'COMPLETED'
    except BaseException as error:
        report.update(status='TIMED_OUT' if isinstance(error,TimeoutError) else 'FAILED',error=repr(error))
        raise
    finally:
        report['elapsed_s'] = time.monotonic()-started
        (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds',type=int,default=600)
    parser.add_argument('--all-ticks',action='store_true',help='diagnose cadence at every known full-window start')
    args = parser.parse_args()
    result = audit(args.source,args.output,args.seconds,all_ticks=args.all_ticks)
    print(json.dumps({k:v for k,v in result.items() if k not in ('sources','strata')},indent=2))
