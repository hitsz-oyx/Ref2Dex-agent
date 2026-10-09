"""Inference-only ACT horizon/overlap audit and native dispatch replay.

Uses frozen checkpoint normalization, episode-separated existing arrays, and
recorded native controls. No training, simulator, future policy input or Y.
"""
import argparse
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(TASK/'src'))
from consequence_evaluator.action_chunk import (  # noqa: E402
    ACTION_CHUNK_SCHEMA, EXECUTED_ACTION_SEMANTICS, HistoryStandardizer,
    NativeActionChunkProposal, _sha, _within, validate_action_episode)


def norm_stats(error):
    error = np.asarray(error, dtype=np.float64)
    groups = {'native18': error, 'wrist_translation_mm': error[...,:3]*1000,
              'wrist_rotation_rad': error[...,3:6]*np.pi,
              'active_fingers_normalized': error[...,[6,8,10,12,14,15]]}
    result = {}
    for name, values in groups.items():
        norms = np.linalg.norm(values, axis=-1).reshape(-1)
        result[name] = dict(mean_l2=float(norms.mean()), p95_l2=float(np.quantile(norms,.95)),
                            max_l2=float(norms.max()))
    return result


def horizon_metrics(prediction, target):
    error = np.asarray(prediction, dtype=np.float64)-target
    return dict(windows=len(error), mse=float(np.mean(error**2)),
                horizon_mse=np.mean(error**2,axis=(0,2)).tolist(),
                horizon_mae=np.mean(np.abs(error),axis=(0,2)).tolist(),
                horizon_bias=np.mean(error,axis=0).tolist(),
                first8_mse=float(np.mean(error[:,:8]**2)),
                last16_mse=float(np.mean(error[:,8:]**2)))


def overlap_metrics(predictions, ticks, *, period=8, target=None):
    index = {int(t):i for i,t in enumerate(ticks)}
    pairs = [(i,index[int(t)+period]) for i,t in enumerate(ticks) if int(t)+period in index]
    if not pairs:
        return None
    error = np.stack([predictions[i,period:]-predictions[j,:24-period] for i,j in pairs])
    result = dict(pairs=len(pairs), overlap=norm_stats(error),
                  boundary_disagreement=norm_stats(error[:,0]))
    early = [n for n,(i,j) in enumerate(pairs) if int(ticks[j])<=120]
    if early:
        result['first120_boundary_disagreement'] = norm_stats(error[early,0])
    if target is not None:
        # The natural change between successive teacher controls is separate
        # from disagreement between predictions for the SAME absolute time.
        natural = np.stack([target[int(ticks[j])]-target[int(ticks[j])-1] for i,j in pairs])
        result['teacher_natural_boundary_change'] = norm_stats(natural)
    return result


def audit_behavior(path):
    with path.open('rb') as stream:
        packet = pickle.load(stream)
    if (packet.get('engineering_only') is not True or packet.get('group_mode') !=
            'synchronous_same_process_act_behavior'):
        raise ValueError('ACT behavior engineering packet required')
    mode = packet['action_chunk_mode']
    period = packet['action_chunk_replan_period']
    actions = np.asarray(packet['actions'])
    chunks = np.asarray(packet['proposal_chunks'])
    ticks = np.asarray(packet.get('proposal_query_ticks', np.arange(len(chunks))*period))
    assert np.array_equal(ticks,np.arange(0,len(actions),period)), 'query schedule drift'
    assert chunks.shape == (len(ticks),4,24,18), 'proposal shape drift'
    reproduced = []
    active = []
    for tick in range(len(actions)):
        covering = np.where((ticks<=tick)&(ticks+24>tick))[0]
        if mode in ('open_loop24','receding8'):
            covering = covering[-1:]
        values = np.stack([chunks[i,1,tick-ticks[i]] for i in covering])
        weights = np.exp(-float(packet.get('temporal_aggregation_decay',.01))*np.arange(len(covering)))
        reproduced.append(np.sum(values * (weights/weights.sum())[:,None],axis=0))
        active.append(len(covering))
    reproduced = np.asarray(reproduced)
    delta = float(np.max(np.abs(reproduced-actions[:,1])))
    assert delta<2e-7, 'executed native ACT differs from absolute-time aggregation'
    requested = packet.get('requested_controls')
    requested_delta = None if requested is None else float(np.max(np.abs(requested-actions)))
    if requested_delta is not None:
        assert requested_delta == 0, 'requested/actual native control mismatch'
    if 'active_chunk_counts' in packet:
        assert np.array_equal(active,packet['active_chunk_counts']), 'covering count drift'
    if 'proposal_input_history' in packet:
        assert np.array_equal(packet['proposal_input_history'],packet['history'][ticks]), 'future/stale query history'
    checkpoint = Path(packet['action_chunk_checkpoint'])
    assert _sha(checkpoint) == packet['action_chunk_checkpoint_sha256'], 'checkpoint drift'
    boundaries = np.arange(8,min(121,len(actions)),8)
    control_changes = {}
    for role,index in [('act',1),('teacher',0)]:
        control_changes[role] = dict(
            first120_all_steps=norm_stats(np.diff(actions[:121,index],axis=0)),
            first120_eight_step_boundaries=norm_stats(actions[boundaries,index]-actions[boundaries-1,index]))
    return dict(path=str(path), sha256=_sha(path), mode=mode, steps=len(actions),
                proposal_queries=len(ticks), max_covering_chunks=max(active),
                aggregation_native_max_abs_delta=delta,
                requested_native_max_abs_delta=requested_delta,
                repeated_act_actions_exact=bool(np.array_equal(actions[:,1],actions[:,3])),
                early_done=bool(packet['done'][:-1].any()),
                role_max_lift_m=packet['role_max_lift_m'], role_outcomes=packet['role_outcomes'],
                pair_drift=packet['pair_drift'], replay_identity=packet['replay_identity'],
                online_overlap8=overlap_metrics(chunks[:,1],ticks),
                control_changes=control_changes,
                engineering_only=True, training_allowed=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=2)
    p.add_argument('--teacher-packet',type=Path,action='append',default=[])
    p.add_argument('--behavior-packet',type=Path,action='append',default=[])
    p.add_argument('--require-held45',action='store_true',help='red-capable symptom assertion for behavior packets')
    a = p.parse_args()
    a.output = a.output.resolve()
    inputs = a.teacher_packet+a.behavior_packet+([a.checkpoint] if a.checkpoint else [])
    if (a.output.exists() or not _within(a.output,ROOT/'outputs/consequence-evaluator')
            or any(not _within(x,ROOT/'outputs/consequence-evaluator') for x in inputs)):
        p.error('new output and task-owned inputs required')
    a.output.mkdir(parents=True)
    started = time.monotonic()
    result = dict(schema='ref2dex.act-execution-audit.v1', engineering_only=True,
                  training_allowed=False, git_commit=subprocess.check_output(
                      ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  source_sha256={str(x):_sha(x) for x in inputs})
    result['behavior'] = [audit_behavior(x.resolve()) for x in a.behavior_packet]
    if a.checkpoint:
        import torch
        occupied = subprocess.check_output(['nvidia-smi','-i',str(a.gpu),
            '--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if occupied:
            raise RuntimeError('GPU occupied: '+occupied)
        torch.set_num_threads(2)
        device = 'cuda:%d' % a.gpu
        payload = torch.load(a.checkpoint,map_location='cpu')
        assert payload['schema']==ACTION_CHUNK_SCHEMA and payload['chunk']==24
        assert payload['executed_action_semantics']==EXECUTED_ACTION_SEMANTICS
        manifest_path = a.checkpoint.parent/'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        assert manifest['checkpoint_sha256']==_sha(a.checkpoint)
        std = HistoryStandardizer(**{key:np.asarray(value,dtype='float32') if key!='clip' else value
                                    for key,value in payload['history_standardizer'].items()})
        model = NativeActionChunkProposal(payload['history_dim'],width=payload['width'],
                                         layers=payload['layers']).to(device).eval()
        model.load_state_dict(payload['state_dict'],strict=True)
        result.update(checkpoint=str(a.checkpoint.resolve()), checkpoint_sha256=_sha(a.checkpoint),
                      normalization_clip=std.clip, device=device)
        result['source_sha256'][str(manifest_path)] = _sha(manifest_path)
        source_manifest_path = Path(manifest['source_manifest'])
        assert _sha(source_manifest_path)==manifest['source_manifest_sha256']
        source = json.loads(source_manifest_path.read_text())
        episodes = []
        for record in source['episodes']:
            if record['episode'] not in manifest['selected_episodes']:
                continue
            path = source_manifest_path.parent/record['path']
            assert _sha(path)==manifest['source_files'][str(path)]
            with np.load(path,allow_pickle=False) as data:
                h,act = validate_action_episode(data['history'],data['action'])
            split = 'val' if record['episode'] in manifest['val_episodes'] else 'train'
            episodes.append((split,record['episode'],h,act))
        for path in a.teacher_packet:
            with path.open('rb') as stream:
                packet = pickle.load(stream)
            assert packet['role_names'][0]=='reactive_teacher' and packet['engineering_only']
            episodes.append((path.parent.name,'reactive_teacher',packet['history'][:,0],packet['actions'][:,0]))
        grouped = {}
        details = []
        for split,name,h,act in episodes:
            if time.monotonic()-started>180:
                raise TimeoutError('bounded inference audit deadline')
            ticks = np.arange(0,len(act)-24+1)
            predictions = []
            with torch.no_grad():
                for first in range(0,len(ticks),256):
                    value = torch.as_tensor(std.transform(h[ticks[first:first+256]]),device=device)
                    predictions.append(model(value).cpu().numpy())
            predictions = np.concatenate(predictions)
            targets = np.stack([act[t:t+24] for t in ticks])
            grouped.setdefault(split,[]).append((predictions,targets))
            detail = dict(split=split,episode=name,horizon=horizon_metrics(predictions,targets),
                          overlap8=overlap_metrics(predictions,ticks,target=act))
            details.append(detail)
            print(json.dumps(dict(episode=name,split=split,mse=detail['horizon']['mse'])),flush=True)
        result['episodes'] = details
        result['groups'] = {key:horizon_metrics(np.concatenate([v[0] for v in rows]),
                                               np.concatenate([v[1] for v in rows])) for key,rows in grouped.items()}
        try:
            import matplotlib
        except ImportError:
            print('Plot skipped: this runtime has no matplotlib; numeric audit is retained.',flush=True)
        else:
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig,axes = plt.subplots(1,2,figsize=(11,4))
            for key,metrics in result['groups'].items():
                label = key if key in ('train','val') else 'teacher '+key.rsplit('-',1)[-1]
                axes[0].plot(np.arange(24),metrics['horizon_mse'],label=label)
                ds=[d for d in details if d['split']==key]
                values=[d['overlap8']['boundary_disagreement']['wrist_translation_mm']['mean_l2'] for d in ds]
                axes[1].scatter([label]*len(values),values,s=15)
            axes[0].axvspan(-.5,7.5,color='gray',alpha=.15)
            axes[0].set(xlabel='Chunk horizon k',ylabel='Native action MSE',yscale='log')
            axes[0].legend(fontsize=8)
            axes[1].set(ylabel='Wrist disagreement (mm)')
            fig.tight_layout(); fig.savefig(a.output/'horizon-overlap.png',dpi=160); plt.close(fig)
    result['elapsed_s'] = time.monotonic()-started
    (a.output/'audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    if a.require_held45:
        assert result['behavior'], 'behavior packet required for grasp assertion'
        assert all(x['role_outcomes']['act_chunk']['maximum_held_frames']>=45
                   for x in result['behavior']), 'ACT grasp regression: no stable hold45'
    print(json.dumps(dict(output=str(a.output),elapsed_s=result['elapsed_s'])),flush=True)


if __name__=='__main__':
    main()
