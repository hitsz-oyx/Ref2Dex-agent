"""Matched position/displacement fit and train-only candidate retrieval."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(TASK / 'src'))
sys.path.insert(0, str(TASK / 'tools/run'))
from consequence_evaluator.contracts import is_within
from consequence_evaluator.data import sha
from consequence_evaluator.hand_execution import transform_future
from consequence_evaluator.proposal_history import condition, MeasuredHistoryToTau, SCHEMA
from train_history_to_tau import stats, encode, metrics


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def collect(root):
    root = Path(root).resolve()
    m = json.loads((root / 'manifest.json').read_text())
    if m.get('status') != 'COMPLETED' or m.get('mode') != 'retarget':
        raise ValueError('completed source retarget packet required')
    keys = ('object_pose', 'hand_keypoints', 'dof_position', 'dof_velocity', 'object_velocity', 'length')
    with np.load(root / 'trajectory.npz', allow_pickle=False) as source:
        data = {k: source[k] for k in keys}
    features = []; current = []; target = []; episode = []; tick_ids = []
    for env, length in enumerate(data['length'].tolist()):
        for tick in range(8, int(length) - 24 + 1, 8):
            sl = slice(tick-3, tick+1)
            value, hand = condition(data['object_pose'][sl, env], data['hand_keypoints'][sl, env],
                data['dof_position'][sl, env], data['dof_velocity'][sl, env], data['object_velocity'][sl, env])
            future = transform_future(data['object_pose'][tick, env], data['hand_keypoints'][tick+1:tick+25, env])
            features.append(value); current.append(hand); target.append(future)
            episode.append('%s:%d' % (m['seed'], env)); tick_ids.append(tick)
    arrays = dict(history=np.asarray(features, dtype='float32'), current=np.asarray(current, dtype='float32'),
        target=np.asarray(target, dtype='float32'), episode=np.asarray(episode), tick=np.asarray(tick_ids))
    if not all(np.isfinite(arrays[k]).all() for k in ('history', 'current', 'target')):
        raise ValueError('nonfinite source windows')
    return arrays, m, {str(root / n): sha(root / n) for n in ('manifest.json', 'trajectory.npz')}


def distribution(prediction, arrays):
    target = arrays['target']; error = np.sum((prediction-target)**2, -1).mean((1, 2))
    base_error = np.sum((arrays['current'][:, None]-target)**2, -1).mean((1, 2))
    rows = []
    for ep in np.unique(arrays['episode']):
        ids = arrays['episode'] == ep
        rows.append(dict(episode=str(ep), point_rmse_m=float(np.sqrt(error[ids].mean())),
            persistence_rmse_m=float(np.sqrt(base_error[ids].mean())), squared_error_sum=float(error[ids].sum()),
            max_current_root_distance_m=float(np.linalg.norm(arrays['current'][ids, 0], axis=-1).max())))
    rows.sort(key=lambda row: row['squared_error_sum'], reverse=True)
    return dict(**metrics(prediction, target), episode_median_rmse_m=float(np.median([r['point_rmse_m'] for r in rows])),
        worst_two_squared_error_share=float(sum(r['squared_error_sum'] for r in rows[:2]) / max(float(error.sum()), 1e-12)),
        worst_episodes=rows[:5])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('train', 'val', 'test'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--seed', type=int, default=296)
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--seconds', type=int, default=600)
    a = p.parse_args(); started = time.monotonic(); output = a.output.resolve()
    if output.exists() or not is_within(output, ROOT / 'outputs/consequence-evaluator'):
        raise ValueError('fresh task output required')
    if not 1 <= a.steps <= 1200 or not 1 <= a.seconds <= 600:
        raise ValueError('bounded matched fit required')
    before = subprocess.check_output(['nvidia-smi', '-i', str(a.gpu),
        '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'], text=True)
    util, used, total = map(int, before.strip().split(','))
    if util > 10 or used > 512 or total-used < 20480:
        raise RuntimeError('GPU not idle')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    torch.set_num_threads(2); torch.manual_seed(a.seed); torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device('cuda:0')
    arrays = {}; manifests = {}; hashes = {}
    for split in ('train', 'val', 'test'):
        arrays[split], manifests[split], source_hash = collect(getattr(a, split)); hashes.update(source_hash)
    if len({m['actor_sha256'] for m in manifests.values()}) != 1:
        raise ValueError('source actor mismatch')
    sets = [set(arrays[s]['episode'].tolist()) for s in ('train', 'val', 'test')]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError('source episodes overlap')
    for path in (Path(__file__).resolve(), TASK / 'src/consequence_evaluator/proposal_history.py',
                 TASK / 'src/consequence_evaluator/hand_execution.py', TASK / 'tools/run/train_history_to_tau.py'):
        hashes[str(path)] = sha(path)
    hs = stats(arrays['train']['history'], floor=.001)
    x = {s: torch.as_tensor(encode(v['history'], hs).clip(-10, 10), device=device) for s,v in arrays.items()}
    labels = {}; target_stats = {}
    for arm in ('absolute', 'displacement'):
        raw = arrays['train']['target']
        if arm == 'displacement':
            raw = raw - arrays['train']['current'][:, None]
        target_stats[arm] = stats(raw)
        labels[arm] = torch.as_tensor(encode(raw, target_stats[arm]), device=device)
    original = MeasuredHistoryToTau().to(device)
    models = {arm: copy.deepcopy(original) for arm in labels}
    initial = copy.deepcopy(original.state_dict())
    opt = {arm: torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4) for arm,model in models.items()}
    groups = [np.flatnonzero(arrays['train']['episode'] == ep) for ep in np.unique(arrays['train']['episode'])]
    rng = np.random.default_rng(a.seed); best = {arm: float('inf') for arm in labels}; chosen = {}
    output.mkdir(parents=True)
    torch.save(dict(schema=SCHEMA, model=initial, input_statistics=hs), output/'initial.pt')
    manifest = dict(schema=SCHEMA, status='RUNNING', git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        physical_gpu=a.gpu, gpu_before=before.strip(), seed=a.seed, input_sha256=hashes,
        steps_cap=a.steps, seconds_cap=a.seconds, matched_initialization=True, matched_batches=True,
        input_contract='Four measured states t-3:t: hand geometry/object SE3/finger q,dq/object velocity; no actor obs, future reference, clock, force, action or label',
        target_contract='Hand t+1:t+24 in current object frame; absolute versus displacement from current hand',
        history_floor=.001, normalized_input_clip=10, retrieval_k=8, retrieval_contract='train-only nearest history; one window per distinct episode; current-hand anchored displacement',
        test_used_for_selection=False, source_seeds={s:m['seed'] for s,m in manifests.items()})
    write(output/'manifest.json', manifest); history=[]

    def check():
        if time.monotonic()-started > a.seconds:
            raise TimeoutError('matched fit deadline')
        processes = subprocess.check_output(['nvidia-smi','-i',str(a.gpu),
            '--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
        for row in processes.strip().splitlines():
            pid, memory = map(int,row.split(','))
            if pid != os.getpid() and memory > 512:
                raise RuntimeError('foreign GPU work')

    def predict(arm, split):
        model=models[arm]; model.eval(); values=[]
        with torch.inference_mode():
            for begin in range(0, len(x[split]), 512):
                y=model(x[split][begin:begin+512]).cpu().numpy()
                y=y*target_stats[arm][1]+target_stats[arm][0]
                if arm == 'displacement':
                    y=y+arrays[split]['current'][begin:begin+512,None]
                values.append(y)
        prediction=np.concatenate(values)
        if not np.isfinite(prediction).all():
            raise ValueError('nonfinite model output')
        return prediction

    try:
        for step in range(1,a.steps+1):
            batch=np.asarray([rng.choice(groups[rng.integers(len(groups))]) for _ in range(256)])
            for arm,model in models.items():
                model.train(); opt[arm].zero_grad(set_to_none=True)
                loss=(model(x['train'][batch])-labels[arm][batch]).abs().mean()
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite fit')
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt[arm].step()
            if step == 1 or step%100 == 0 or step == a.steps:
                check(); row=dict(step=step,elapsed_s=time.monotonic()-started,val_l1={})
                for arm in models:
                    error=float(np.abs(predict(arm,'val')-arrays['val']['target']).mean())
                    row['val_l1'][arm]=error
                    if error < best[arm]:
                        best[arm]=error; chosen[arm]=step
                        torch.save(dict(schema=SCHEMA, model=models[arm].state_dict(), arm=arm,
                            statistics=dict(history=hs,target=target_stats[arm]),step=step,
                            input_contract=manifest['input_contract'], input_clip=10),output/(arm+'-best.pt'))
                row['gpu']=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),
                    '--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).strip()
                history.append(row);print(json.dumps(row),flush=True)
        predictions={}; results={}
        for arm in models:
            packet=torch.load(output/(arm+'-best.pt'),map_location=device,weights_only=False)
            models[arm].load_state_dict(packet['model'])
        train_displacement=torch.as_tensor(arrays['train']['target']-arrays['train']['current'][:,None],device=device)
        for split,data in arrays.items():
            check(); predictions[split]={arm:predict(arm,split) for arm in models}
            result={arm:distribution(y,data) for arm,y in predictions[split].items()}
            result['persistence']=metrics(np.repeat(data['current'][:,None],24,axis=1),data['target'])
            candidate_values=[]; neighbor_ids=[]; neighbor_distances=[]
            with torch.inference_mode():
                for begin in range(0,len(x[split]),128):
                    query=x[split][begin:begin+128]
                    distance=torch.cdist(query,x['train']).square()/query.shape[1]
                    episode_distance=[]; episode_row=[]
                    for ids in groups:
                        value,local=distance[:,ids].min(1)
                        episode_distance.append(value);episode_row.append(torch.as_tensor(ids,device=device)[local])
                    scores=torch.stack(episode_distance,1);source_ids=torch.stack(episode_row,1)
                    if split=='train':
                        same=data['episode'][begin:begin+128,None]==np.unique(arrays['train']['episode'])[None]
                        scores[torch.as_tensor(same,device=device)]=float('inf')
                    values,index=scores.topk(8,largest=False,sorted=True)
                    ids=source_ids.gather(1,index)
                    candidates=train_displacement[ids]+torch.as_tensor(data['current'][begin:begin+128,None,None],device=device)
                    candidate_values.append(candidates.cpu().numpy());neighbor_ids.append(ids.cpu().numpy());neighbor_distances.append(values.cpu().numpy())
            candidates=np.concatenate(candidate_values);ids=np.concatenate(neighbor_ids);distances=np.concatenate(neighbor_distances)
            error=np.sum((candidates-data['target'][:,None])**2,-1).mean((2,3))
            selected=error.argmin(1); best_candidate=candidates[np.arange(len(candidates)),selected]
            pair_rms=[np.sqrt(np.sum((candidates[:,i]-candidates[:,j])**2,-1).mean((1,2))).mean()
                      for i in range(8) for j in range(i)]
            result['retrieval_top1']=distribution(candidates[:,0],data)
            result['retrieval_best_of_8_coverage']=distribution(best_candidate,data)
            steps=np.diff(np.concatenate((np.broadcast_to(data['current'][:,None,None],
                (len(data['current']),8,1,11,3)),candidates),axis=2),axis=2)
            step_distance=np.linalg.norm(steps,axis=-1)
            result['retrieval_diagnostics']=dict(mean_pair_rms_m=float(np.mean(pair_rms)),
                median_nearest_normalized_rms=float(np.median(np.sqrt(distances[:,0]))),
                point_step_p99_m=float(np.quantile(step_distance,.99)),
                point_step_max_m=float(step_distance.max()),
                distinct_source_episodes=bool(all(len(np.unique(arrays['train']['episode'][r]))==8 for r in ids)))
            results[split]=result
            if split=='test':
                np.savez_compressed(output/'test-candidates.npz',absolute=predictions[split]['absolute'],
                    displacement=predictions[split]['displacement'],retrieval=candidates,source_rows=ids,
                    normalized_distances=distances,coverage_oracle_index=selected,current=data['current'],
                    target=data['target'],episode=data['episode'],tick=data['tick'])
        check()
        for path,digest in hashes.items():
            if sha(path)!=digest:
                raise ValueError('source drift: '+path)
        screen={arm: bool(results['test'][arm]['point_rmse_m']<=.9*results['test']['persistence']['point_rmse_m']
            and results['test'][arm]['h24_rmse_m']<results['test']['persistence']['h24_rmse_m'])
            for arm in ('absolute','displacement','retrieval_top1')}
        result=dict(status='PROMISING' if any(screen.values()) else 'UNCLEAR',test_screen=screen,
            metrics=results,selected_steps=chosen,history=history,elapsed_s=time.monotonic()-started,
            scope='Offline proposal only; best-of-8 is GT coverage, not deployable selection or execution',
            checkpoint_sha256={arm:sha(output/(arm+'-best.pt')) for arm in models})
        write(output/'result.json',result)
        manifest.update(status='COMPLETED',elapsed_s=result['elapsed_s'],selected_steps=chosen,checkpoint_sha256=result['checkpoint_sha256'])
        print(json.dumps(dict(status=result['status'],test_screen=screen,test=results['test'])),flush=True)
    except Exception as error:
        manifest.update(status='FAILED',error=repr(error),elapsed_s=time.monotonic()-started)
        raise
    finally:
        write(output/'manifest.json',manifest)


if __name__ == '__main__':
    main()
