"""Separate stored-state fit, native reset drift, and batch-dependent shared inference."""
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-code-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=6)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[5]
    out, code = a.output.resolve(), a.model_code_root.resolve()
    if out.exists() or root/'outputs/ptv3-shared-wam-v2' not in out.parents:
        raise ValueError('Fresh Task output required')
    if shutil.disk_usage(root).free < 20*1024**3:
        raise RuntimeError('20GiB free space reserve')
    if subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid',
                                '--format=csv,noheader'],text=True).strip():
        raise RuntimeError('Assigned GPU occupied')
    scratch = root/'tmp'/out.name
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu), TMPDIR=str(scratch), OMP_NUM_THREADS='2',
        REF2DEX_RUN_ROOT=str(root), TORCH_EXTENSIONS_DIR=str(root/'tmp/shared-wam-v2-extensions'))
    task = code/'src/task/ptv3-shared-wam-v2'
    sys.path.insert(0,str(task/'src'))
    import numpy as np
    import torch
    from ptv3_shared_wam_v2.robot_data import V2RLRobotWindows, robot_collate
    from ptv3_shared_wam_v2.robot_model import SharedWAMV2Robot
    from ptv3_shared_wam_v2.robot_metrics import physical_errors, summarize_errors
    from ptv3_shared_wam.loading import to_device
    from ptv3_shared_wam.checkpoint import atomic_json
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('900s offline diagnostic')))
    signal.alarm(900)
    out.mkdir()
    started = time.monotonic()
    old = root/'outputs/ptv3-shared-wam-v2/shared-actor-shared-20261015-r1'
    native = root/'outputs/ptv3-shared-wam-v2/shared-native-grasp-20261015-r1'
    oracle = root/'outputs/ptv3-shared-wam-v2/inspire-component-oracle-20261015-r1'
    checkpoint = native/'inputs/control-step7000.pt'
    status = dict(status='RUNNING',pid=os.getpid(),gpu=a.gpu,checkpoint=str(checkpoint),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        model_code_root=str(code),training=False,scope='Stored TRAIN/DEV fit and inference contract diagnostics')
    atomic_json(out/'process.json',status)
    try:
        packet = torch.load(checkpoint,map_location='cpu',weights_only=False)
        config, representation = packet['config'], packet['config']['action_representation']
        assert packet['step'] == 7000 and config['action_interface'] == 'shared'
        source = Path(packet['provenance']['source'])
        snapshot = json.loads((old/'encoder_workspace_admission.json').read_text())
        datasets = {}
        for split in ('train','val'):
            dataset = V2RLRobotWindows(source,split,config['object_points'],representation)
            # Restore exact, previously audited training membership; no new filtering policy.
            seq = {r['sequence']:i for i,r in enumerate(dataset.records)}
            keys = [(int(t),seq[r['sequence']]) for r in snapshot[split]['records'] for t in r['admitted_ticks']]
            assert len(keys) == snapshot[split]['retained_windows'] and set(keys).issubset(set(dataset.keys))
            dataset.keys = keys
            dataset.phase_pools = [[np.asarray([i for i,(tick,s) in enumerate(keys)
                if s==j and (tick<16)==startup],dtype=np.int64) for j in range(len(dataset.records))]
                for startup in (True,False)]
            dataset.phase_pools = [[p for p in pools if len(p)] for pools in dataset.phase_pools]
            datasets[split] = dataset
        assert [len(datasets[s]) for s in ('train','val')] == [93680,10376]
        model = SharedWAMV2Robot(packet['stats'],config,representation)
        model.load_state_dict(packet['model'],strict=True)
        del packet
        model.cuda().eval()
        panels = dict(TRAIN_balanced256=('train',datasets['train'].sample_indices(np.random.default_rng(192),256)),
                      DEV_fixed256=('val',np.load(old/'validation.npy')))
        groups = [('TRAIN_airplane','train','s1_airplane_pass_1',oracle/'TRAIN-airplane/native')]
        groups += [('VAL_'+short,'val',name,native/('control-'+short)/'native') for short,name in
                   [('airplane','s10_airplane_pass_1'),('sphere','s4_spheresmall_pass_1'),('mug','s10_mug_pass_1')]]
        for name,split,sequence,_ in groups:
            panels[name] = (split,np.asarray([i for i,(t,s) in enumerate(datasets[split].keys)
                                             if datasets[split].records[s]['sequence']==sequence]))
        memberships = {name:[dict(sequence=datasets[split].records[datasets[split].keys[int(i)][1]]['sequence'],
                                 tick=datasets[split].keys[int(i)][0]) for i in indices]
                       for name,(split,indices) in panels.items()}
        atomic_json(out/'panels.json',memberships)
        result = dict(panels={},initial_parity={},batch_contract={},scope=status['scope'])
        with torch.no_grad():
            for name,(split,indices) in panels.items():
                data = datasets[split]
                errors, chunks = {}, []
                torch.manual_seed(148)
                for start in range(0,len(indices),16):
                    batch = to_device(robot_collate([data[int(i)] for i in indices[start:start+16]]),'cuda')
                    prediction = model(batch)
                    for key,value in physical_errors(prediction,batch,data.controller,representation).items():
                        errors.setdefault(key,[]).append(value)
                    chunks.append(prediction['action'].cpu().numpy())
                ticks = np.asarray([data.keys[int(i)][0] for i in indices])
                metrics = summarize_errors(errors)
                for phase,mask in [('startup',ticks<16),('later',ticks>=16)]:
                    values = np.concatenate(errors['PD_xyz_mm'])[mask]
                    metrics[phase] = dict(count=len(values),mean_mm=float(values.mean()) if len(values) else None,
                                          p95_mm=float(np.percentile(values,95)) if len(values) else None)
                result['panels'][name] = metrics
                np.save(out/(name+'-chunks.npy'),np.concatenate(chunks))
                atomic_json(out/'result.partial.json',result)
                print(name,json.dumps(metrics['PD_xyz_mm']),flush=True)
            # Fixed initial stored TRAIN sample, identical seed, varied batch context.
            train_index = int(panels['TRAIN_airplane'][1][0])
            sample = datasets['train'][train_index]
            assert datasets['train'].keys[train_index][0] == 0
            examples = [('single_seed148',[sample],148),('single_repeat148',[sample],148),
                        ('single_seed149',[sample],149),('duplicate2',[sample]*2,148),
                        ('duplicate16',[sample]*16,148),
                        ('mixed16',[sample]+[datasets['val'][int(i)] for i in panels['DEV_fixed256'][1][:15]],148)]
            actions = {}
            for name,samples,seed in examples:
                torch.manual_seed(seed)
                batch = to_device(robot_collate(samples),'cuda')
                actions[name] = model(batch)['action'][0].cpu().numpy()
            baseline = actions['single_seed148']
            for name,value in actions.items():
                result['batch_contract'][name] = dict(first_XYZ_difference_mm=float(np.linalg.norm(value[0,:3]-baseline[0,:3])*1000),
                    full_chunk_max_difference=float(np.abs(value-baseline).max()))
            np.savez_compressed(out/'batch-contract-chunks.npz',**actions)
            for name,split,sequence,path in groups:
                data = datasets[split];index = int(panels[name][1][0]);tick,seq = data.keys[index]
                in_membership = tick == 0
                first_admitted_tick = tick
                if not in_membership:
                    # Reset contract only: tick0 is excluded from strict fitting metrics.
                    data = copy.copy(data)
                    data.keys = [(0,seq)]
                    index = 0
                stored,_ = data.sequence(seq)
                sample = data[index]
                torch.manual_seed(148)
                prediction = model(to_device(robot_collate([sample,sample]),'cuda'))['action'][0].cpu().numpy()
                with np.load(path/'trajectory.npz') as z:
                    compare = {key:float(np.abs(stored[key][0]-z[key][0,2]).max())
                               for key in ('q','dq','hand','obj','velocity')}
                    compare.update(tick0_in_strict_membership=in_membership,first_admitted_tick=first_admitted_tick)
                    compare['state87'] = float(np.abs(sample['native_state']-z['R_state'][0]).max())
                    compare['H4_hand'] = float(np.abs(stored['hand'][[0]*4]-z['wam_hand_history'][0]).max())
                    compare['H4_object'] = float(np.abs(stored['obj'][[0]*4]-z['wam_object_history'][0]).max())
                    compare['object_goal24'] = float(np.abs(stored['reference_obj'][24]-z['wam_goal'][0]).max())
                    compare['predicted_first_XYZ_difference_mm'] = float(np.linalg.norm(prediction[0,:3]-z['R_output_chunk'][0,1,0,:3])*1000)
                result['initial_parity'][name] = compare
        result.update(status='COMPLETED',elapsed_s=time.monotonic()-started)
        atomic_json(out/'result.json',result)
        status.update(status='COMPLETED',elapsed_s=result['elapsed_s'])
    except BaseException as error:
        status.update(status='FAILED',error=repr(error));raise
    finally:
        atomic_json(out/'process.json',status);signal.alarm(0)


if __name__=='__main__':
    main()
