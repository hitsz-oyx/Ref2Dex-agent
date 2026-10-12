"""Independent GT-only chunk-R physical gate; no WAM or world policy involved."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

TASK=Path(__file__).resolve().parents[2]
ROOT=TASK.parents[2]
OLD=ROOT/'src/task/consequence-evaluator'
sys.path[:0]=[str(TASK/'src'),str(TASK/'tools/audit'),str(TASK/'tools/run')]
from audit_wam_ref2_shadow import sha,write
from collect_competent_pd import OFFICIAL,OFFICIAL_SHA,RUNTIME,gpu_state


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fit',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();fit=a.fit.resolve();out=a.output.resolve()
    if (out.exists() or ROOT/'outputs/trajectory-policy' not in out.parents
            or Path(sys.prefix).resolve()!=RUNTIME.resolve()):
        raise ValueError('fresh output and pinned native runtime required')
    if gpu_state(2)['pids']:raise RuntimeError('GPU2 occupied')
    scratch=ROOT/'tmp/wam-v2';scratch.mkdir(exist_ok=True,parents=True)
    os.environ.update(CUDA_VISIBLE_DEVICES='2',TMPDIR=str(scratch),
        TORCH_EXTENSIONS_DIR=str(ROOT/'tmp/consequence-official-generator/torch201-extensions'),
        OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    native_root=ROOT/'third_party/DExplore/dexplore'
    sys.path[:0]=['/home2/wyy/isaac-gym/isaacgym/python',str(OLD/'src'),str(OLD/'tools/audit'),
        str(ROOT),str(native_root),str(ROOT/'src/task/CmResidual/tools')]
    sys.path.append('/home2/wyy/.local/lib/python3.8/site-packages')
    from isaacgym import gymtorch
    import dexplore_ddp_rank_bootstrap
    import torch
    import evaluate as native
    from learning import common_player
    from rl_games.common import player as base_player
    from probe_official_generator import install_legacy_player_compat,model_state
    from consequence_evaluator.native_reset import install_reset_patch
    from consequence_evaluator.physical_geometry import PhysicalGeometry,poses
    from consequence_evaluator.value_geometry import TableSupport
    from env.tasks.base_dexplore_task import DexploreTask
    from trajectory_policy.pd_retargeter import load_model,decode_target,command,pd_from_action
    from trajectory_policy.native_R_inference import calibrated_PD_forward
    from trajectory_policy.chunk_retargeter import load_chunk
    from trajectory_policy.retargeter_data import features,ACTIVE
    from execute_wam_native_bridge import partial_grasp_outcome
    if torch.__version__!='2.0.1+cu118' or sha(OFFICIAL)!=OFFICIAL_SHA:
        raise ValueError('pinned runtime and official teacher required')
    torch.set_num_threads(2);torch.manual_seed(292);torch.set_float32_matmul_precision('highest')
    owner=ROOT/'outputs/trajectory-policy';source=owner/'competent-pd-teacher-20261011-r2'
    sm=json.loads((source/'manifest.json').read_text());fm=json.loads((fit/'manifest.json').read_text())
    fr=json.loads((fit/'result.json').read_text())
    if (sm['status']!='COMPLETED' or not sm['training_allowed'] or fm['status']!='COMPLETED'
            or sha(fit/'chunk.pt')!=fr['checkpoint_sha256']
            or fm['source']!=str(source) or not json.loads((source/'audit.json').read_text())['pass']):
        raise ValueError('completed chunk fit and qualified audited source required')
    with np.load(source/'trajectory.npz',allow_pickle=False) as f:oracle=f['hand'][:,24].copy()
    checkpoint=fit/'chunk.pt';cp=torch.load(checkpoint,map_location='cuda:0')
    chunk=load_chunk(cp,'cuda:0')
    r_path=owner/'competent-pd-fit-20261011-r1/tau.pt'
    old_r=load_model(torch.load(r_path,map_location='cuda:0'),'cuda:0')
    run_dir=ROOT/'outputs/consequence-evaluator/baseline-transfer-s3-20261007-r2/airplane_base'
    cfg=json.loads((run_dir/'config.json').read_text());train_cfg=native_root/'data/cfg/train/rlg/inspire.yaml'
    out.mkdir();roles=['teacher','GT_tau_old_PD_R','GT_tau_chunk_R']
    m=dict(schema='ref2dex.chunk-R-GT-execution.v1',status='RUNNING',task='goal-point-wam',
        experiment_id=fm['experiment_id'],run_id=out.name,seed=292,physical_gpu=2,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        controls=128,roles=roles,dt=1/30,oracle_source_env=24,training_allowed=False,
        checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
        source=str(source),source_trajectory_sha256=sha(source/'trajectory.npz'),
        controller='chunk predicts independent native action8; replan every tick; execute only first',
        scope='privileged GT hand trajectory oracle; does not establish autonomous WAM utility')
    write(out/'manifest.json',m);install_legacy_player_compat(native,common_player,base_player);install_reset_patch()
    started=time.monotonic();logs={};executed=0

    class Player(native.EvalPlayer):
        def restore(self,filename):
            packet=native.torch_ext.load_checkpoint(filename)
            self.model.load_state_dict(model_state(packet['model']),strict=True)
            if self.normalize_input:self.running_mean_std.load_state_dict(packet['running_mean_std'],strict=True)
            if self._normalize_amp_input:self._amp_input_mean_std.load_state_dict(packet['amp_input_mean_std'],strict=True)

        def run(self):
            nonlocal executed
            task=self.env.task
            if task.num_envs!=3 or self.is_rnn:raise ValueError('three nonrecurrent GPU rows required')
            task._state_init=DexploreTask.StateInit.Start;task._hybrid_init_prob=1.
            task._adaptive_kappa_enabled=False;task._enable_early_termination=False
            obs=self.env_reset(torch.arange(3,device=task.device));self.get_batch_size(obs['obs'],1)
            if (task.start_times!=0).any():raise ValueError('frame0 reset required')
            geometry=PhysicalGeometry(task,native_root/'data/assets',distance_device=task.device)
            support=TableSupport(native_root/'data/assets',task.device)
            offset,scale=task._pd_action_offset.clone(),task._pd_action_scale.clone()
            controller=dict(offset=offset.tolist(),scale=scale.tolist())
            if controller!=cp['native_controller']:raise ValueError('native control units differ from fit')
            m['native_controller']=controller
            captured=actual_pd=None;original_pre,original_pd=task.pre_physics_step,task._action_to_pd_targets
            def pre(action):
                nonlocal captured
                captured=action.clone();return original_pre(action)
            def pd(action):
                nonlocal actual_pd
                actual_pd=original_pd(action).clone();return actual_pd
            task.pre_physics_step,task._action_to_pd_targets=pre,pd
            logs.update({k:[] for k in ('q','dq','hand','obj','velocity','surface_gap','support_gap','table_footprint',
                'action','applied','pd_targets','intended','clipped','R_state','R_tau','chunk_action')})
            def record():
                hand,gap=geometry.measure(task);sup,foot=support.measure(task,geometry)
                d=dict(q=task._dof_pos.clone(),dq=task._dof_vel.clone(),hand=hand,obj=poses(task._target_states),
                    velocity=task._target_states[:,7:13].clone(),surface_gap=gap,support_gap=sup,table_footprint=foot)
                for k,v in d.items():logs[k].append(v.cpu().numpy().copy())
                return d
            current=record()
            m['reset_max_error']=max(float(np.abs(logs[k][0]-logs[k][0][:1]).max()) for k in ('q','dq','hand','obj','velocity'))
            if m['reset_max_error']>1e-5:raise ValueError('measured initial states differ')
            for tick in range(128):
                with torch.no_grad():
                    padded=obs['obs'].repeat(22,1)[:64]
                    action=self.get_action(dict(obs,obs=padded),True)[:3].clamp(-1,1).clone()
                    intended=action.clone()
                    future=np.broadcast_to(oracle[tick+np.arange(1,25)],(2,24,11,3))
                    state,tau=features(*(current[k][1:].cpu().numpy() for k in ('q','dq','hand','obj','velocity')),future)
                    s,t=[torch.as_tensor(v,device=task.device) for v in (state,tau)]
                    target=decode_target(calibrated_PD_forward(old_r,s[:1],t[:1]),current['q'][1:2])
                    raw,applied,_=command(target,current['q'][1:2],offset,scale)
                    intended[1],action[1]=raw[0],applied[0]
                    predicted=chunk(s[1:],t[1:])
                    intended[2].zero_();intended[2,list(ACTIVE)]=predicted[0,0]
                    action[2]=intended[2].clamp(-1,1)
                    expected=pd_from_action(action,current['q'],offset,scale)
                    obs,_,done,_=self.env_step(self.env,action.clone());executed+=1
                    if not isinstance(obs,dict):obs={'obs':obs}
                    if (captured is None or not torch.equal(captured,action)
                            or (actual_pd-expected).abs().max()>1e-6):raise ValueError('native applied action/PD differs')
                    if tick<127 and done.any():raise ValueError('unexpected reset')
                    vals=dict(action=action,applied=captured,pd_targets=actual_pd,intended=intended,
                        clipped=(intended-action).abs().amax(-1)>1e-6,R_state=s,R_tau=t,chunk_action=predicted[0])
                    for k,v in vals.items():logs[k].append(v.cpu().numpy().copy())
                    current=record()
                if tick==0 or (tick+1)%32==0:
                    gpu=gpu_state(2)
                    if set(gpu['pids'])-{os.getpid()}:raise RuntimeError('foreign GPU2 process')
                    row=dict(control=tick+1,elapsed_s=time.monotonic()-started,gpu=gpu)
                    print(json.dumps(row),flush=True)
                    with (out/'monitor.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            packed={k:np.stack(v) for k,v in logs.items()}
            np.savez_compressed(out/'trajectory.npz',**packed)
            outcomes=[]
            for env,role in enumerate(roles):
                value=partial_grasp_outcome(dict(object_pose=packed['obj'][:,env],object_velocity=packed['velocity'][:,env],
                    surface_gap=packed['surface_gap'][:,env],support_gap=packed['support_gap'][:,env],table_footprint=packed['table_footprint'][:,env]))
                value.update(role=role,clipping_rate=float(packed['clipped'][:,env].mean()));outcomes.append(value)
            calibration=all(v['maximum_held_frames']>=45 and v['clipping_rate']<.01 for v in outcomes[:2])
            gate=calibration and outcomes[2]['maximum_held_frames']>=max(45,.8*outcomes[1]['maximum_held_frames']) and outcomes[2]['clipping_rate']<.01
            result=dict(status='COMPLETED',outcomes=outcomes,calibration_pass=calibration,GT_execution_gate=bool(gate),
                conclusion='PROMISING' if gate else 'UNPROMISING' if calibration else 'UNCLEAR',
                scope='partial128 GT-oracle grasp only; no placement or predicted WAM claim')
            write(out/'result.json',result);m.update(status='COMPLETED',trajectory_sha256=sha(out/'trajectory.npz'),
                result_sha256=sha(out/'result.json'),elapsed_s=time.monotonic()-started);write(out/'manifest.json',m)
            print(json.dumps(result),flush=True)
    native.EvalPlayer=Player
    sys.argv=[sys.argv[0],'--task','Dexplore_Inspire','--cfg_env',cfg['cfg_env'],'--cfg_train',str(train_cfg),
        '--motion_file',cfg['motion_root'],'--checkpoint',str(OFFICIAL),'--headless','--num_envs','3','--seed','292',
        '--sim_device','cuda:0','--rl_device','cuda:0','--pipeline','gpu','--graphics_device_id','0','--num_threads','1',
        '--disable-early-termination','--output',str(out/'unused.json'),'--output_path',str(out/'native')]
    cwd=Path.cwd()
    def deadline(signum,frame):raise TimeoutError('chunk GT gate600s deadline')
    signal.signal(signal.SIGALRM,deadline);signal.alarm(600)
    try:
        os.chdir(ROOT/'third_party/DExplore');native.main()
    except BaseException as error:
        m.update(status='FAILED',error=repr(error),executed_controls=executed)
        if logs:np.savez_compressed(out/'partial.npz',**{k:np.stack(v) for k,v in logs.items() if v})
        write(out/'manifest.json',m);raise
    finally:signal.alarm(0);os.chdir(cwd)


if __name__=='__main__':main()
