"""Replay selected official teacher motions and capture same-execution controls/state.

Use the pinned legacy simulator runtime, not the PTv3 training interpreter.
The 660 export determines membership; geometric tensors are decision-known plans.
All supervised robot coordinates and commands are freshly measured together.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
NATIVE = ROOT / 'third_party/DExplore/dexplore'
OLD = ROOT / 'src/task/consequence-evaluator'
TRAJECTORY = ROOT / 'src/task/trajectory-policy'
RUNTIME = Path('/home2/wyy/oyx_ws/.runtime_envs/dexplore_v120_train')
OFFICIAL = Path('/home2/wyy/oyx_ws/_external/dexplore_official_v120/checkpoint/inspire.pth')
FULL_ASSETS = Path('/home2/wyy/oyx_ws/dexplore/dexplore/data/assets')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--seed', type=int, default=120)
    parser.add_argument('--seconds', type=int, default=900)
    parser.add_argument('--capture-current-observation', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    names = json.loads(args.selection.read_text())
    allowed = [ROOT/'outputs/ptv3-shared-wam']
    if args.capture_current_observation:
        allowed.append(ROOT/'outputs/ptv3-shared-wam-v2')
    if (out.exists() or not any(p in out.parents for p in allowed)
            or not 1 <= len(names) <= 32 or len(set(names)) != len(names)
            or Path(sys.prefix).resolve() != RUNTIME.resolve()):
        raise ValueError('fresh local output, 1..32 unique motions and pinned runtime required')
    occupied = subprocess.check_output(['nvidia-smi','-i',str(args.gpu),
        '--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if occupied:
        raise RuntimeError('assigned GPU occupied: '+occupied)
    source = ROOT/'data/processed_data/inspire_rl_object_dexplore'
    source_meta = json.loads((source/'manifest.json').read_text())
    geometric = Path(source_meta['source_geometric_root'])
    for name in names:
        if Path(name).name != name or not (source/name/'interaction_hand_inspire.pt').is_file():
            raise ValueError('selection outside frozen 660-member corpus')
        if not (geometric/name/'interaction_hand_inspire.pt').is_file():
            raise ValueError('missing corresponding geometric plan')
    out.mkdir(parents=True)
    motions = out/'motions'
    motions.mkdir()
    for name in names:
        (motions/name).symlink_to(geometric/name)
    # Native object loading contains paths relative to the process cwd. Give
    # this process an isolated read-only full-asset view; the shared repository
    # asset link is a historical partial expert bank and must remain intact.
    runtime = out/'gym_runtime'
    asset_view = runtime/'dexplore/data/assets'
    asset_view.parent.mkdir(parents=True)
    asset_view.symlink_to(FULL_ASSETS)
    scratch = ROOT/'tmp'/out.name
    scratch.mkdir(parents=True,exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu),TMPDIR=str(scratch),
        OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',
        TORCH_EXTENSIONS_DIR=str(ROOT/'tmp/consequence-official-generator/torch201-extensions'))
    sys.dont_write_bytecode = True
    sys.path[:0] = ['/home2/wyy/isaac-gym/isaacgym/python',str(NATIVE),str(ROOT),
        str(OLD/'src'),str(OLD/'tools/audit'),str(TRAJECTORY/'src'),
        str(ROOT/'src/task/CmResidual/tools')]
    sys.path.append('/home2/wyy/.local/lib/python3.8/site-packages')
    from isaacgym import gymtorch  # must precede torch
    import dexplore_ddp_rank_bootstrap
    import numpy as np
    import torch
    import run as native
    from learning import common_player, dexplore_players
    from rl_games.common import player as base_player
    from probe_official_generator import install_legacy_player_compat, model_state, write
    from consequence_evaluator.native_reset import install_reset_patch
    from consequence_evaluator.physical_geometry import poses
    from trajectory_policy.wam_shadow import NATIVE_LINKS
    from trajectory_policy.pd_retargeter import pd_from_action
    from src.task.CmResidual.dexplore_cm_geometry import _surface_geometry_class
    if torch.__version__ != '2.0.1+cu118':
        raise ValueError('pinned Torch 2.0.1 required')
    torch.set_num_threads(2)
    install_legacy_player_compat(native,common_player,base_player)
    # run.py's native player constructor takes the legacy flat config, whereas
    # the older evaluate.py adapter used by the helper takes a params keyword.
    native.Runner.create_player = lambda runner: runner.player_factory.create(
        runner.algo_name, config=runner.config)
    install_reset_patch()
    started = time.monotonic()
    manifest = dict(schema='ref2dex.rl660-batch.v1',status='RUNNING',sequences=names,
        seed=args.seed,physical_gpu=args.gpu,fps=30,source_membership=str(source.resolve()),
        geometric_plan_root=str(geometric),official_generator=str(OFFICIAL),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        actor_role='official data generator only; no official weights in learned WAM/Chunk-R')
    manifest['capture_current_observation'] = args.capture_current_observation
    write(out/'manifest.json',manifest)

    class Player(dexplore_players.DexplorePlayerContinuous):
        def restore(self,filename):
            packet = torch.load(filename,map_location=self.device,weights_only=False)
            self.model.load_state_dict(model_state(packet['model']),strict=True)
            if self.normalize_input:
                self.running_mean_std.load_state_dict(packet['running_mean_std'],strict=True)
            if self._normalize_amp_input:
                self._amp_input_mean_std.load_state_dict(packet['amp_input_mean_std'],strict=True)

        @torch.no_grad()
        def _run_export_rl(self,task):
            n = task.num_envs
            if n != len(names) or sorted(Path(p).name for p in task.motion_file) != sorted(names):
                raise ValueError('one unique sequence per environment required')
            task._adaptive_kappa_enabled = False
            task._enable_early_termination = False
            lengths = task.max_episode_length.clone()
            maximum = int(lengths.max())
            task.max_episode_length[:] = maximum
            task.rollout_length = maximum
            self.has_batch_dimension = True
            obs = self.env_reset()
            if (task.start_times != 0).any() or not torch.equal(task.data_id,torch.arange(n,device=task.device)):
                raise ValueError('frame-zero and exact motion identity required')
            body_names = task.gym.get_actor_rigid_body_names(task.envs[0],task.humanoid_handles[0])
            ids = [body_names.index(name) for name in NATIVE_LINKS]
            offset,scale = task._pd_action_offset.clone(),task._pd_action_scale.clone()
            manifest['native_controller'] = dict(offset=offset.tolist(),scale=scale.tolist())
            assets = asset_view
            geometry = {}
            for name in set(task.object_name):
                surface = _surface_geometry_class()(hand_urdf=assets/'inspire_hand_new/inspire_hand_right.urdf',
                    object_urdf=assets/('mjcf/'+name+'.urdf'),query_links=NATIVE_LINKS,
                    object_count=512,hand_count=128,seed=42,device=task.device)
                points,normals = surface.object_local,surface.object_normal
                geometry[name] = dict(points=points.astype(np.float32),normals=normals.astype(np.float32),
                    center=points.mean(0).astype(np.float32),
                    radius=np.asarray(np.sqrt(np.mean(np.sum((points-points.mean(0))**2,-1))),np.float32))
            logs = {k:[] for k in ('q','dq','hand','obj','velocity','action','applied','pd_targets')}
            state_fields = set(('q','dq','hand','obj','velocity'))
            if args.capture_current_observation:
                if len(task._key_body_ids) != 16 or len(task._contact_body_ids) != 5 or task.obs_buf.shape[1] != 1442:
                    raise ValueError('pinned Inspire 16-key-body/5-contact/1442-observation contract required')
                state_fields.update(('native_current_obs','body_state','body_contact_forces',
                                     'object_contact_force','reference_current','frame_tick'))
                logs.update({k:[] for k in state_fields if k not in logs})
                manifest['observation_contract'] = dict(schema='ref2dex.dexplore-current-observation.v1',
                    dimension=721, reference_delta=0, teacher_reference_deltas=[1,16],
                    timing='state[t]/current_ref[t] before action[t]; q_PD[t] captured in pre_physics_step',
                    contact_slice=[252,257],reference_contact_error_slice=[401,406],
                    body_names=body_names,key_body_ids=task._key_body_ids.tolist(),
                    contact_body_ids=task._contact_body_ids.tolist(),
                    reference_dim=int(task.hoi_data.shape[-1]),
                    body_state_order='xyz, quaternion_xyzw, linear_velocity, angular_velocity',
                    future_reference_inputs_saved=False)
            capture = actual_pd = None
            pre,pd = task.pre_physics_step,task._action_to_pd_targets
            def capture_pre(action):
                nonlocal capture
                capture = action.clone()
                return pre(action)
            def capture_pd(action):
                nonlocal actual_pd
                actual_pd = pd(action).clone()
                return actual_pd
            task.pre_physics_step,task._action_to_pd_targets = capture_pre,capture_pd
            def record(tick):
                bodies = task._rigid_body_state.view(n,-1,13)
                values = dict(q=task._dof_pos,dq=task._dof_vel,hand=bodies[:,ids,:3],
                    obj=poses(task._target_states),velocity=task._target_states[:,7:13])
                if args.capture_current_observation:
                    if not bool((task.progress_buf == tick).all()):
                        raise ValueError('unexpected reset/tick drift during same-execution capture')
                    env_ids = torch.arange(n,device=task.device)
                    original_actor_obs = task.obs_buf.clone()
                    native_current = task._compute_observations_iter(env_ids,0)
                    if native_current.shape != (n,721) or not torch.equal(task.obs_buf,original_actor_obs):
                        raise ValueError('current observation changed teacher input or dimension')
                    reference = task.hoi_data[task.data_id,task.progress_buf].clone()
                    forces = task._contact_forces[:,task._contact_body_ids,:]
                    bits = (forces.abs()>.1).any(-1).float()
                    ref_bits = reference[:,119+16*3+1:119+16*3+1+16][:,[3,6,9,12,15]]
                    contact_error = ref_bits*((ref_bits+1)/2-bits)
                    if (not torch.equal(native_current[:,252:257],bits)
                            or not torch.equal(native_current[:,401:406],contact_error)):
                        raise ValueError('native current contact slices differ from raw same-tick forces/reference')
                    if tick == 0:
                        # A live negative control: future reference edits cannot affect delta0.
                        future = task.hoi_data[:,1:].clone()
                        try:
                            task.hoi_data[:,1:] = 0
                            changed_future = task._compute_observations_iter(env_ids,0)
                            if not torch.equal(native_current,changed_future):
                                raise ValueError('future reference leaked into current-only observation')
                        finally:
                            task.hoi_data[:,1:] = future
                        if not torch.equal(task.obs_buf,original_actor_obs):
                            raise ValueError('audit changed original teacher observation')
                        manifest['observation_contract']['live_future_invariance_passed'] = True
                        write(out/'manifest.json',manifest)
                    body_state = torch.cat((task._rigid_body_pos,task._rigid_body_rot,
                                            task._rigid_body_vel,task._rigid_body_ang_vel),-1)
                    if body_state.shape[1] != len(body_names):
                        raise ValueError('body-state identity mismatch')
                    values.update(native_current_obs=native_current,body_state=body_state,
                        body_contact_forces=task._contact_forces,
                        object_contact_force=task._tar_contact_forces,
                        reference_current=reference,frame_tick=task.progress_buf)
                for key,value in values.items():
                    logs[key].append(value.cpu().numpy().copy())
                return values
            current = record(0)
            for tick in range(maximum-1):
                action = self.get_action(obs,True).clamp(-1,1).clone()
                expected = pd_from_action(action,current['q'],offset,scale)
                raw,_,_,_ = self.env_step(self.env,action.clone())
                obs = self.obs_to_torch(raw)
                if not torch.equal(capture,action) or (expected-actual_pd).abs().max() > 1e-6:
                    raise ValueError('captured command or same-tick PD conversion mismatch')
                for key,value in dict(action=action,applied=capture,pd_targets=actual_pd).items():
                    logs[key].append(value.cpu().numpy().copy())
                current = record(tick+1)
                if tick == 0 or (tick+1)%128==0:
                    elapsed = time.monotonic()-started
                    gpu = subprocess.check_output(['nvidia-smi','-i',str(args.gpu),
                        '--query-gpu=utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip()
                    print(json.dumps(dict(control=tick+1,maximum=maximum,elapsed_s=elapsed,gpu=gpu)),flush=True)
            data = {k:np.stack(v) for k,v in logs.items()}
            if any(not np.isfinite(v).all() for v in data.values()):
                raise FloatingPointError('nonfinite actual execution')
            records = []
            for env,motion in enumerate(task.motion_file):
                name = Path(motion).name
                length = int(lengths[env])
                folder = out/'sequences'/name
                folder.mkdir(parents=True)
                values = {k:v[:length if k in state_fields else length-1,env]
                          for k,v in data.items()}
                reference = task.hoi_data[env,:length]
                states = torch.zeros(length,13,device=task.device)
                states[:,:3],states[:,3:7] = reference[:,106:109],reference[:,109:113]
                values['reference_obj'] = poses(states).cpu().numpy()
                np.savez_compressed(folder/'trajectory.npz',**values)
                objname = name.split('_')[1]
                np.savez_compressed(folder/'canonical.npz',**geometry[objname])
                error = np.linalg.norm(values['obj'][:,:3,3]-values['reference_obj'][:,:3,3],axis=-1)
                records.append(dict(sequence=name,object=objname,frames=length,
                    reference_position_rmse_m=float(np.sqrt(np.mean(error**2))),
                    saturation_rate=float((np.abs(values['applied'])>=.999999).any(-1).mean()),
                    path=str(folder.relative_to(out))))
            manifest.update(status='COMPLETED',training_allowed=True,records=records,
                elapsed_s=time.monotonic()-started)
            write(out/'manifest.json',manifest)
            print(json.dumps(dict(status='COMPLETED',sequences=len(records),elapsed_s=manifest['elapsed_s'])),flush=True)

    dexplore_players.DexplorePlayerContinuous = Player
    cfg = ROOT/'outputs/consequence-evaluator/baseline-transfer-inputs-20261007-r1/cfg/inspire.yaml'
    sys.argv = [sys.argv[0],'--task','Dexplore_Inspire','--cfg_env',str(cfg),'--cfg_train',
        str(NATIVE/'data/cfg/train/rlg/inspire.yaml'),'--test','--checkpoint',str(OFFICIAL),
        '--headless','--motion_file',str(motions),'--num_envs',str(len(names)),
        '--export_rl','--export_output_dir',str(out),'--seed',str(args.seed),
        '--sim_device','cuda:0','--rl_device','cuda:0','--pipeline','gpu','--graphics_device_id','0',
        '--num_threads','1','--output_path',str(out/'native')]
    def timeout(number,frame):
        raise TimeoutError('bounded collector deadline')
    signal.signal(signal.SIGALRM,timeout)
    signal.alarm(args.seconds)
    try:
        os.chdir(runtime)
        native.main()
    except BaseException as error:
        manifest.update(status='FAILED',error=repr(error))
        write(out/'manifest.json',manifest)
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
