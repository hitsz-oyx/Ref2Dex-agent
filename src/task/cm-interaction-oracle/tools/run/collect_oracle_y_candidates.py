#!/usr/bin/env python3
"""Fresh simulator + frozen full prefix replay; no warm-state restoration."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT), str(ROOT/'third_party/DExplore/dexplore')]
from isaacgym import gymtorch  # must precede torch
import torch
import evaluate as original
from src.task.CmResidual.paired_evaluation import (capture_initial, restore_initial, capture_rng,
    restore_rng, cpu_copy, fingerprint, physical_property_value)
from src.task.CmResidual.physical_value_contract import HoldTracker
from src.task.CmResidual.physical_value_live import snapshot, contacts
from src.task.CmResidual.dexplore_cm_geometry import DExploreCmv2GeometryBridge, dexplore_root_pose
sys.path.insert(0, str(ROOT/'src/task/cm-interaction-oracle/src'))
from intervention import HISTORY, all_arms_have_headroom, update_predecision_hold
from oracle_y_utility import POST_WINDOW, CANDIDATES, candidate_deltas, align_native_reference_tables
from collect_interventions import SOURCE_SHA
ARGS = None


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CandidatePlayer(original.EvalPlayer):
    @torch.no_grad()
    def run(self):
        begin = time.monotonic(); torch.set_num_threads(2)
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
        task = self.env.task
        device = task.device
        task._enable_early_termination = False; task._adaptive_kappa_enabled = False
        task._hybrid_init_prob = .5
        self.model.eval()
        n = task.num_envs; ids = torch.arange(n, device=device); empty = ids[:0]
        if abs(task.dt-1/30)>1e-8 or len(task.motion_file)!=3:
            raise ValueError('motion/control interval drift')
        if not torch.allclose(task._pd_action_scale[:3], torch.ones(3,device=device)):
            raise ValueError('wrist scale drift')
        migrated_tables=align_native_reference_tables(task)
        obs = self.env_reset(ids); self.get_batch_size(obs['obs'],1)
        if self.is_rnn: self.init_rnn()
        properties = []
        for env in task.envs:
            actors = []
            for actor in range(task.gym.get_actor_count(env)):
                actors.append(dict(name=task.gym.get_actor_name(env,actor),
                    dof=cpu_copy(task.gym.get_actor_dof_properties(env,actor)),
                    body=physical_property_value(task.gym.get_actor_rigid_body_properties(env,actor)),
                    shape=physical_property_value(task.gym.get_actor_rigid_shape_properties(env,actor))))
            properties.append(actors)
        trace = None
        if ARGS.reference is None:
            initial = capture_initial(task,self,obs,properties)
            triggers = torch.full((n,),-1,dtype=torch.long,device=device); length = 2000
        else:
            initial = torch.load(ARGS.reference/'initial_state.pt',map_location='cpu',weights_only=False)
            trace = torch.load(ARGS.reference/'trace.pt',map_location='cpu',weights_only=False)
            triggers = trace['triggers'].to(device)
            length = int(triggers.max())+POST_WINDOW if (triggers>=0).any() else 1
        try:
            obs = restore_initial(task,self,initial,gymtorch.unwrap_tensor,properties,cpu_pose_atol=2.5e-7 if str(device)=='cpu' else 0.)
        except ValueError:
            differences={}
            for key in ('_root_states','_dof_state'):
                actual=getattr(task,key).cpu(); expected=initial['tensors'][key]
                delta=(actual-expected).abs()
                differences[key]=dict(max_abs=float(delta.max()),shape=list(actual.shape),
                    changed_indices=(delta>0).nonzero().tolist()[:20],
                    max_by_axis=delta.reshape(-1,delta.shape[-1]).amax(0).tolist())
            (ARGS.run_dir/'cold_restore_diagnostic.json').write_text(json.dumps(differences,indent=2)+'\n')
            raise
        initial_hash = fingerprint(initial)
        if trace is not None and trace['initial_fingerprint']!=initial_hash:
            raise ValueError('cold initial identity drift')
        if trace is None: torch.save(initial,ARGS.run_dir/'initial_state.pt')
        model_hash = fingerprint(self.model.state_dict())
        rms_hash = fingerprint(self.running_mean_std.state_dict()) if self.normalize_input else None
        asset = ROOT/'third_party/DExplore/dexplore/data/assets'
        bridge = DExploreCmv2GeometryBridge(hand_urdf=asset/'inspire_hand_new/inspire_hand_right.urdf',
            object_urdf=asset/'mjcf/airplane.urdf',device=device)
        tip_names = ('index_tip','middle_tip','pinky_tip','ring_tip','thumb_tip')
        tip_ids = task._key_body_ids[[task.cfg['env']['keyBodies'].index(k) for k in tip_names]]
        base_id = task._key_body_ids[task.cfg['env']['keyBodies'].index('hand_base_link')]
        def physical():
            pos = task._rigid_body_pos[:,task._contact_body_ids]; rot = task._rigid_body_rot[:,task._contact_body_ids]
            points,_ = bridge.geometry.object(dexplore_root_pose(task._target_states))
            distance = torch.cdist(pos,points[:,::8]).amin(-1)
            return torch.cat((task._target_states.clone(),torch.cat((pos,rot),-1).flatten(1),
                task._contact_forces[:,task._contact_body_ids].flatten(1),task._tar_contact_forces.clone(),
                distance,contacts(task).bool().all(-1,keepdim=True).float()),-1)
        tracker = HoldTracker(n,device); tracker.reset(ids,task._target_states[:,2])
        rest = task.hoi_refs[task.data_id,task.ref_index,0,108].clone()
        delta = candidate_deltas(device)
        previous = torch.zeros(n,18,device=device); history = torch.zeros(n,HISTORY,139,device=device)
        hold = torch.zeros(n,dtype=torch.long,device=device); terminal = torch.zeros(n,dtype=torch.bool,device=device)
        packet = dict(before=torch.zeros(n,72,device=device),history=torch.zeros_like(history),
            actor_obs=torch.zeros(n,obs['obs'].shape[-1],device=device),hand_root=torch.zeros(n,13,device=device),
            before_fingertip_positions=torch.zeros(n,5,3,device=device),before_hand_base_pose=torch.zeros(n,7,device=device),
            native_q=torch.zeros(n,32,18,device=device),trajectory=torch.zeros(n,32,72,device=device),
            fingertip_positions=torch.zeros(n,32,5,3,device=device),hand_base_pose=torch.zeros(n,32,7,device=device),
            actions=torch.zeros(n,8,18,device=device),base_actions=torch.zeros(n,8,18,device=device),
            pd_targets=torch.zeros(n,8,18,device=device),height=torch.zeros(n,POST_WINDOW,device=device),
            pair=torch.zeros(n,POST_WINDOW,dtype=torch.bool,device=device),
            valid_steps=torch.zeros(n,POST_WINDOW,dtype=torch.bool,device=device))
        # Raw per-environment pre-branch max errors. Units/tolerances are audited separately.
        errors = torch.zeros(n,7,device=device) # physical72, q/dq36, root13, obs, history, shadowaction, done
        logs = {k:[] for k in ('physical','dof','root','action','done')}; rng = []
        clipping = torch.zeros(n,dtype=torch.long,device=device)
        motion = task.data_id.clone(); start = task.start_times.clone()
        for tick in range(length):
            if time.monotonic()-begin>ARGS.wall_seconds: raise TimeoutError('candidate branch deadline')
            if trace is not None: restore_rng(trace['rng'][tick]['before_reset'])
            r_reset = capture_rng(); obs = self.env_reset(empty)
            phys = physical(); dof = torch.cat((task._dof_pos,task._dof_vel),-1).clone()
            root = task._humanoid_root_states.clone()
            hold = update_predecision_hold(hold,phys[:,2],rest,phys[:,71]>.5)
            compact = torch.cat((snapshot(task,tracker),previous,root,phys[:,13:66]),-1)
            if compact.shape[-1]!=139: raise ValueError('history contract drift')
            history = torch.cat((history[:,1:],compact[:,None]),1)
            if trace is not None: restore_rng(trace['rng'][tick]['before_action'])
            r_action = capture_rng(); base = self.get_action(obs,True).clamp(-1,1).to(device).clone()
            if trace is None:
                eligible = (~terminal)&(triggers<0)&(tick>=HISTORY-1)&(hold>=6)&(phys[:,71]>.5)
                eligible &= task.max_episode_length[task.data_id]-task.progress_buf>POST_WINDOW+1
                eligible &= task.rollout_length-(task.progress_buf-task.start_times)>POST_WINDOW+1
                eligible &= phys[:,66:71].amin(-1)<.06
                eligible &= all_arms_have_headroom(base,delta)
                triggers[eligible] = tick
                logs['physical'].append(phys.cpu()); logs['dof'].append(dof.cpu()); logs['root'].append(root.cpu())
            else:
                prefix = ((triggers<0)|(tick<=triggers))&~terminal
                for column,(value,key) in enumerate(((phys,'physical'),(dof,'dof'),(root,'root'))):
                    error = (value-trace[key][tick].to(device)).abs().amax(-1)
                    errors[prefix,column] = torch.maximum(errors[prefix,column],error[prefix])
                shadow = (base-trace['action'][tick].to(device)).abs().amax(-1)
                errors[prefix,5] = torch.maximum(errors[prefix,5],shadow[prefix])
            if ARGS.diagnose_prefix and trace is not None:
                inspected=(triggers>=0)&(tick<=triggers)&~terminal
                raw={key:float((value[inspected]-trace[key][tick].to(device)[inspected]).abs().max())
                     for value,key in ((phys,'physical'),(dof,'dof'),(root,'root'))}
                if tick<4 or tick%20==0: print(json.dumps(dict(prefix_diagnostic_tick=tick,errors=raw)),flush=True)
                if max(raw.values())>1e-4 or tick>=64:
                    diagnose=dict(engineering_only=True,tick=tick,raw=raw,initial_fingerprint=initial_hash,
                        q_position_max=float((dof[inspected,:18]-trace['dof'][tick].to(device)[inspected,:18]).abs().max()),
                        q_velocity_max=float((dof[inspected,18:]-trace['dof'][tick].to(device)[inspected,18:]).abs().max()),
                        physical_error_by_axis=(phys[inspected]-trace['physical'][tick].to(device)[inspected]).abs().amax(0).cpu().tolist())
                    (ARGS.run_dir/'prefix_diagnostic.json').write_text(json.dumps(diagnose,indent=2)+'\n')
                    return
            chosen = (triggers==tick).nonzero(as_tuple=False).flatten()
            if len(chosen):
                packet['before'][chosen] = phys[chosen]; packet['history'][chosen] = history[chosen]
                packet['actor_obs'][chosen] = obs['obs'][chosen].to(device); packet['hand_root'][chosen] = root[chosen]
                packet['before_fingertip_positions'][chosen] = task._rigid_body_pos[chosen][:,tip_ids]
                packet['before_hand_base_pose'][chosen] = torch.cat((task._rigid_body_pos[chosen,base_id],task._rigid_body_rot[chosen,base_id]),-1)
                if trace is not None:
                    for column,key in ((3,'actor_obs'),(4,'history')):
                        error=(packet[key][chosen]-trace['anchors'][key][chosen.cpu()].to(device)).abs().flatten(1).amax(-1)
                        errors[chosen,column]=error
            age = tick-triggers
            active = (triggers>=0)&(age>=0)&(age<8)&~terminal
            action = base.clone()
            if trace is not None:
                prefix = (triggers<0)|(age<0)
                action[prefix] = trace['action'][tick].to(device)[prefix]
                intended = base[active]+delta[ARGS.candidate]
                clipping[active] += ((intended<-1)|(intended>1)).any(-1).long()
                action[active] = intended.clamp(-1,1)
            ids_post = ((triggers>=0)&(age>=0)&(age<POST_WINDOW)&~terminal).nonzero(as_tuple=False).flatten()
            ids8 = ((triggers>=0)&(age>=0)&(age<8)&~terminal).nonzero(as_tuple=False).flatten()
            if len(ids8):
                packet['actions'][ids8,age[ids8]]=action[ids8]
                packet['base_actions'][ids8,age[ids8]]=base[ids8]
                packet['pd_targets'][ids8,age[ids8]]=task._action_to_pd_targets(action.clone())[ids8]
            if trace is not None: restore_rng(trace['rng'][tick]['before_physics'])
            r_physics = capture_rng()
            _,_,done,_ = self.env_step(self.env,action); done=done.bool().reshape(-1).to(device)
            tracker.step(task._target_states[:,2],contacts(task).bool().all(-1))
            after = physical()
            if not torch.isfinite(after).all(): raise FloatingPointError('nonfinite candidate states')
            if len(ids_post):
                packet['height'][ids_post,age[ids_post]]=after[ids_post,2]
                packet['pair'][ids_post,age[ids_post]]=after[ids_post,71]>.5
                packet['valid_steps'][ids_post,age[ids_post]]=True
                ids32=ids_post[age[ids_post]<32]; offsets=age[ids32]
                packet['trajectory'][ids32,offsets]=after[ids32]
                packet['native_q'][ids32,offsets]=task._dof_pos[ids32]
                packet['fingertip_positions'][ids32,offsets]=task._rigid_body_pos[ids32][:,tip_ids]
                packet['hand_base_pose'][ids32,offsets]=torch.cat((task._rigid_body_pos[ids32,base_id],task._rigid_body_rot[ids32,base_id]),-1)
            if trace is None:
                logs['action'].append(action.cpu()); logs['done'].append(done.cpu())
                rng.append(dict(before_reset=r_reset,before_action=r_action,before_physics=r_physics))
            else:
                prefix=(triggers<0)|(tick<triggers)
                errors[prefix,6]=torch.maximum(errors[prefix,6],(done!=trace['done'][tick].to(device))[prefix].float())
            terminal |= done; previous=action.clone()
            if tick%250==0: print(json.dumps(dict(tick=tick,anchors=int((triggers>=0).sum()),done=int(terminal.sum()),candidate=ARGS.candidate)),flush=True)
            if trace is None and terminal.all(): break
        else:
            if trace is None: raise RuntimeError('baseline episodes exceed bounded trace')
        assigned=(triggers>=0)
        if not packet['valid_steps'][assigned].all(): raise ValueError('assigned anchor has incomplete outcome; no post-treatment filtering allowed')
        packet.update(triggers=triggers,motion_id=motion,start_frame=start,rest_height=rest,
            prefix_errors=errors,clipped_steps=clipping,delta=delta,initial_fingerprint=initial_hash,
            candidate=ARGS.candidate,candidate_name=CANDIDATES[ARGS.candidate],control_dt=task.dt)
        packet=cpu_copy(packet); torch.save(packet,ARGS.run_dir/'panel.pt')
        if trace is None:
            value={k:torch.stack(v) for k,v in logs.items()}
            value.update(rng=rng,triggers=packet['triggers'],anchors={k:packet[k] for k in ('actor_obs','history')},initial_fingerprint=initial_hash)
            torch.save(value,ARGS.run_dir/'trace.pt')
        if fingerprint(self.model.state_dict())!=model_hash: raise ValueError('actor updated')
        if self.normalize_input and fingerprint(self.running_mean_std.state_dict())!=rms_hash: raise ValueError('normalizer updated')
        result=dict(anchors=int(assigned.sum()),motion_counts=[int((assigned&(motion==k)).sum()) for k in range(3)],
            ticks=tick+1,candidate=ARGS.candidate,wall_seconds=time.monotonic()-begin,
            model_fingerprint=model_hash,rms_fingerprint=rms_hash,initial_fingerprint=initial_hash,
            aligned_reference_tables=migrated_tables,physics_device=str(task.device),actor_device=str(next(self.model.parameters()).device),
            actor_and_rms_unchanged=True,prefix_errors_max=errors[assigned].amax(0).cpu().tolist() if assigned.any() else [],
            output_sha256=sha(ARGS.run_dir/'panel.pt'))
        (ARGS.run_dir/'result.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result),flush=True)


def main():
    global ARGS
    parser=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    parser.add_argument('--run-dir',type=Path,required=True); parser.add_argument('--reference',type=Path)
    parser.add_argument('--candidate',type=int,default=0,choices=range(len(CANDIDATES)))
    parser.add_argument('--wall-seconds',type=int,default=180)
    parser.add_argument('--diagnose-prefix',action='store_true')
    ARGS,remaining=parser.parse_known_args()
    if ARGS.reference is None and ARGS.candidate!=0: raise ValueError('reference is baseline')
    ARGS.run_dir.mkdir(parents=True,exist_ok=False)
    checkpoint=Path(remaining[remaining.index('--checkpoint')+1])
    if sha(checkpoint)!=SOURCE_SHA: raise ValueError('pinned self-trained source_e260 required')
    paths=[checkpoint,Path(__file__)]
    paths += [ROOT/p for p in (
        'src/task/cm-interaction-oracle/tools/run/collect_interventions.py',
        'src/task/cm-interaction-oracle/tools/run/run_oracle_y_candidate.sh',
        'src/task/cm-interaction-oracle/src/oracle_y_utility.py',
        'src/task/cm-interaction-oracle/src/intervention.py',
        'src/task/cm-interaction-oracle/src/consequence_sufficiency.py',
        'src/task/CmResidual/paired_evaluation.py',
        'src/task/CmResidual/physical_value_live.py',
        'src/task/CmResidual/physical_value_contract.py',
        'src/task/CmResidual/dexplore_cm_geometry.py',
        'third_party/DExplore/dexplore/evaluate.py',
        'third_party/DExplore/dexplore/env/tasks/dexplore_inspire.py',
        'third_party/DExplore/dexplore/env/tasks/base_dexplore_task.py',
        'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf',
        'third_party/DExplore/dexplore/data/assets/mjcf/airplane.urdf')]
    motion_root=Path(remaining[remaining.index('--motion_file')+1])
    motions=sorted(p/'interaction_hand_inspire.pt' for p in motion_root.iterdir() if p.is_dir())
    if len(motions)!=3: raise ValueError('three pinned canonical motion files required')
    paths += motions
    for flag in ('--cfg_env','--cfg_train'): paths.append(Path(remaining[remaining.index(flag)+1]))
    if ARGS.reference is not None: paths += [ARGS.reference/k for k in ('initial_state.pt','trace.pt','panel.pt')]
    inputs={str(p.resolve()):sha(p) for p in paths}
    manifest=dict(status='RUNNING',pid=os.getpid(),command=sys.argv,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        input_sha256=inputs,physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),no_training=True,candidate=ARGS.candidate)
    begin=time.monotonic()
    try:
        original.EvalPlayer=CandidatePlayer; sys.argv=[sys.argv[0],*remaining]; original.main()
        if any(sha(p)!=h for p,h in inputs.items()): raise ValueError('input drift')
        manifest.update(status='COMPLETED',inputs_unchanged=True)
    except BaseException as error:
        manifest.update(status='FAILED',error=repr(error)); raise
    finally:
        manifest['wall_seconds']=time.monotonic()-begin
        (ARGS.run_dir/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__': main()
