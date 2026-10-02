#!/usr/bin/env python3
"""Actual randomized H10 feedback programs; no old macro-Cm used for new laws."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from collect_contact_consequences import build_player
from run_paired_evaluator_resolution import sha


def make_player(original, args, torch, gymtorch):
    from src.task.CmResidual.contact_relative_feedback import (
        ALLOCATION_TO_OPTION, OPTION_NAMES, feedback_candidates, relative_anchor, verify_translation_asset)
    from src.task.CmResidual.executable_contact_options import hold_target, obj_vertices, TableClearance
    from src.task.CmResidual.paired_evaluation import fingerprint
    from src.task.CmResidual.weight_normalized_contact import weight_normalized_contacts
    parent = build_player(original, args, torch, gymtorch)

    class RelativePlayer(parent):
        @torch.no_grad()
        def run(self):
            begin = time.monotonic(); torch.set_num_threads(2)
            torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
            task = self.env.task; n = task.num_envs; w = 2 * args.windows_per_stratum
            task._hybrid_init_prob = 0.; task._enable_early_termination = False; task._adaptive_kappa_enabled = False
            if n != 96 or self.is_rnn or abs(task.dt-1/30) > 1e-8 or not 1 <= args.windows_per_stratum <= 4:
                raise ValueError('bounded native execution contract')
            if set(getattr(task, 'object_name', ['airplane'])) != {'airplane'}:
                raise ValueError('airplane only')
            asset = Path(task.cfg['env']['asset']['assetRoot']) / task.robot_type
            asset_contract = verify_translation_asset(asset)
            names = task.gym.get_actor_dof_names(task.envs[0], task.humanoid_handles[0])
            if names[:6] != ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']:
                raise ValueError('native DOF order drift')
            assets = ROOT / 'third_party/DExplore/dexplore/data/assets/mjcf'
            geometry = TableClearance(obj_vertices(assets/'objects/airplane/airplane.obj', self.device),
                obj_vertices(assets/'objects/table/table.obj', self.device), getattr(task, 'ball_size', 1.))
            ids = torch.arange(n, device=self.device); obs = self.env_reset(ids)
            if self.get_batch_size(obs['obs'], 1) != n or obs['obs'].shape != (n, 1442):
                raise ValueError('native observation and initialized player batch')
            motion, start = task.data_id.clone(), task.start_times.clone()
            rest = task.hoi_refs[task.data_id, task.ref_index, 0, 108].clone()
            mass = torch.tensor([task.gym.get_actor_rigid_body_properties(e, h)[0].mass
                for e, h in zip(task.envs, task._target_handles)], device=self.device)
            gravity = abs(task.sim_params.gravity.z); body_ids = task._contact_body_ids
            key_ids = task._key_body_ids[[0, 3, 6, 9, 12, 15]]
            def contact_state():
                return weight_normalized_contacts(task._contact_forces[:, body_ids], task._tar_contact_forces, mass, gravity)
            def zeros(*shape): return torch.zeros(*shape, device=self.device)
            def integer(fill=0): return torch.full((n,), fill, dtype=torch.long, device=self.device)
            history, previous = zeros(n, 10, 69), zeros(n, 18)
            count, elapsed, arm, cooldown, contact_run = integer(), integer(-1), integer(), integer(), integer()
            ended = torch.zeros(n, dtype=torch.bool, device=self.device)
            strata_count = torch.zeros(n, 2, dtype=torch.long, device=self.device)
            anchor, relative = zeros(n, 18), zeros(n, 3)
            trigger = torch.full((n, w), -1, dtype=torch.long, device=self.device)
            assignment, allocation, steps = trigger.clone(), trigger.clone(), torch.zeros_like(trigger)
            pre, hist, anchors, relatives = zeros(n, w, 49), zeros(n, w, 10, 69), zeros(n, w, 18), zeros(n, w, 3)
            candidates, tables = zeros(n, w, 4, 18), zeros(n, w, 7)
            initial_clearance, future_clearance = zeros(n, w), zeros(n, w, 10)
            future, native_obs, future_obs = zeros(n, w, 10, 49), zeros(n, w, 10, 1442), zeros(n, w, 10, 1442)
            bank_log, actual, pd, corrections = zeros(n, w, 10, 6, 18), zeros(n, w, 10, 18), zeros(n, w, 10, 18), zeros(n, w, 10, 3, 3)
            initial_hand, initial_object = zeros(n, w, len(body_ids), 3), zeros(n, w, 3)
            future_hand, future_object, ratio = zeros(n, w, 10, len(body_ids), 3), zeros(n, w, 10, 3), zeros(n, w, 10, 2)
            bits = torch.zeros(n, w, 10, 2, dtype=torch.bool, device=self.device)
            terminal = torch.zeros(n, w, 10, dtype=torch.bool, device=self.device)
            initial_keys, initial_keyvel = zeros(n, w, 6, 3), zeros(n, w, 6, 3)
            future_keys, future_keyvel = zeros(n, w, 10, 6, 3), zeros(n, w, 10, 6, 3)
            before = fingerprint([dict(model=m.state_dict(), rms=r.state_dict()) for m, r in self.frozen_experts])
            allocator = torch.Generator(device=self.device).manual_seed(args.assignment_seed)
            mapping = torch.tensor(ALLOCATION_TO_OPTION, device=self.device)
            reset = ids[:0]; detections = 0
            for tick in range(args.max_steps):
                if time.monotonic()-begin > args.wall_seconds: raise TimeoutError('bounded relative feedback source')
                obs = self.env_reset(reset)
                roots = task._humanoid_root_states
                if roots[:, :6].abs().max() > 1e-6 or (roots[:, 6]-1).abs().max() > 1e-6:
                    raise ValueError('world-axis root must have identity reset transform')
                state = torch.cat((task._dof_pos.clone(), task._dof_vel.clone(), task._target_states.clone()), -1)
                contact = contact_state()[0].float()
                history = torch.cat((history[:, 1:], torch.cat((state, contact, previous), -1)[:, None]), 1)
                contact_run = torch.where(contact.bool().all(-1), contact_run+1, 0)
                bank = self.candidates(obs)
                clearance = geometry.clearance(task._target_states, task._table_states)
                lifted = (state[:, 38]-rest >= .03) & (clearance >= .002); stratum = lifted.long()
                eligible = (~ended) & (elapsed < 0) & (cooldown <= 0) & (contact_run >= 3)
                eligible &= (state[:, 38]-rest >= .005) & (tick >= 10) & (tick <= args.max_steps-10)
                eligible &= (task.max_episode_length[task.data_id]-task.progress_buf > 11)
                eligible &= strata_count[ids, stratum] < args.windows_per_stratum
                rows = eligible.nonzero().flatten(); slot = count[rows]
                if len(rows):
                    anchor[rows] = hold_target(task._dof_pos[rows], task._pd_action_offset, task._pd_action_scale)
                    relative[rows] = relative_anchor(state[rows])
                    options, _ = feedback_candidates(bank[rows], state[rows], anchor[rows], relative[rows],
                        task._pd_action_offset, task._pd_action_scale, task.dt)
                    # No initial-command difference filter: position programs agree
                    # initially but react differently to later observations.
                    draw = torch.randint(6, (len(rows),), device=self.device, generator=allocator)
                    selected = mapping[draw]; detections += len(rows)
                    trigger[rows, slot], assignment[rows, slot], allocation[rows, slot] = tick, selected, draw
                    arm[rows], elapsed[rows] = selected, 0
                    pre[rows, slot], hist[rows, slot], candidates[rows, slot] = state[rows], history[rows], options
                    anchors[rows, slot], relatives[rows, slot] = anchor[rows], relative[rows]
                    tables[rows, slot], initial_clearance[rows, slot] = task._table_states[rows, :7], clearance[rows]
                    initial_hand[rows, slot], initial_object[rows, slot] = task._contact_forces[rows][:, body_ids], task._tar_contact_forces[rows]
                    initial_keys[rows, slot], initial_keyvel[rows, slot] = task._rigid_body_pos[rows][:, key_ids], task._rigid_body_vel[rows][:, key_ids]
                    strata_count[rows, stratum[rows]] += 1
                live = (elapsed >= 0) & ~ended; rows = live.nonzero().flatten()
                slot, step = count[rows], elapsed[rows]; action = bank[:, 4].clone()
                if len(rows):
                    programmes, correction = feedback_candidates(bank[rows], state[rows], anchor[rows], relative[rows],
                        task._pd_action_offset, task._pd_action_scale, task.dt)
                    action[rows] = programmes[torch.arange(len(rows), device=self.device), arm[rows]]
                    bank_log[rows, slot, step], native_obs[rows, slot, step] = bank[rows], obs['obs'][rows]
                    corrections[rows, slot, step] = correction
                applied = action.clone(); targets = task._action_to_pd_targets(applied.clone()).clone()
                holding = live & (arm != 0)
                if holding.any() and not torch.allclose(targets[holding, 3:6], anchor[holding, 3:6], atol=2e-5, rtol=1e-5):
                    raise ValueError('fixed rotation native target not realizable')
                actual[rows, slot, step], pd[rows, slot, step] = applied[rows], targets[rows]
                next_obs, _, done, _ = self.env_step(self.env, action); done = done.bool().reshape(-1)
                future_obs[rows, slot, step] = next_obs['obs'][rows] if isinstance(next_obs, dict) else next_obs[rows]
                future[rows, slot, step] = torch.cat((task._dof_pos.clone(), task._dof_vel.clone(), task._target_states.clone()), -1)[rows]
                observed_bits, observed_ratio = contact_state()
                bits[rows, slot, step], ratio[rows, slot, step] = observed_bits[rows], observed_ratio[rows]
                future_hand[rows, slot, step], future_object[rows, slot, step] = task._contact_forces[rows][:, body_ids], task._tar_contact_forces[rows]
                future_keys[rows, slot, step], future_keyvel[rows, slot, step] = task._rigid_body_pos[rows][:, key_ids], task._rigid_body_vel[rows][:, key_ids]
                future_clearance[rows, slot, step] = geometry.clearance(task._target_states, task._table_states)[rows]
                terminal[rows, slot, step] = done[rows]; steps[rows, slot] += 1
                elapsed[live] += 1; complete = live & (elapsed == 10)
                count[complete] += 1; elapsed[complete], cooldown[complete] = -1, 6
                cooldown = (cooldown-1).clamp_min(0); ended |= done
                reset, previous = done.nonzero().flatten(), applied
                if tick % 50 == 0:
                    print(json.dumps(dict(tick=tick, complete_windows=int((steps == 10).sum()))), flush=True)
                if (ended | (count >= w)).all(): break
            valid = steps == 10
            if ((steps > 0) & (~valid | terminal.any(-1))).any(): raise ValueError('incomplete/reset window')
            saved_obs = native_obs[valid].flatten(0, 1); saved_bank = bank_log[valid].flatten(0, 1)
            replay_error = 0.
            for index in range(0, len(saved_obs), 96):
                batch = saved_obs[index:index+96]; length = len(batch)
                padded = torch.cat((batch, batch[:1].expand(96-length, -1)), 0) if length < 96 else batch
                replay_error = max(replay_error, float((self.candidates({'obs':padded})[:length]-saved_bank[index:index+length]).abs().max()))
            if replay_error > 2e-5: raise ValueError('frozen six-expert observation replay')
            after = fingerprint([dict(model=m.state_dict(), rms=r.state_dict()) for m, r in self.frozen_experts])
            if before != after: raise ValueError('expert drift')
            env = ids[:, None].expand(-1, w)[valid]; selected = assignment[valid]
            clear = (initial_clearance[valid] >= .002) & (pre[valid][:, 38]-rest[env] >= .03)
            support = bits[valid][:, -3:].all(-1).all(-1) & (future_clearance[valid][:, -3:] >= .002).all(-1)
            score = ((future[valid][:, -3:, 38].amin(-1)-rest[env]).clamp_min(0)*support - (pre[valid][:, 38]-rest[env]).clamp_min(0))*1000
            payload = dict(schema='ref2dex.contact_relative_feedback_source.v1', seed=args.seed, assignment_seed=args.assignment_seed,
                smoke_only=args.smoke_only, episode_id=[f's{args.seed}/env{int(i)}/first' for i in env.cpu()], env_id=env.cpu(),
                motion_id=motion[env].cpu(), start_frame=start[env].cpu(), trigger=trigger[valid].cpu(),
                window_index=torch.arange(w, device=self.device)[None].expand(n, -1)[valid].cpu(),
                split_group_seed=13671, split_group_bucket=torch.tensor([int(hashlib.sha256(f'13671/{int(i)}/{int(j)}'.encode()).hexdigest()[:8], 16)%100 for i,j in zip(motion[env].cpu(),start[env].cpu())]),
                state=pre[valid].cpu(), history=hist[valid].cpu(), hold_target=anchors[valid].cpu(), initial_relative=relatives[valid].cpu(),
                candidate_actions=candidates[valid].cpu(), table_pose=tables[valid].cpu(), assignment=selected.cpu(), allocation=allocation[valid].cpu(),
                allocation_to_option=list(ALLOCATION_TO_OPTION), propensity=torch.tensor([1/3,1/3,1/6,1/6],device=self.device)[selected].cpu(),
                allocation_probabilities=torch.full((len(env),6),1/6),
                initial_clearance=initial_clearance[valid].cpu(), future_clearance=future_clearance[valid].cpu(),
                future_state=future[valid].cpu(), future_contact=bits[valid].cpu(), future_done=terminal[valid].cpu(),
                native_observation=native_obs[valid].cpu(), future_native_observation=future_obs[valid].cpu(), expert_bank=bank_log[valid].cpu(),
                actual_action=actual[valid].cpu(), actual_pd_targets=pd[valid].cpu(), requested_corrections=corrections[valid].cpu(),
                pd_offset=task._pd_action_offset.cpu(), pd_scale=task._pd_action_scale.cpu(),
                initial_hand_force=initial_hand[valid].cpu(), initial_object_force=initial_object[valid].cpu(),
                future_hand_force=future_hand[valid].cpu(), future_object_force=future_object[valid].cpu(), future_force_ratio=ratio[valid].cpu(),
                initial_key_positions=initial_keys[valid].cpu(), initial_key_velocities=initial_keyvel[valid].cpu(),
                future_key_positions=future_keys[valid].cpu(), future_key_velocities=future_keyvel[valid].cpu(),
                mass_kg=mass[env].cpu(), gravity_magnitude=gravity, control_dt_seconds=task.dt,
                contact_collection=int(task.sim_params.physx.contact_collection), substeps=task.sim_params.substeps, rest_z=rest[env].cpu(),
                option_names=list(OPTION_NAMES), gain=.5, native_asset_sha256=sha(asset), asset_contract=asset_contract,
                root_identity_verified=True, native_dof_names=names, object_scale=getattr(task,'ball_size',1.),
                outcome=dict(initially_clear=clear.cpu(), supported_change_mm=score.cpu(), retained_clear=support.cpu()),
                sampling_contract='first episode currentjoint3, rest>=5mm, warmup10, H10+1 remaining; max per-stratum;6tickcooldown',
                intervention='H10 current-relative translation around fresh Cup; start-rotation anchor; base outside fragment',
                frozen_experts=True, expert_fingerprint_before=before, expert_fingerprint_after=after,
                cm_used=False, optimizer_used=False, model_training=False, assignment_after_observation=True)
            torch.save(payload, args.output/'records.pt')
            result = dict(run_status='COMPLETED', smoke_only=args.smoke_only, rows=len(env), episodes=len(set(payload['episode_id'])),
                arms=torch.bincount(selected,minlength=4).tolist(), initially_clear=int(clear.sum()), detections=detections,
                saved_observation_expert_replay_max_error=replay_error, record_sha256=sha(args.output/'records.pt'),
                elapsed_seconds=time.monotonic()-begin, frozen_experts=True, cm_used=False)
            (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result),flush=True)
    return RelativePlayer


def main():
    p=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    p.add_argument('--output-dir',dest='output',type=Path,required=True);p.add_argument('--panel-seed',dest='seed',type=int,required=True)
    p.add_argument('--assignment-seed',type=int,required=True);p.add_argument('--windows-per-stratum',type=int,default=1)
    p.add_argument('--max-steps',type=int,default=60);p.add_argument('--wall-seconds',type=int,default=180);p.add_argument('--smoke-only',action='store_true')
    args,remaining=p.parse_known_args();base=(ROOT/'src/task/CmResidual/research/contact_consequence/output').resolve()
    if base not in args.output.resolve().parents or args.output.is_symlink():raise ValueError('owned unique output')
    args.output.mkdir(parents=True,exist_ok=False);begin=time.monotonic()
    manifest=dict(run_status='STARTED',pid=os.getpid(),command=sys.argv,gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        smoke_only=args.smoke_only,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    path=args.output/'run_manifest.json';path.write_text(json.dumps(manifest,indent=2)+'\n')
    try:
        sys.path.insert(0,str(ROOT/'third_party/DExplore/dexplore'));from isaacgym import gymtorch
        import torch
        import evaluate as original
        original.EvalPlayer=make_player(original,args,torch,gymtorch);sys.argv=[sys.argv[0],*remaining]
        original.main();manifest['run_status']='COMPLETED'
    except BaseException as e:manifest.update(run_status='FAILED',error=repr(e));raise
    finally:manifest['elapsed_seconds']=time.monotonic()-begin;path.write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
