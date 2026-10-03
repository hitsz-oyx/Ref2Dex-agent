"""Engineering native prefix replay and complete attributed-contact recording."""
import argparse
import json
import sys
import time
import types
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--env-config', type=Path, required=True)
    p.add_argument('--references-manifest', type=Path, required=True)
    p.add_argument('--policy-checkpoint', type=Path, required=True)
    p.add_argument('--seed', type=int, default=751)
    p.add_argument('--envs', type=int, default=48)
    p.add_argument('--ticks', type=int, default=72)
    p.add_argument('--physics', choices=('cpu','gpu'), default='gpu')
    p.add_argument('--source', type=Path)
    p.add_argument('--options', type=Path)
    p.add_argument('--decision', type=int)
    p.add_argument('--oracle-horizon',type=int,default=32)
    p.add_argument('--contact-window-only',action='store_true')
    p.add_argument('--retain-pd-target',action='store_true')
    args = p.parse_args()
    if args.output.exists() or ROOT not in args.output.resolve().parents or args.envs % 3:
        raise ValueError('unique owned balanced output')
    args.output.mkdir()
    begin = time.monotonic()
    from isaacgym import gymtorch
    import torch
    import numpy as np
    from scripts.run_contact_response_probe import sha
    from src.task.CmResidual.oracle_native import create_session
    from src.task.CmResidual.observation_hold_policy import hold_context, canonical_action
    from src.task.CmResidual.dexplore_bc_policy import normalized_action
    from src.task.CmResidual.reference_target_policy import target_from_residual
    from src.task.CmResidual.static_hold_feasibility import static_pd_action, mesh_vertices
    from src.task.CmResidual.continuous_critic_cm import executable_target
    from src.task.CmResidual.tabletop_clearance import clearance
    task, model, checkpoint, initial, metadata = create_session(args)
    torch.save(initial, args.output/'initial.pt')
    (args.output/'physical_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    old = torch.load(args.source/'trace.pt', map_location='cpu', weights_only=False) if args.source else None
    if old is not None:
        expected = torch.load(args.source/'initial.pt', map_location='cpu', weights_only=False)
        for key in ('root_states','dof_state','motion','offsets','phase_stop','lift_start'):
            if not torch.equal(expected[key], initial[key]):
                raise ValueError('identical native initial packet: '+key)
    options = np.load(args.options) if args.options else np.zeros((args.envs,12), np.float32)
    if options.shape != (args.envs,12) or not np.isfinite(options).all():
        raise ValueError('fixed option requests')
    options = torch.tensor(options, dtype=torch.float32)
    stops = initial['phase_stop'][initial['motion']]
    steps = initial['lift_start'][initial['motion']]-8
    if args.decision is not None:
        steps = torch.full_like(steps,args.decision)
    mean = checkpoint['observation_mean'].to('cuda')
    std = checkpoint['observation_std'].to('cuda')
    asset = ROOT/'third_party/DExplore/dexplore/data/assets/mjcf/objects'
    vertices = mesh_vertices(asset/'airplane/airplane.obj').to('cuda')
    table_vertices = mesh_vertices(asset/'table/table.obj').to('cuda')
    table = initial['table_root'].to('cuda')
    raw_contacts, contact_offsets, physics_states = [], [0], []
    trace = {key: [] for key in ('context','object_root','native_q','native_dq','target','base_target',
             'model_residual','action','request','clearance','rigid_state','net_force')}
    physics_frames = []

    def step_physics(session):
        for subtick in range(session.control_freq_inv):
            session.render()
            session.gym.simulate(session.sim)
            session.gym.fetch_results(session.sim, True)
            session.gym.refresh_rigid_body_state_tensor(session.sim)
            session.gym.refresh_net_contact_force_tensor(session.sim)
            rigid = session._rigid_body_state.reshape(args.envs, -1, 13).clone()
            force = gymtorch.wrap_tensor(session.gym.acquire_net_contact_force_tensor(session.sim)).reshape(args.envs,-1,3).clone()
            count = 0
            for env_index, env in enumerate(session.envs):
                if args.contact_window_only and not int(steps[env_index]) <= len(trace['context']) < int(steps[env_index])+args.oracle_horizon:
                    continue
                raw = session.gym.get_env_rigid_contacts(env).copy()
                raw_contacts.append(raw)
                count += len(raw)
                physics_frames.append(dict(tick=len(trace['context']),subtick=subtick,env=env_index,
                                           offset=contact_offsets[-1],count=len(raw)))
                contact_offsets.append(contact_offsets[-1]+len(raw))
            physics_states.append(dict(rigid=rigid, net_force=force))
    task._physics_step = types.MethodType(step_physics, task)
    pd_error = 0.
    try:
        with torch.no_grad():
            for tick in range(args.ticks):
                if time.monotonic()-begin > 240:
                    raise TimeoutError('bounded native engineering phase')
                context = hold_context(task, initial['phase_stop'])
                residual = canonical_action(normalized_action(model, context.to('cuda'), mean, std).cpu())
                reference = task.hoi_refs[task.data_id, task.ref_index,
                    torch.minimum(task.progress_buf+1, stops),119:137]
                base = target_from_residual(reference, residual, task.dof_limits_lower, task.dof_limits_upper)
                goal = base.clone()
                if old is not None:
                    before = tick < steps
                    goal[before] = old['target'][tick,before]
                active = (tick >= steps) & (tick < stops+30)
                request = torch.where(active[:,None], options, torch.zeros_like(options))
                if torch.any(request):
                    goal[active] = executable_target(base[active], request[active], task._dof_pos[active],
                            task.dof_limits_lower, task.dof_limits_upper, task._pd_action_scale)
                elif old is not None:
                    goal[active] = old['target'][tick,active]
                action, error = static_pd_action(task, goal)
                pd_error = max(pd_error,error)
                task.step(action.clone())
                if task.reset_buf.any() or not torch.equal(task.progress_buf, torch.full_like(task.progress_buf,tick+1)):
                    raise ValueError('native progress/no terminal reset')
                separation = clearance(task._target_states.to('cuda'),table,vertices,table_vertices).cpu()
                values = dict(context=context, object_root=task._target_states, native_q=task._dof_pos,
                              native_dq=task._dof_vel,target=goal,base_target=base,model_residual=residual,
                              action=action,request=request,clearance=separation,
                              rigid_state=physics_states[-1]['rigid'],net_force=physics_states[-1]['net_force'])
                for key, value in values.items():
                    if not torch.isfinite(value).all():
                        raise ValueError('finite trace: '+key)
                    trace[key].append(value.cpu().clone())
                if tick % 24 == 23:
                    print(json.dumps(dict(tick=tick+1, contact_records=contact_offsets[-1])),flush=True)
        torch.save({key: torch.stack(value) for key,value in trace.items()},args.output/'trace.pt')
        torch.save(dict(rigid_state=torch.stack([r['rigid'] for r in physics_states]),
                        net_force=torch.stack([r['net_force'] for r in physics_states])),args.output/'physics_states.pt')
        raw = np.concatenate(raw_contacts)
        np.save(args.output/'contacts.npy',raw)
        (args.output/'contact_frames.json').write_text(json.dumps(physics_frames)+'\n')
        result = dict(run_status='COMPLETED', engineering_only=True, physics=args.physics,
                      cpu_data_pipeline=True, inference_device='cuda',seed=args.seed,envs=args.envs,
                      ticks=args.ticks, physics_frames=args.ticks*2, raw_contacts=len(raw),
                      max_pd_goal_error=pd_error, no_official_actor_loaded=True,
                      no_state_writes_after_frame0=True, policy_sha256=sha(args.policy_checkpoint),
                      input_options_sha256=sha(args.options) if args.options else None,
                      decision=args.decision,oracle_horizon=args.oracle_horizon,
                      contact_window_only=args.contact_window_only,
                      wall_seconds=time.monotonic()-begin)
        (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
    finally:
        task.gym.destroy_sim(task.sim)


if __name__ == '__main__':
    main()
