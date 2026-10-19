"""Matched deterministic first-episode evaluation on the native task."""
import json
import random
import sys
import time
import numpy as np
import torch


def evaluate(env, actor, args, iteration):
    actor.eval()
    # Evaluation-only reset RNG: independent of CUDA/JIT initialization history.
    reset_rng = np.random.RandomState(args.seed)
    task_module = sys.modules[type(env.task).__module__]
    def reset_random(lower, upper, shape, device):
        values = reset_rng.uniform(lower, upper, tuple(shape)).astype(np.float32)
        return torch.as_tensor(values, device=device)
    task_module.torch_rand_float = reset_random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    env.task.reset_goal_buf[:] = 0
    env.task.root_state_tensor[env.task.table_indices, 7:13] = 0
    env.task.reset_buf[:] = 1
    started = time.monotonic()
    with torch.inference_mode():
        # Match the existing replay protocol: warm simulator buffers once,
        # then reset again from the same RNG state before measuring episodes.
        env.base.reset()
        reset_rng.seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        # Native pre_physics_step otherwise conditionally draws another random
        # goal reset before the full reset, depending on warmup terminal flags.
        env.task.reset_goal_buf[:] = 0
        # Native reset restores table pose but leaves its old velocity intact.
        env.task.root_state_tensor[env.task.table_indices, 7:13] = 0
        env.task.reset_buf[:] = 1
        observation = env.reset()
        # Canonicalize the table state after the reset physics tick as well.
        # Native reset restores its pose but permits residual table velocity.
        from isaacgym import gymtorch
        table_ids = env.task.table_indices.to(torch.int32)
        env.task.root_state_tensor[table_ids.long()] = env.task.saved_root_tensor[table_ids.long()]
        env.task.root_state_tensor[table_ids.long(), 7:13] = 0
        env.task.gym.set_actor_root_state_tensor_indexed(env.task.sim,
            gymtorch.unwrap_tensor(env.task.root_state_tensor),
            gymtorch.unwrap_tensor(table_ids), len(table_ids))
        initial = env.snapshot()
        np.savez_compressed(args.output / 'initial-state.npz',
            observation=observation.cpu().numpy(), pose=initial['pose'],
            hand=initial['hand'], goal=initial['goal'],
            root=env.task.root_state_tensor.cpu().numpy() if hasattr(env.task,'root_state_tensor') else np.empty(0))
        finished = np.zeros(env.num_envs, bool)
        returns = np.zeros(env.num_envs)
        records = []
        horizon = int(env.task.max_episode_length) + 2
        for tick in range(horizon):
            action = actor.act_inference(observation)
            observation, reward, done, _ = env.step(action)
            returns[~finished] += reward.cpu().numpy()[~finished]
            terminal = done.cpu().numpy().astype(bool)
            for i in np.flatnonzero(terminal & ~finished):
                records.append(dict(env=int(i),steps=tick+1,
                    native_success=bool(env.task.successes[i].item()>0),
                    held60_lift10cm=bool(env.longest[i]>=60),
                    workspace_failure=bool(env.workspace_invalid[i]),
                    reward=float(returns[i])))
            finished |= terminal
            if finished.all():break
        if not finished.all():raise RuntimeError('First episodes did not terminate within native horizon')
    records.sort(key=lambda r:r['env'])
    report=dict(iteration=iteration,checkpoint=str(args.rl_checkpoint),episodes=len(records),seed=args.seed,
        deterministic_actions=True,optimizer_steps=0,random_time=False,
        native_only=env.native_only,observation_dim=env.num_observations,
        reset_protocol="cpu-reset-rng-and-canonical-table.v1",
        native_success_rate=float(np.mean([r['native_success'] for r in records])),
        held60_lift10cm_rate=float(np.mean([r['held60_lift10cm'] for r in records])),
        workspace_failures=sum(r['workspace_failure'] for r in records),
        mean_return=float(returns.mean()),elapsed_s=time.monotonic()-started,
        scope='fixed TRAIN panel; first episodes only; not held-out generalization',records=records)
    (args.output/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)
    return [report]
