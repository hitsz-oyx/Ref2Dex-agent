"""Matched deterministic first-episode evaluation on the native task."""
import json
import random
import time
import numpy as np
import torch


def evaluate(env, actor, args, iteration):
    actor.eval()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    env.task.reset_buf[:] = 1
    started = time.monotonic()
    with torch.inference_mode():
        # Match the existing replay protocol: warm simulator buffers once,
        # then reset again from the same RNG state before measuring episodes.
        env.base.reset()
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        env.task.reset_buf[:] = 1
        observation = env.reset()
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
    report=dict(iteration=iteration,episodes=len(records),seed=args.seed,
        deterministic_actions=True,optimizer_steps=0,random_time=False,
        native_success_rate=float(np.mean([r['native_success'] for r in records])),
        held60_lift10cm_rate=float(np.mean([r['held60_lift10cm'] for r in records])),
        workspace_failures=sum(r['workspace_failure'] for r in records),
        mean_return=float(returns.mean()),elapsed_s=time.monotonic()-started,
        scope='fixed TRAIN panel; first episodes only; not held-out generalization',records=records)
    (args.output/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)
    return [report]
