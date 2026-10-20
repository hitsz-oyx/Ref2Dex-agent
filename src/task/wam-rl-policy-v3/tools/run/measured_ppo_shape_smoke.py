"""Bounded offline shape smoke for the measured 421/686 PPO bridge."""
import argparse
import json
import os
from pathlib import Path


TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-checkpoint", type=Path, required=True)
    parser.add_argument("--scheduled-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.exists() or ROOT / "outputs/wam-rl-policy-v3" not in args.output.parents:
        raise ValueError("fresh task-local output required")
    args.output.mkdir(parents=True)

    import numpy as np
    import torch
    from wam_rl_policy_v3.measured_ppo import MeasuredCritic, MeasuredGaussianActor, update_measured
    from wam_rl_policy_v3.measured_reward_bridge import MeasuredRewardTracker, reward_features
    from wam_rl_policy_v3.task_runtime import TaskController
    from wam_rl_policy_v3.task_spec import TaskSpec

    torch.set_num_threads(2)
    reference = np.repeat(np.eye(4, dtype=np.float32)[None], 8, axis=0)
    reference[:, 0, 3] = np.arange(8, dtype=np.float32) * 0.01
    points = np.stack((np.linspace(-.02, .02, 32), np.sin(np.linspace(0, 3.1, 32)) * .01,
                       np.ones(32) * .01), axis=1)
    spec = TaskSpec.from_reference(reference, source_fps=30, terminal_mode="HOLD", trim_interaction=False)
    reward_values = []
    controller = TaskController(spec, points)
    tracker = MeasuredRewardTracker()
    hand = np.zeros((11, 3), np.float32)
    velocity = np.zeros(6, np.float32)
    contact = np.zeros((5, 3), np.float32)
    for tick in range(6):
        pose = reference[min(tick, len(reference) - 1)].copy()
        pose[0, 3] += tick * 0.001
        decision = controller.observe(pose, hand, velocity, tick / 30.0,
                                      contact_forces=contact, force_valid=False)["decision"]
        features = reward_features(pose, hand, points, decision, velocity, contact)
        if tracker.previous_decision is None:
            tracker.initialize(decision, features)
            value = 0.0
        else:
            value, _ = tracker.step(decision, features, np.zeros(12, np.float32))
        reward_values.append(value)
    reward_values = torch.tensor(reward_values, dtype=torch.float32).reshape(6, 1).repeat(1, 2)

    results = {}
    for label, checkpoint, with_tau in (("current421", args.current_checkpoint, False),
                                         ("scheduled686", args.scheduled_checkpoint, True)):
        packet = torch.load(checkpoint.resolve(), map_location="cpu", weights_only=False)
        actor = MeasuredGaussianActor.from_bc_packet(packet, with_tau=with_tau)
        critic = MeasuredCritic(actor.observation_dim)
        torch.manual_seed(246)
        observations = torch.randn(6, 2, actor.observation_dim)
        with torch.no_grad():
            action, logp, mean, std = actor.sample(observations.flatten(0, 1))
            action = action.reshape(6, 2, 12)
            logp = logp.reshape(6, 2)
            mean = mean.reshape(6, 2, 12)
            std = std.reshape(6, 2, 12)
            value = critic(observations.flatten(0, 1)).reshape(6, 2)
            next_observation = torch.randn(2, actor.observation_dim)
            bootstrap = critic(next_observation)
        batch = dict(observation=observations, action=action, logp=logp, mean=mean, std=std,
                     value=value, reward=reward_values, done=torch.zeros(6, 2, dtype=torch.bool),
                     bootstrap=bootstrap)
        update = update_measured(actor, critic, batch, torch.optim.Adam(actor.parameters(), 1e-5),
                                 torch.optim.Adam(critic.parameters(), 1e-4), epochs=1)
        results[label] = dict(observation_dim=actor.observation_dim, action_dim=actor.action_dim,
                              checkpoint=str(checkpoint.resolve()), update=update)
    manifest = dict(schema="ref2dex.inspire-measured-ppo-shape-smoke.v1", status="COMPLETED",
                    git_commit=os.popen("git rev-parse HEAD").read().strip(), controls=6,
                    reward_source="causal TaskController + MeasuredRewardTracker synthetic transition",
                    reward_values=reward_values[:, 0].tolist(), results=results,
                    native_reward_smokes=[
                        "outputs/wam-rl-policy-v2/inspire-current421-reward-smoke-20261020-r1/manifest.json",
                        "outputs/wam-rl-policy-v2/inspire-scheduled686-reward-smoke-20261020-r1/manifest.json",
                    ],
                    task_success_claim=False)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
