"""
Evaluate tracking success rate using the standard rl_games player pipeline.

This extends the DexplorePlayerContinuous to collect per-episode metrics
(reward, steps, survival, tracking errors) during inference.

Usage:
    python dexplore/evaluate.py \
        --task Dexplore_Inspire \
        --cfg_env dexplore/data/cfg/inspire_slow_slow_energy_reset_contact_table_adjust_parameter.yaml \
        --cfg_train dexplore/data/cfg/train/rlg/inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2.yaml \
        --checkpoint robot/inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/GRAB.pth \
        --headless --num_envs 64 --output eval_results.json
"""
import os
import sys
import json
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(_SCRIPT_DIR, '../../..')))

from isaacgym import gymapi  # noqa: must import before torch

import numpy as np
import torch

if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int

from utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
from utils.parse_task import parse_task

from rl_games.algos_torch import torch_ext
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import AlgoObserver
from rl_games.torch_runner import Runner
from rl_games.algos_torch import model_builder

from learning import dexplore_agent
from learning import dexplore_players
from learning import dexplore_models
from learning import dexplore_network_builder
from utils.reference_action import inspire_reference_action


def parse_eval_args():
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output', type=str, default='eval_results.json')
    parser.add_argument('--visualize-success-loop', action='store_true')
    parser.add_argument('--render-sleep', type=float, default=0.01)
    parser.add_argument('--transition-output', type=str)
    parser.add_argument('--reference-action-lead', type=int)
    parser.add_argument('--disable-early-termination', action='store_true')
    parser.add_argument('--cmlite-selector-checkpoint', type=str)
    parser.add_argument('--cmlite-selector-sha256', type=str)
    eval_args, remaining = parser.parse_known_args()
    if bool(eval_args.cmlite_selector_checkpoint) != bool(eval_args.cmlite_selector_sha256):
        parser.error('CmLite selector checkpoint and SHA256 must be specified together')
    if eval_args.cmlite_selector_checkpoint and eval_args.reference_action_lead is not None:
        parser.error('CmLite selector cannot be combined with reference action override')
    sys.argv = [sys.argv[0]] + remaining
    return eval_args


class EvalPlayer(dexplore_players.DexplorePlayerContinuous):
    """Extended player that collects per-episode metrics."""

    def __init__(self, config):
        super().__init__(config)
        self.episode_results = []
        self.visualize_success_loop = False
        self.visualize_render_sleep = 0.01
        self._focused_success_env = None
        self.transition_output = None
        self.reference_action_lead = None
        self.disable_early_termination = False
        self.cmlite_selector_checkpoint = None
        self.cmlite_selector_sha256 = None
        self._transitions = {}

    def _record_transition(self, **values):
        if self.transition_output is None:
            return
        for key, value in values.items():
            self._transitions.setdefault(key, []).append(value.detach().cpu())

    def _save_transitions(self):
        if self.transition_output is None:
            return
        payload = {key: torch.cat(parts, dim=0) for key, parts in self._transitions.items()}
        payload['schema'] = 'ref2dex.cmlite_transition.v1'
        output = os.path.abspath(self.transition_output)
        os.makedirs(os.path.dirname(output), exist_ok=True)
        torch.save(payload, output)
        print("REF2DEX_TRANSITIONS " + json.dumps({
            "output": output, "samples": int(payload['action'].shape[0]),
            "keys": sorted(key for key in payload if isinstance(payload[key], torch.Tensor)),
        }, sort_keys=True), flush=True)

    def _focus_success_env(self, task, env_id, lift, run_steps):
        if task.viewer is None or self._focused_success_env == env_id:
            return
        origin = task.gym.get_env_origin(task.envs[env_id])
        cam_pos = gymapi.Vec3(origin.x, origin.y - 1.35, origin.z + 1.25)
        cam_target = gymapi.Vec3(origin.x, origin.y, origin.z + 0.95)
        task.gym.viewer_camera_look_at(task.viewer, None, cam_pos, cam_target)
        self._focused_success_env = env_id
        print("REF2DEX_VISUAL_SUCCESS " + json.dumps({
            "env_id": env_id, "lift_m": float(lift),
            "consecutive_contact_steps": int(run_steps),
        }, sort_keys=True), flush=True)

    def run(self):
        evaluation_started = time.perf_counter()
        # Disable adaptive termination during evaluation (use fixed thresholds)
        if hasattr(self.env.task, '_adaptive_kappa_enabled'):
            self.env.task._adaptive_kappa_enabled = False
        if self.disable_early_termination:
            self.env.task._enable_early_termination = False

        n_games = self.games_num
        n_game_life = self.n_game_life
        is_deterministic = self.is_deterministic
        n_games = n_games * n_game_life
        if self.visualize_success_loop:
            n_games = 1_000_000_000
        games_played = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        need_init_rnn = self.is_rnn

        for _ in range(n_games):
            if games_played >= n_games:
                break

            obs_dict = self.env_reset()
            batch_size = 1
            batch_size = self.get_batch_size(obs_dict['obs'], batch_size)

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            cr = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            steps = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            cum_hand_err = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            cum_obj_err = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

            task = self.env.task
            selector = None
            if self.cmlite_selector_checkpoint is not None:
                from src.task.CmResidual.cmlite import FrozenCmLite
                from src.task.CmResidual.cmlite_policy_select import (
                    ProposalConfig, select_cmlite_action)
                selector = FrozenCmLite(self.cmlite_selector_checkpoint,
                                        task._dof_pos.device,
                                        self.cmlite_selector_sha256)
                selector_config = ProposalConfig()
                selector_histogram = torch.zeros(5, dtype=torch.long, device=self.device)
                stable_contact_steps = torch.zeros(batch_size, dtype=torch.long, device=self.device)
                selector_override_steps = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            episode_start_frame = task.start_times.clone()
            episode_motion_id = task.data_id.clone()
            initial_object_z = task._target_states[:, 2].clone()
            max_lift = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            max_contact_lift = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            contact_steps = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            airborne_steps = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            lift_contact_run = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            max_lift_contact_run = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            lift_success = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
            recorded_env = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

            done_indices = []

            for n in range(self.max_steps):
                obs_dict = self.env_reset(done_indices)
                if len(done_indices):
                    reset_ids = done_indices.reshape(-1).long()
                    initial_object_z[reset_ids] = task._target_states[reset_ids, 2]
                    episode_start_frame[reset_ids] = task.start_times[reset_ids]
                    episode_motion_id[reset_ids] = task.data_id[reset_ids]
                    if selector is not None:
                        stable_contact_steps[reset_ids] = 0

                if has_masks:
                    masks = self.env.get_action_mask()
                    action = self.get_masked_action(obs_dict, masks, is_deterministic)
                else:
                    action = self.get_action(obs_dict, is_deterministic)
                if self.reference_action_lead is not None:
                    action = inspire_reference_action(task, self.reference_action_lead)
                if selector is not None:
                    goal_index = (task.progress_buf + 1).clamp_max(task.hoi_data.shape[1] - 1)
                    goal_position = task.hoi_data[task.data_id, goal_index, 106:109]
                    action, _, selected_id = select_cmlite_action(
                        selector, task._dof_pos, action, task._target_states,
                        goal_position, actual_contact=stable_contact_steps >= 5,
                        config=selector_config)
                    active = ~recorded_env
                    selector_histogram += torch.bincount(selected_id[active], minlength=5)
                    selector_override_steps += (selected_id.ne(0) & active).long()
                q_before = task._dof_pos.clone()
                object_before = task._target_states.clone()
                progress_before = task.progress_buf.clone()
                data_id_before = task.data_id.clone()
                obs_dict, r, done, info = self.env_step(self.env, action)
                cr += r
                steps += 1

                lift = task._target_states[:, 2] - initial_object_z
                max_lift = torch.maximum(max_lift, lift)
                hand_contact = (task._contact_forces[:, task._contact_body_ids].norm(dim=-1) > 0.1).any(dim=-1)
                object_contact = task._tar_contact_forces.norm(dim=-1) > 0.1
                hand_object_contact = hand_contact & object_contact
                if selector is not None:
                    stable_contact_steps = torch.where(
                        hand_object_contact, stable_contact_steps + 1,
                        torch.zeros_like(stable_contact_steps))
                contact_steps += hand_object_contact.float()
                airborne_steps += (lift >= 0.03).float()
                max_contact_lift = torch.maximum(
                    max_contact_lift, torch.where(hand_object_contact, lift, torch.zeros_like(lift)))
                held_lift = (lift >= 0.03) & hand_object_contact
                lift_contact_run = torch.where(held_lift, lift_contact_run + 1,
                                               torch.zeros_like(lift_contact_run))
                max_lift_contact_run = torch.maximum(max_lift_contact_run, lift_contact_run)
                lift_success |= lift_contact_run >= 5
                self._record_transition(
                    q=q_before, action=action, object_state=object_before,
                    next_q=task._dof_pos, next_object_state=task._target_states,
                    hand_contact=hand_contact[:, None], object_contact=object_contact[:, None],
                    done=done.bool()[:, None], progress=progress_before[:, None],
                    data_id=data_id_before[:, None])
                if self.visualize_success_loop:
                    successful = (lift_contact_run >= 5).nonzero(as_tuple=False).reshape(-1)
                    if successful.numel():
                        chosen = int(successful[0].item())
                        self._focus_success_env(
                            task, chosen, lift[chosen].item(), lift_contact_run[chosen].item())

                # Collect tracking metrics
                if hasattr(self.env.task, 'metric_1'):
                    cum_hand_err += self.env.task.metric_1
                if hasattr(self.env.task, 'metric_2'):
                    cum_obj_err += self.env.task.metric_2

                self._post_step(info)

                if self.visualize_success_loop:
                    task.render()
                    time.sleep(self.visualize_render_sleep)

                all_done_indices = done.nonzero(as_tuple=False)
                done_indices = all_done_indices[::self.num_agents].reshape(-1)
                new_done_indices = done_indices[~recorded_env[done_indices]]
                done_count = len(new_done_indices)
                games_played += done_count

                if done_count > 0:
                    if self.is_rnn:
                        for s in self.states:
                            s[:, all_done_indices, :] = s[:, all_done_indices, :] * 0.0

                    for idx in new_done_indices:
                        i = idx.item()
                        ep_len = max(steps[i].item(), 1)
                        early_term = False
                        if hasattr(self.env.task, '_terminate_buf'):
                            early_term = self.env.task._terminate_buf[i].item() > 0

                        episode_result = {
                            'reward': cr[i].item(),
                            'env_id': i,
                            'motion_id': int(episode_motion_id[i].item()),
                            'start_frame': int(episode_start_frame[i].item()),
                            'steps': int(steps[i].item()),
                            'survived': not early_term,
                            'mean_hand_error': cum_hand_err[i].item() / ep_len,
                            'mean_obj_error': cum_obj_err[i].item() / ep_len,
                            'max_lift_m': max_lift[i].item(),
                            'max_contact_lift_m': max_contact_lift[i].item(),
                            'hand_object_contact_fraction': contact_steps[i].item() / ep_len,
                            'airborne_fraction': airborne_steps[i].item() / ep_len,
                            'max_lift_contact_run_steps': int(max_lift_contact_run[i].item()),
                            'lift_success': bool(lift_success[i].item()),
                        }
                        if selector is not None:
                            episode_result['selector_override_fraction'] = (
                                selector_override_steps[i].item() / ep_len)
                        self.episode_results.append(episode_result)
                    recorded_env[new_done_indices] = True

                    cr = cr * (1.0 - done.float())
                    steps = steps * (1.0 - done.float())
                    cum_hand_err = cum_hand_err * (1.0 - done.float())
                    cum_obj_err = cum_obj_err * (1.0 - done.float())
                    max_lift = max_lift * (1.0 - done.float())
                    max_contact_lift = max_contact_lift * (1.0 - done.float())
                    contact_steps = contact_steps * (1.0 - done.float())
                    airborne_steps = airborne_steps * (1.0 - done.float())
                    lift_contact_run = lift_contact_run * (1 - done.long())
                    max_lift_contact_run = max_lift_contact_run * (1 - done.long())
                    lift_success &= ~done.bool()
                    if selector is not None:
                        selector_override_steps *= 1 - done.long()

                    if batch_size // self.num_agents == 1 or games_played >= n_games:
                        break

        self._save_transitions()

        # Batched environments may finish several episodes on the step that
        # crosses n_games.  Keep the requested evaluation budget exact.
        self.episode_results = self.episode_results[:n_games]

        # Print and save results at end of run
        if self.episode_results:
            n_survived = sum(1 for r in self.episode_results if r['survived'])
            total = len(self.episode_results)
            success_rate = n_survived / total
            mean_reward = np.mean([r['reward'] for r in self.episode_results])
            std_reward = np.std([r['reward'] for r in self.episode_results])
            mean_steps = np.mean([r['steps'] for r in self.episode_results])
            mean_hand_err = np.mean([r['mean_hand_error'] for r in self.episode_results])
            mean_obj_err = np.mean([r['mean_obj_error'] for r in self.episode_results])
            lift_success_rate = np.mean([r['lift_success'] for r in self.episode_results])
            mean_max_lift = np.mean([r['max_lift_m'] for r in self.episode_results])
            mean_max_contact_lift = np.mean([r['max_contact_lift_m'] for r in self.episode_results])
            mean_contact_fraction = np.mean([r['hand_object_contact_fraction'] for r in self.episode_results])
            mean_airborne_fraction = np.mean([r['airborne_fraction'] for r in self.episode_results])
            mean_max_lift_contact_run = np.mean(
                [r['max_lift_contact_run_steps'] for r in self.episode_results])

            print(f"\n{'=' * 60}")
            print(f"EVALUATION RESULTS ({total} episodes)")
            print(f"{'=' * 60}")
            print(f"  Success Rate:    {success_rate:.1%}")
            print(f"  Mean Reward:     {mean_reward:.2f} +/- {std_reward:.2f}")
            print(f"  Mean Steps:      {mean_steps:.1f}")
            print(f"  Mean Hand Error: {mean_hand_err:.4f}")
            print(f"  Mean Obj Error:  {mean_obj_err:.4f}")
            print(f"  Lift Success:    {lift_success_rate:.1%}")
            print(f"  Mean Max Lift:   {mean_max_lift:.4f} m")
            print(f"  Max Contact Lift:{mean_max_contact_lift:.4f} m")
            print(f"  Contact Fraction:{mean_contact_fraction:.4f}")
            print(f"  Airborne Fraction:{mean_airborne_fraction:.4f}")
            print(f"  Lift Contact Run:{mean_max_lift_contact_run:.2f} steps")
            print(f"{'=' * 60}")

            # Save to file if output path set
            output_file = getattr(self, 'output_file', 'eval_results.json')
            summary = {
                'num_episodes': total,
                'success_rate': round(success_rate, 4),
                'mean_reward': round(float(mean_reward), 2),
                'std_reward': round(float(std_reward), 2),
                'mean_steps': round(float(mean_steps), 1),
                'mean_hand_error': round(float(mean_hand_err), 4),
                'mean_obj_error': round(float(mean_obj_err), 4),
                'lift_success_rate': round(float(lift_success_rate), 4),
                'mean_max_lift_m': round(float(mean_max_lift), 5),
                'mean_max_contact_lift_m': round(float(mean_max_contact_lift), 5),
                'mean_hand_object_contact_fraction': round(float(mean_contact_fraction), 5),
                'mean_airborne_fraction': round(float(mean_airborne_fraction), 5),
                'mean_max_lift_contact_run_steps': round(float(mean_max_lift_contact_run), 3),
                'lift_success_definition': 'object dz >= 0.03 m with hand+object contact for >=5 consecutive steps',
                'episode_sampling': 'first completed episode from every parallel environment',
                'early_termination_disabled': self.disable_early_termination,
                'selector_enabled': self.cmlite_selector_checkpoint is not None,
                'evaluation_wall_seconds': round(time.perf_counter() - evaluation_started, 3),
            }
            if self.cmlite_selector_checkpoint is not None:
                histogram = selector_histogram.cpu().tolist()
                summary['selector_histogram'] = histogram
                summary['selector_override_rate'] = round(
                    1 - histogram[0] / max(sum(histogram), 1), 5)
                summary['selector_contact_gate_steps'] = 5
                summary['selector_candidate_count'] = 5
            output = {'summary': summary, 'per_episode': self.episode_results}
            os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(output, f, indent=2)
            print(f"Results saved to {output_file}")


# Store output path globally so player can access it
_eval_output_file = 'eval_results.json'

# ---- Same run.py infrastructure ----
args = None
cfg = None
cfg_train = None


def create_rlgpu_env(**kwargs):
    from run import RLGPUEnv as _  # ensure registered
    sim_params = parse_sim_params(args, cfg, cfg_train)
    task, env = parse_task(args, cfg, cfg_train, sim_params)
    print(f'num_envs: {env.num_envs}, num_actions: {env.num_actions}, num_obs: {env.num_obs}')
    return env


# Import and re-register with our create fn
from run import RLGPUEnv, RLGPUAlgoObserver
vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
env_configurations.register('rlgpu', {
    'env_creator': lambda **kwargs: create_rlgpu_env(**kwargs),
    'vecenv_type': 'RLGPU',
})


def main():
    global args, cfg, cfg_train

    eval_args = parse_eval_args()
    set_np_formatting()
    args = get_args()

    # Force test mode
    args.play = True
    args.train = False
    args.test = True

    cfg, cfg_train, logdir = load_cfg(args)
    cfg_train['params']['seed'] = set_seed(
        cfg_train['params'].get("seed", -1),
        cfg_train['params'].get("torch_deterministic", False))

    if args.motion_file:
        cfg['env']['motion_file'] = args.motion_file

    cfg_train['params']['config']['train_dir'] = args.output_path

    # Build runner with EvalPlayer instead of standard player
    algo_observer = RLGPUAlgoObserver()
    runner = Runner(algo_observer)
    runner.algo_factory.register_builder('dexplore', lambda **kwargs: dexplore_agent.DexploreAgent(**kwargs))
    def _make_eval_player(params):
        p = EvalPlayer(params)
        p.output_file = eval_args.output
        p.visualize_success_loop = eval_args.visualize_success_loop
        p.visualize_render_sleep = eval_args.render_sleep
        p.transition_output = eval_args.transition_output
        p.reference_action_lead = eval_args.reference_action_lead
        p.disable_early_termination = eval_args.disable_early_termination
        p.cmlite_selector_checkpoint = eval_args.cmlite_selector_checkpoint
        p.cmlite_selector_sha256 = eval_args.cmlite_selector_sha256
        return p
    runner.player_factory.register_builder('dexplore', lambda **kwargs: _make_eval_player(**kwargs))
    if hasattr(model_builder, 'register_model'):
        model_builder.register_model('dexplore', dexplore_models.ModelDexploreContinuous)
        model_builder.register_network('dexplore', dexplore_network_builder.DexploreBuilder)
    else:
        runner.model_builder.model_factory.register_builder(
            'dexplore', lambda network, **kwargs:
            dexplore_models.ModelDexploreContinuous(network))
        runner.model_builder.network_factory.register_builder(
            'dexplore', lambda **kwargs: dexplore_network_builder.DexploreBuilder())

    # One completed episode per requested environment, with deterministic
    # actions and no unbounded player loop.
    cfg_train['params']['config']['player'] = {
        'games_num': cfg['env']['numEnvs'], 'deterministic': True,
        'print_stats': False, 'render': eval_args.visualize_success_loop,
        'render_sleep': eval_args.render_sleep,
    }

    runner.load(cfg_train)
    runner.reset()

    # The EvalPlayer.run() prints results and saves to eval_args.output internally
    runner.run(vars(args))


if __name__ == '__main__':
    main()
