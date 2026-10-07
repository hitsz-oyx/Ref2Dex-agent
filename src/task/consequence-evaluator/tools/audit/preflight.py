"""Read-only inventory for the ref1 continuous-rollout evaluator route."""
import argparse
import json
from pathlib import Path
import subprocess

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--asset-root', type=Path, default=ROOT)
    p.add_argument('--route-config', type=Path, default=ROOT / 'src/task/CmResidual/configs/multitrajectory_object_router_with_cup_probe.json')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    routes = json.loads(a.route_config.read_text())
    checkpoints = {name: dict(path=str(a.asset_root / spec['checkpoint']),
                              available=(a.asset_root / spec['checkpoint']).is_file(),
                              expected_sha256=spec['sha256'])
                   for name, spec in routes['experts'].items()}
    motion = a.asset_root / 'outputs/CmResidual/agent_contact_option_airplane_motions'
    local_assets = ROOT / 'third_party/DExplore/dexplore/data/assets'
    external_assets = Path('/home2/wyy/oyx_ws/Ref2Dex/third_party/DExplore/dexplore/data/assets')
    data_root = ROOT / 'outputs/cm-interaction-oracle'
    native = ROOT / 'outputs/cm-pointflow-effect-pretrain/ref5-data-expansion-20261007/native3d-r6/audit.json'
    try:
        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                       '--format=csv,noheader'], text=True).splitlines()
    except (OSError, subprocess.CalledProcessError):
        gpu = ['GPU inventory unavailable']
    missing = [value['path'] for value in checkpoints.values() if not value['available']]
    if not motion.is_dir():
        missing.append(str(motion))
    report = dict(status='BLOCKED_INPUTS' if missing else 'ASSETS_PRESENT_HASH_CHECK_REQUIRED',
                  task='consequence-evaluator', horizon=24, execution_horizon=8,
                  self_trained_experts=checkpoints, missing_inputs=missing,
                  old_oracle_rollout_directory=dict(path=str(data_root), available=data_root.is_dir()),
                  local_simulator_assets=dict(path=str(local_assets), available=local_assets.is_dir()),
                  external_simulator_assets=dict(path=str(external_assets), available=external_assets.is_dir(), mode='READ_ONLY'),
                  native3d_audit_available=native.is_file(), native3d_role='world-model pretraining; no robot task-quality supervision',
                  gpu_inventory=gpu,
                  reusable_code={'expert_loading': 'third_party/DExplore/dexplore/evaluate_object_router.py',
                                 'continuous_collection_reference': 'src/task/cm-interaction-oracle/tools/run/collect_interventions.py',
                                 'rigid_hand_geometry': 'src/task/cm-interaction-oracle/src/execution_geometry.py',
                                 'episode_hold_metrics': 'src/task/CmResidual/physical_value_contract.py'},
                  excluded_reuse=['old Y labels', 'forked candidate packs as continuous rollout',
                                  'missing outputs inferred from experiment summaries',
                                  'human motion as robot success/preference labels'])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
