"""Check real SDK shape ownership against Inspire collision-filter assignments.

Asset metadata only: no simulation step, model fitting or checkpoint loading.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    assert ROOT in output.parents and not output.exists()
    from isaacgym import gymapi
    gym = gymapi.acquire_gym()
    parameters = gymapi.SimParams()
    parameters.up_axis = gymapi.UP_AXIS_Z
    parameters.dt = 1 / 60
    parameters.physx.use_gpu = False
    sim = gym.create_sim(0, -1, gymapi.SIM_PHYSX, parameters)
    assert sim is not None
    try:
        options = gymapi.AssetOptions()
        options.angular_damping = .01
        options.max_angular_velocity = 100.
        options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        options.fix_base_link = True
        options.disable_gravity = gymapi.RIGID_BODY_DISABLE_GRAVITY
        asset_dir = ROOT / 'third_party/DExplore/dexplore/data/assets/inspire_hand_new'
        asset = gym.load_asset(sim, str(asset_dir), 'inspire_hand_right.urdf', options)
        assert asset is not None
        env = gym.create_env(sim, gymapi.Vec3(-5, -5, 0), gymapi.Vec3(5, 5, 5), 1)
        actor = gym.create_actor(env, asset, gymapi.Transform(), 'hand', 0, 1, 0)
        names = gym.get_actor_rigid_body_names(env, actor)
        ranges = gym.get_actor_rigid_body_shape_indices(env, actor)
        shape_count = len(gym.get_actor_rigid_shape_properties(env, actor))
        owners = [None] * shape_count
        for name, span in zip(names, ranges):
            start = int(span.start if hasattr(span, 'start') else span['start'])
            count = int(span.count if hasattr(span, 'count') else span['count'])
            for index in range(start, start + count):
                assert owners[index] is None
                owners[index] = name
        assert all(owner is not None for owner in owners)
        source_path = ROOT / 'third_party/DExplore/dexplore/env/tasks/dexplore_inspire.py'
        source = source_path.read_text()
        tree = ast.parse(source)
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Dexplore_Inspire')
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '_apply_collision_filter')
        namespace = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source_path), 'exec'), namespace)
        namespace['_apply_collision_filter'](SimpleNamespace(gym=gym), env, actor)
        properties = gym.get_actor_rigid_shape_properties(env, actor)
        rows = []
        for index, (owner, prop) in enumerate(zip(owners, properties)):
            excluded = ('thumb' in owner and 'distal' in owner) or ('thumb' not in owner and 'intermediate' in owner)
            expected = 3 if excluded else 2
            rows.append(dict(shape_index=index, actual_body_name=owner,
                             actual_filter=int(prop.filter), expected_filter=expected,
                             mismatch=int(prop.filter) != expected))
        result = dict(run_status='COMPLETED', body_count=len(names), shape_count=shape_count,
                      native_body_names=names, rows=rows,
                      mismatches=sum(row['mismatch'] for row in rows),
                      source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                      native_asset_urdf_sha256=hashlib.sha256((asset_dir/'inspire_hand_right.urdf').read_bytes()).hexdigest(),
                      simulation_steps=0, model_calls=0, execution_device='cpu',
                      device_reason='SDK asset metadata inspection; no simulation or neural compute',
                      policy_performance_effect_unmeasured=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
        assert result['mismatches'] == 0, 'collision filters disagree with actual SDK shape ownership'
    finally:
        gym.destroy_sim(sim)


if __name__ == '__main__':
    main()
