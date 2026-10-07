"""Stage read-only native motions/assets for fresh self-trained experts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(TASK / 'src'))

import torch
import yaml
from consequence_evaluator.contracts import is_within

SEQUENCES = ('s1_airplane_lift', 's3_airplane_lift', 's7_airplane_lift_Retake',
             's9_airplane_lift', 's7_apple_lift', 's1_mug_lift', 's1_toothpaste_lift',
             's1_alarmclock_lift', 's1_cubesmall_lift', 's1_cup_lift', 's1_duck_lift',
             's1_phone_lift', 's1_waterbottle_lift')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(output, source):
    output, source = Path(output).resolve(), Path(source).resolve()
    if not is_within(output, ROOT/'outputs/consequence-evaluator') or output.exists():
        raise ValueError('fresh task-owned staging directory required')
    if shutil.disk_usage(ROOT).free < 20*2**30:
        raise RuntimeError('free disk below20GiB reserve')
    native = source/'third_party/DExplore/dexplore/data'
    motions = source/'data/processed_data/inspire_geometric_dexplore'
    records = []
    for name in SEQUENCES:
        path = motions/name/'interaction_hand_inspire.pt'
        tensor = torch.load(path, map_location='cpu', weights_only=True)
        if (tensor.ndim != 2 or tensor.shape[1] != 598 or not torch.isfinite(tensor).all()
                or (tensor[:,206:222]>.5).any()):
            raise ValueError('native geometric motion contract/filter failed: '+name)
        records.append(dict(sequence=name, path=str(path), sha256=digest(path), frames=len(tensor)))
    # All writes are local. Copy assets to avoid external VHACD/asset cache writes.
    output.mkdir(parents=True)
    (output/'assets').mkdir()
    for folder in ('inspire_hand_new', 'mjcf'):
        shutil.copytree(native/'assets'/folder, output/'assets'/folder)
    shutil.copytree(native/'cfg', output/'cfg')
    env = yaml.safe_load((output/'cfg/inspire.yaml').read_text())
    env['env'].update(numEnvs=64, objectMotionSampling=True, hardObjectOversampling=False)
    env['env']['asset']['assetRoot'] = 'dexplore/data/assets'
    (output/'cfg/inspire_object_balanced.yaml').write_text(yaml.safe_dump(env, sort_keys=False))
    runtime_data = ROOT/'third_party/DExplore/dexplore/data'
    old_link = None
    if runtime_data.is_symlink() and not runtime_data.exists():
        old_link = os.readlink(runtime_data)
        runtime_data.rename(output/'previous-native-data-link')
    runtime_data.mkdir(parents=True, exist_ok=True)
    for name in ('assets', 'cfg'):
        target = runtime_data/name
        if target.exists() or target.is_symlink():
            raise FileExistsError('refuse to replace existing runtime dependency: '+str(target))
        target.symlink_to(output/name, target_is_directory=True)
    sets = dict(parent_s1=['s1_airplane_lift'], airplane_base=['s3_airplane_lift'],
                mixed12=[name for name in SEQUENCES if name != 's1_airplane_lift'],
                train5=['s3_airplane_lift','s1_toothpaste_lift','s1_mug_lift','s1_cubesmall_lift','s1_waterbottle_lift'],
                balanced5=['s3_airplane_lift','s1_toothpaste_lift','s1_mug_lift','s1_cubesmall_lift','s1_waterbottle_lift'],
                duck=['s1_duck_lift'], cup=['s1_cup_lift'])
    (output/'specs').mkdir()
    for expert, names in sets.items():
        folder = output/'motions'/expert
        folder.mkdir(parents=True)
        for name in names:
            (folder/name).symlink_to(motions/name, target_is_directory=True)
        spec = dict(input_classification='filtered_geometric_dexplore',
                    description='new self-trained baseline; not restored historical checkpoints',
                    motions=[str(motions/name) for name in names])
        (output/'specs'/(expert+'.json')).write_text(json.dumps(spec, indent=2)+'\n')
    manifest = dict(status='STAGED_CPU_CONTRACT_PASS', task='consequence-evaluator',
                    raw_motion_role='geometric imitation references, not robot evaluator examples',
                    cpu_reason='input tensor/asset inspection only; no model computation',
                    source_project=str(source), motion_inputs=records, source_mode='READ_ONLY',
                    previous_broken_native_data_link=old_link,
                    assets={str(p.relative_to(output)):digest(p) for p in (output/'assets').rglob('*') if p.is_file()},
                    configs={str(p.relative_to(output)):digest(p) for p in (output/'cfg').rglob('*') if p.is_file()},
                    maximum_source_frames=max(r['frames'] for r in records),
                    first_training='parent_s1 random initialization; verify full frame0 rollout before s3 transfer',
                    runtime_gpu_smoke='PENDING')
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source-project', type=Path, default=Path('/home2/wyy/oyx_ws/Ref2Dex'))
    a = p.parse_args()
    result = prepare(a.output, a.source_project)
    print(json.dumps({key:result[key] for key in ('status','maximum_source_frames','first_training')}, indent=2))


if __name__ == '__main__':
    main()
