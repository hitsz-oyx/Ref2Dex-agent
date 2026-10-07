"""Stage already recovered native references without converting or relinking runtime data."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(TASK/'src'))
from consequence_evaluator.contracts import is_within

SEQUENCES = ('s3_airplane_lift', 's7_airplane_lift_Retake', 's9_airplane_lift',
             's7_apple_lift', 's1_mug_lift', 's1_toothpaste_lift',
             's1_alarmclock_lift', 's1_cubesmall_lift', 's1_cup_lift',
             's1_duck_lift', 's1_phone_lift', 's1_waterbottle_lift')
TRAIN5 = ['s3_airplane_lift', 's1_toothpaste_lift', 's1_mug_lift',
          's1_cubesmall_lift', 's1_waterbottle_lift']
SETS = dict(airplane_base=['s3_airplane_lift'], mixed12=list(SEQUENCES),
            train5=TRAIN5, balanced5=TRAIN5,
            duck=['s1_duck_lift'], cup=['s1_cup_lift'])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(output, assets_input, airplane, cup, batch):
    import torch
    output = Path(output).resolve()
    sources = [Path(path).resolve() for path in (assets_input, airplane, cup, batch)]
    owned = ROOT/'outputs/consequence-evaluator'
    if output.exists() or not is_within(output, owned) or any(not is_within(p, owned) for p in sources):
        raise ValueError('fresh owned output and owned recovered inputs required')
    if shutil.disk_usage(ROOT).free < 20*2**30:
        raise RuntimeError('disk below 20 GiB reserve')
    assets_input, airplane, cup, batch = sources
    files, dependencies = {}, {}
    def recovery(directory, name):
        manifest_path, audit_path = directory/'run_manifest.json', directory/'audit.json'
        manifest = json.loads(manifest_path.read_text())
        audit = json.loads(audit_path.read_text())
        path = Path(manifest['corrected_tensor']).resolve()
        if manifest['status'] != 'COMPLETED' or not is_within(path, owned):
            raise ValueError('incomplete/unowned reference recovery: '+name)
        digest = sha(path)
        if digest != manifest['corrected_sha256'] or digest != audit['corrected_sha256']:
            raise ValueError('recovery hash mismatch: '+name)
        if (audit['left_active'] != 0 or audit['right_wrist_object_error_m'] > 1e-5
                or audit['object_table_relative_error_m'] > 1e-5):
            raise ValueError('reference geometry audit failed: '+name)
        dependencies.update({str(manifest_path): sha(manifest_path), str(audit_path): sha(audit_path)})
        files[name] = path
    recovery(airplane, 's3_airplane_lift')
    recovery(cup, 's1_cup_lift')
    batch_manifest = batch/'run_manifest.json'
    batch_record = json.loads(batch_manifest.read_text())
    if batch_record['status'] != 'COMPLETED' or len(batch_record['recovered']) != 10:
        raise ValueError('ten-reference recovery is incomplete')
    dependencies[str(batch_manifest)] = sha(batch_manifest)
    for item in batch_record['recovered']:
        name = item['sequence']
        recovery(batch/name, name)
        if sha(files[name]) != item['sha256'] or sha(batch/name/'audit.json') != item['audit_sha256']:
            raise ValueError('batch provenance mismatch: '+name)
    if set(files) != set(SEQUENCES):
        raise ValueError('expected exactly the corrected 12 references')
    source_stage = assets_input/'manifest.json'
    old = json.loads(source_stage.read_text())
    dependencies[str(source_stage)] = sha(source_stage)
    if old['status'] != 'STAGED_CPU_CONTRACT_PASS':
        raise ValueError('native assets were not staged successfully')
    for key in ('assets', 'configs'):
        for rel, digest in old[key].items():
            if sha(assets_input/rel) != digest:
                raise ValueError('owned native dependency drift: '+rel)
    records = []
    for name in SEQUENCES:
        tensor = torch.load(files[name], map_location='cpu', weights_only=True)
        if (tensor.ndim != 2 or tensor.shape[1] != 598 or not torch.isfinite(tensor).all()
                or (tensor[:, 206:222] > .5).any()):
            raise ValueError('native motion tensor contract failed: '+name)
        records.append(dict(sequence=name, path=str(files[name]), sha256=sha(files[name]), frames=len(tensor)))
    output.mkdir(parents=True)
    (output/'assets').symlink_to(assets_input/'assets', target_is_directory=True)
    shutil.copytree(assets_input/'cfg', output/'cfg')
    (output/'specs').mkdir()
    for role, names in SETS.items():
        folder = output/'motions'/role
        folder.mkdir(parents=True)
        for name in names:
            (folder/name).symlink_to(files[name].parent, target_is_directory=True)
        spec = dict(input_classification='filtered_geometric_dexplore',
                    description='corrected native imitation references; self-trained expert transfer',
                    motions=[str(files[name].parent) for name in names])
        (output/'specs'/(role+'.json')).write_text(json.dumps(spec, indent=2)+'\n')
    manifest = dict(status='STAGED_CPU_CONTRACT_PASS', task='consequence-evaluator',
                    work_version='corrected-native-expert-reference-staging',
                    raw_motion_role='imitation references; not evaluator examples or grasp evidence',
                    cpu_reason='file/tensor contract inspection only; no model fit',
                    motion_inputs=records, recovery_dependencies=dependencies,
                    assets=old['assets'], configs={str(p.relative_to(output)):sha(p)
                                                   for p in (output/'cfg').rglob('*') if p.is_file()},
                    source_mode='READ_ONLY', source_assets=str(assets_input/'assets'),
                    maximum_source_frames=max(r['frames'] for r in records),
                    runtime_data_links_modified=False, runtime_gpu_smoke='PENDING')
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    base = ROOT/'outputs/consequence-evaluator'
    parser.add_argument('--assets-input', type=Path, default=base/'baseline-inputs-20261007-r2')
    parser.add_argument('--airplane-recovery', type=Path, default=base/'s3-corrected-rebuild-20261007-r1')
    parser.add_argument('--cup-recovery', type=Path, default=base/'cup-corrected-rebuild-20261007-r1')
    parser.add_argument('--batch-recovery', type=Path, default=base/'expert-reference-recovery-20261007-r1')
    args = parser.parse_args()
    result = prepare(args.output, args.assets_input, args.airplane_recovery, args.cup_recovery, args.batch_recovery)
    print(json.dumps(dict(status=result['status'], references=len(result['motion_inputs']),
                         maximum_source_frames=result['maximum_source_frames']), indent=2))


if __name__ == '__main__':
    main()
