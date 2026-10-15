"""Convert only clock-consistent HOI4D rigid sequences into explicit synthetic30Hz."""
import argparse
import json
from pathlib import Path
import sys
import time

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(TASK/'src'))
from pose_mesh_data.resampling import resample_native15
from clean_native_corpus import REPO, load_npz, save_npz, write_json, footprint
import shutil
import subprocess


def prepare(source, output, max_seconds=900):
    source,output = source.resolve(),output.resolve()
    if REPO/'outputs'/TASK.name not in output.parents or output.exists():
        raise ValueError('fresh Task output required')
    original = json.loads((source/'manifest.json').read_text())
    if original.get('schema') != 'ref2dex.native-pose-mesh.v1' or original.get('training_allowed') is not False:
        raise ValueError('known provisional HOI4D source required')
    output.mkdir(parents=True,exist_ok=False)
    (output/'sequences').mkdir()
    write_json(output/'input-manifest.json',original)
    started=time.monotonic()
    rows, rejected = [], []
    progress=dict(status='RUNNING',pid=__import__('os').getpid(),input=str(source),run_id=output.name,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        max_seconds=max_seconds,artifact_budget_bytes=1_000_000_000,training_allowed=False,
        device='CPU: array/SO(3) resampling; no neural computation')
    try:
        for i,row in enumerate(original['sequences']):
            if time.monotonic()-started > max_seconds or shutil.disk_usage(output).free < 20*2**30 or (output/'STOP').exists():
                raise RuntimeError('HOI4D preparation deadline/disk/STOP boundary')
            try:
                values=resample_native15(load_npz(source/row['path']),row)
            except ValueError as error:
                rejected.append(dict(source_sequence=row['source_sequence'],source_path=str(source/row['path']),reason=str(error)))
                continue
            relative='sequences/sequence-%05d.npz'%i
            save_npz(output/relative,values)
            rows.append({**row,'path':relative,'format':'canonical','target_slot':0,
                'profile':'hoi4d-native15-resampled30.v1','original_profile':row['profile'],
                'native_source_path':str((source/row['path']).resolve()),'source_fps':15,'stored_fps':30,
                'resampling':'translation/points linear; object rotation short-arc SLERP midpoint; no extrapolation',
                'observed_frames':300,'interpolated_frames':299})
            if i%64 == 0:
                progress.update(processed=i+1,total=len(original['sequences']),converted=len(rows),rejected=len(rejected),
                    elapsed_s=time.monotonic()-started,checked_at=time.time())
                write_json(output/'progress.json',progress)
                print(json.dumps(progress),flush=True)
                if footprint(output)>progress['artifact_budget_bytes']:
                    raise RuntimeError('HOI4D preparation artifact budget')
        meta=dict(schema='ref2dex.pose-mesh.v1',units='m',fps=30,training_allowed=False,
            source_manifest=str(source/'manifest.json'),source_manifests=[str(output)],
            source_clock='native15 documented consistent subset -> explicit30 resampling; original vs interpolated masks retained',
            objects=[{**r,'path':str((source/r['path']).resolve())} for r in original['objects']],sequences=rows,
            limitations=['Engineering resampling, not extra native observations or training admission.',
                         '69 duration-conflicting and 6 duration-missing source sequences remain isolated.'])
        write_json(output/'rejected-sequences.json',rejected)
        write_json(output/'manifest.pending.json',meta)
        progress.update(status='COMPLETED',converted=len(rows),rejected=len(rejected),elapsed_s=time.monotonic()-started)
        if footprint(output)>progress['artifact_budget_bytes'] or shutil.disk_usage(output).free<20*2**30 or time.monotonic()-started>max_seconds:
            raise RuntimeError('HOI4D final resource boundary')
        write_json(output/'progress.json',progress)
        (output/'manifest.pending.json').replace(output/'manifest.json')
        return progress
    except BaseException as error:
        progress.update(status='FAILED',reason=str(error),elapsed_s=time.monotonic()-started)
        write_json(output/'progress.json',progress)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=int,default=900)
    a=p.parse_args()
    print(json.dumps(prepare(a.input,a.output,a.seconds)),flush=True)
