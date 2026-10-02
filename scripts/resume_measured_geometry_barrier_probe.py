"""Reuse exactly one validated native cohort after terminal contention failure."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    source=(ROOT/'scripts/run_measured_geometry_barrier_probe.py').read_text()
    changes={
        "p.add_argument('--gpu-index',type=int,default=4);":"p.add_argument('--resume-source',type=Path,required=True);p.add_argument('--gpu-index',type=int,default=7);",
        "    verify();gpu=admission(a.gpu_index);out.mkdir();begin=time.monotonic()":'''    previous=a.resume_source.resolve();pm=json.loads((previous/'run_manifest.json').read_text());assert pm['run_status']=='FAILED' and pm['experiment_id']=='P-20261002-measured-geometry-barriers'
    assert [q['name'] for q in pm['phases'] if q['run_status']=='COMPLETED']==['s578','s578_audit']
    assert pm['phases'][-1]['name']=='s579' and 'contention' in pm['phases'][-1]['error'] and not (previous/'s579').exists()
    childpid=pm['phases'][-1]['pid'];assert not Path('/proc') .joinpath(str(childpid)).exists(),'old owned process remains live'
    prior_wall=pm['wall_seconds'];prior_bytes=bytes_in(previous);hashes.update(pm['input_sha256']);hashes[str(previous/'run_manifest.json')]=sha(previous/'run_manifest.json')
    for f in ['scripts/resume_measured_geometry_barrier_probe.py','docs/decisions/D-20261002-geometry-barrier-device-resume.md']:hashes[str(ROOT/f)]=sha(ROOT/f)
    verify();gpu=admission(a.gpu_index);out.mkdir();(out/'s578').symlink_to(previous/'s578',target_is_directory=True);begin=time.monotonic()''',
        "m['wall_seconds']=time.monotonic()-begin;":"m['wall_seconds']=time.monotonic()-begin;m['conservative_cumulative_wall_seconds']=prior_wall+m['wall_seconds'];m['prior_failed_run']=str(previous);m['prior_run_manifest_sha256']=sha(previous/'run_manifest.json');",
        "time.monotonic()-begin>1200 or bytes_in(out)>1<<30":"time.monotonic()-begin+prior_wall>1200 or bytes_in(out)+prior_bytes>1<<30",
        "1200-(time.monotonic()-begin)":"1200-(time.monotonic()-begin+prior_wall)",
        "    save()\n    try:\n        for seed in (578,579,580):":"    m['phases'].append(dict(name='s578_reuse',run_status='COMPLETED',reused_validated_source=str(previous/'s578'),no_native_or_audit_repeat=True));save()\n    try:\n        for seed in (579,580):",
        "bytes=bytes_in(out),inputs_unchanged=True":"bytes=bytes_in(out),conservative_cumulative_bytes=bytes_in(out)+prior_bytes,inputs_unchanged=True",
    }
    for old,new in changes.items():
        if source.count(old)!=1:raise ValueError('frozen parent marker drift '+old)
        source=source.replace(old,new)
    exec(compile(source,str(ROOT/'scripts/run_measured_geometry_barrier_probe.py'),'exec'),{'__name__':'__main__','__file__':str(ROOT/'scripts/run_measured_geometry_barrier_probe.py')})

if __name__=='__main__':main()
