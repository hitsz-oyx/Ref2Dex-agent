"""Materialize the complete probe and paper in a new durable, verified package."""
import argparse,hashlib,json,shutil,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while True:
            block=f.read(8<<20)
            if not block:break
            h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--compile',type=Path,required=True);p.add_argument('--destination',type=Path,required=True);a=p.parse_args()
    run=a.run.resolve();dest=a.destination.resolve();compiled=a.compile.resolve();begin=time.monotonic()
    if dest.exists() or dest.parent!=Path('/home2/wyy/tmp'):raise ValueError('unique own durable package only')
    m=json.loads((run/'run_manifest.json').read_text());cm=json.loads((compiled/'compile_manifest.json').read_text())
    if m['run_status']!='COMPLETED' or cm['run_status']!='COMPLETED':raise ValueError('completed scientific run and native paper only')
    if shutil.disk_usage(dest.parent).free<12<<30:raise ValueError('durable storage margin')
    for path,h in cm['source_sha256'].items():assert sha(path)==h,path
    assert sha(cm['pdf'])==cm['pdf_sha256']
    dest.mkdir();mapping={};hashes={};by_hash={}
    def copy(source,relative):
        source=Path(source);target=dest/relative;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():raise FileExistsError(target)
        shutil.copy2(source,target);digest=sha(target)
        assert sha(source)==digest,source
        mapping[str(source.resolve())]=dict(path=str(target.relative_to(dest)),sha256=digest)
        hashes[str(target.relative_to(dest))]=digest;by_hash.setdefault(digest,str(target.relative_to(dest)))
    for seed in list(range(547,567))+[568,569]:
        d=run/f's{seed}'
        for name in ['initial.pt','trace.pt','physical_metadata.json','results.json','rows.json','panel_audit.json']:
            copy(d/name,Path('full-run')/d.name/name)
        print(json.dumps(dict(materialized_panel=seed)),flush=True)
    for update in range(21):
        d=run/f'u{update:02d}'
        for name in ['policy_heads.pt','results.json']+(['update_packet.pt','gradient_audit.json'] if update else []):
            copy(d/name,Path('full-run')/d.name/name)
    for f in run.iterdir():
        if f.is_file():copy(f,Path('full-run')/f.name)
    copy(m['base_checkpoint'],'full-run/base-policy.pt')
    # Preserve every execution failure, its command/log, and diagnostic proof.
    base=run.parent
    names=['P-20261002-continuous-critic-policy-resume-r1','P-20261002-continuous-critic-policy-resume-r2',
           'P-20261002-continuous-native-memory-diagnosis-r1','P-20261002-continuous-gradient-diagnosis-r1',
           'P-20261002-continuous-gradient-correction-r1','P-20261002-continuous-critic-policy-closeout-r3',
           'P-20261002-continuous-critic-host-stop-r1','P-20261002-continuous-critic-migration-inspection-r1',
           'P-20261002-continuous-critic-audit-resume-inspection-r1','P-20261002-paper-v11-audit-r1']
    for name in names:
        for f in (base/name).iterdir():
            if f.is_file():copy(f,Path('provenance')/name/f.name)
        if name in ['P-20261002-continuous-critic-policy-resume-r1','P-20261002-continuous-native-memory-diagnosis-r1']:
            for f in (base/name/'s550').iterdir():
                if f.is_file():copy(f,Path('provenance')/name/'s550'/f.name)
    original=Path(m['original_source_run'])
    for f in original.iterdir():
        if f.is_file():copy(f,Path('provenance')/original.name/f.name)
    for f in (ROOT/'paper/native-v10').iterdir():
        if f.is_file():copy(f,Path('provenance/native-v10')/f.name)
    # Input snapshots are deduplicated against the full physical/optimizer data.
    for filename,digest in cm['source_sha256'].items():
        real=str(Path(filename).resolve())
        if digest in by_hash:
            mapping[real]=dict(path=by_hash[digest],sha256=digest)
        else:copy(filename,Path('inputs')/(digest+'-'+Path(filename).name))
    for f in compiled.iterdir():
        if f.is_file():copy(f,Path('paper/native-v11')/f.name)
    copy(ROOT/'paper/manuscript-v11.tex','paper/manuscript-v11.tex')
    copy(ROOT/'paper/source_manifest-v11.json','paper/source_manifest-v11.json')
    for folder in ['tables-v11','figures']:
        for f in (ROOT/'paper'/folder).iterdir():
            if f.is_file():copy(f,Path('paper')/folder/f.name)
    copy(ROOT/'docs/STATE.md','docs/STATE.md')
    subprocess.run(['git','bundle','create',str(dest/'code.bundle'),'--all'],cwd=str(ROOT),check=True)
    subprocess.run(['git','bundle','verify',str(dest/'code.bundle')],cwd=str(ROOT),check=True)
    hashes['code.bundle']=sha(dest/'code.bundle')
    readme='''Complete continuous-policy Probe and native manuscript v11.\n\nThis package materializes all 22 native raw panels, all 21 actor/critic/Adam\ncheckpoints, all 20 optimizer packets and audits, preserved failure/diagnostic\nrecords, native paper and exact hashed manuscript inputs. External IsaacGym,\nDExplore dependencies and asset datasets are not redistributed; their original\npaths/hashes remain in the source manifests. The experiment includes an explicit\nGPU migration and witnessed u17 floating-point ReLU branch audit correction.\nAbsolute paths in original manifests are retained for provenance; delivery.json\nmaps original source paths to verified local snapshots. Earlier delivery packages\nare untouched. This is one optimization seed and does not establish journal\nreadiness, generalization or a distinctive method.\n'''
    (dest/'README.txt').write_text(readme);hashes['README.txt']=sha(dest/'README.txt')
    for relative,digest in hashes.items():assert sha(dest/relative)==digest,relative
    record=dict(run_status='COMPLETED',files_sha256=hashes,original_source_mapping=mapping,
                code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(ROOT),text=True).strip(),
                scientific_commit=m['scientific_git_commit'],label=m['label'],native_panels=22,
                actual_optimizer_steps=9120,total_bytes=sum((dest/f).stat().st_size for f in hashes),
                all_copies_hash_verified=True,previous_packages_unchanged=True,
                external_dependencies_not_redistributed=True,journal_ready=False,
                wall_seconds=time.monotonic()-begin)
    (dest/'delivery.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ['files_sha256','original_source_mapping']}),flush=True)

if __name__=='__main__':main()
