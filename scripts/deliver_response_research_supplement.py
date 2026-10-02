"""Immutable supplemental delivery, preserving old v11 and materializing new proof."""
import argparse,json,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_contact_response_probe import sha
BASE=ROOT/'src/task/CmResidual/research/contact_response/output'
RUNS=['P-20261002-delayed-request-response-r1','P-20261002-state-response-field-r1','P-20261002-state-response-mask-diagnosis-r1','P-20261002-state-response-field-r2','P-20261002-state-response-field-audit-r1','P-20261002-upstream-design-input-r1','P-20261002-rotation-retention-feasibility-r1']

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve()
    assert out.parent==Path('/home2/wyy/tmp') and out.name.startswith('ref2dex-contact-response-20261002-supplement-') and not out.exists()
    prior=Path('/home2/wyy/tmp/ref2dex-contact-response-20261002-v11');delivery=prior/'delivery.json';old=json.loads(delivery.read_text());assert old['run_status']=='COMPLETED'
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    for name in RUNS:
        d=BASE/name;f=d/'run_manifest.json' if (d/'run_manifest.json').exists() else d/'results.json';m=json.loads(f.read_text());assert m['run_status'] in ('COMPLETED','FAILED'),name
    begin=time.monotonic();out.mkdir();files={};mapping={};canonical={}
    def copy(source,dest):
        source=Path(source);dest=out/dest;real=source.resolve();h=sha(source);dest.parent.mkdir(parents=True,exist_ok=True)
        if real in canonical:dest.symlink_to(os.path.relpath(canonical[real],dest.parent))
        else:shutil.copyfile(source,dest);canonical[real]=dest
        assert sha(dest)==h;files[str(dest.relative_to(out))]=h;mapping[str(real)]=str(dest.relative_to(out))
    for name in RUNS:
        root=BASE/name
        for directory,dirs,names in os.walk(root,followlinks=True):
            dirs[:]=sorted(d for d in dirs if d not in ('cache','player','__pycache__'))
            for namefile in sorted(names):
                source=Path(directory)/namefile;copy(source,Path('new-runs')/root.name/source.relative_to(root))
    for rel in subprocess.check_output(['git','diff','--name-only','bb2d497..HEAD'],cwd=ROOT,text=True).splitlines():
        source=ROOT/rel
        if source.is_file():copy(source,Path('current-code')/rel)
    for rel in ['docs/STATE.md','docs/MISSION.md','docs/CAMPAIGN.md','docs/research/README.md']:
        target=Path('current-code')/rel
        if str(target) not in files:copy(ROOT/rel,target)
    # Existing original raw547/u00 proof remains in the immutable full delivery.
    inheritance={}
    source_mapping=old['original_source_mapping']
    for original,target in source_mapping.items():
        relative=target if isinstance(target,str) else target['path']
        inherited=prior/relative
        if inherited.is_file():inheritance[original]=dict(package=str(prior),relative_path=relative,sha256=old['files_sha256'][relative])
    subprocess.run(['git','bundle','create',str(out/'code.bundle'),'--all'],cwd=ROOT,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    subprocess.run(['git','bundle','verify',str(out/'code.bundle')],cwd=ROOT,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE);files['code.bundle']=sha(out/'code.bundle')
    copy(delivery,Path('base_delivery_manifest.json'))
    for relative,h in files.items():assert sha(out/relative)==h
    result=dict(run_status='COMPLETED',files_sha256=files,original_source_mapping=mapping,inherited_source_mapping=inheritance,base_delivery=str(prior),base_delivery_manifest_sha256=sha(delivery),code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),all_copies_hash_verified=True,base_delivery_unchanged=True,new_runs=RUNS,external_dependencies_not_redistributed=True,total_unique_bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file() and not p.is_symlink()),wall_seconds=time.monotonic()-begin,journal_ready=False)
    (out/'delivery.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['files_sha256','original_source_mapping','inherited_source_mapping']}),flush=True)

if __name__=='__main__':main()
