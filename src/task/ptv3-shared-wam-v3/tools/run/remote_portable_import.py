"""Owned remote CPU import supervisor; no GPU work or pre-existing data writes."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scope',type=Path,required=True)
    p.add_argument('--main3',type=Path,required=True)
    p.add_argument('--python',required=True)
    p.add_argument('--vendor',type=Path,required=True)
    a=p.parse_args()
    scope=a.scope.resolve(); stage=scope/'staging'
    def progress(status,**extra):
        value=dict(status=status,pid=os.getpid(),updated_at=time.time(),**extra)
        (scope/'remote-progress.json').write_text(json.dumps(value,indent=2))
        print(json.dumps(value),flush=True)
    try:
        progress('ASSEMBLING')
        descriptor=json.loads((stage/'portable.json').read_text())
        archive=stage/'native-and-recipe.tar'
        if not archive.exists():
            with (stage/'native-and-recipe.tar.part').open('wb') as target:
                for part in sorted(stage.glob('native.part-*')):
                    shutil.copyfileobj(part.open('rb'),target,2**20)
            os.rename(stage/'native-and-recipe.tar.part',archive)
        if archive.stat().st_size!=descriptor['archive_bytes']:
            raise ValueError('assembled archive is incomplete')
        code=scope/'code'
        if not (code/'CODE_VERSION.json').exists():
            with tarfile.open(next(stage.glob('code-minimal-*.tar'))) as tar:
                for item in tar:
                    if item.issym() or item.islnk() or item.name.startswith('/') or '..' in Path(item.name).parts:
                        raise ValueError('unsafe code archive member')
                    tar.extract(item,code)
            (code/'third_party').mkdir(exist_ok=True)
            os.symlink(a.vendor,code/'third_party/PointWorld')
            (code/'CODE_VERSION.json').write_text(json.dumps(dict(git_commit=next(stage.glob('code-minimal-*.tar')).stem.replace('code-minimal-',''),vendor=str(a.vendor)),indent=2))
        run=scope/'data'
        if run.exists() and not (run/'portable-reader-audit.json').exists():
            raise ValueError('previous partial import retained; choose a fresh scope for a new import')
        environment=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='',TMPDIR=str(scope/'runtime'))
        (scope/'runtime').mkdir(exist_ok=True)
        tools=code/'src/task/ptv3-shared-wam-v3/tools'
        commands=[]
        if not run.exists():
            commands.append([a.python,str(tools/'run/import_portable_nine.py'),'--export',str(stage),'--main3',str(a.main3),'--output',str(run)])
        commands.append([a.python,str(tools/'audit/audit_portable_nine.py'),'--import-root',str(run)])
        for command in commands:
            progress('IMPORTING' if 'import_portable_nine.py' in command[1] else 'AUDITING',command=command)
            subprocess.run(command,cwd=code,env=environment,check=True,timeout=5400)
        progress('COMPLETED',data=str(run),audit=str(run/'portable-reader-audit.json'),code=str(code))
    except Exception as error:
        progress('FAILED',error=repr(error))
        raise


if __name__=='__main__':
    main()
