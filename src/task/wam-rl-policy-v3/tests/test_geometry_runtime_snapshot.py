"""Test the isolated package actually loaded by the geometry worker, not live imports."""
import importlib.util,os,shutil,subprocess,sys
from pathlib import Path


def test_worker_snapshot_imports_source_pd_and_detects_omitted_dependency(tmp_path):
    task=Path(__file__).resolve().parents[1];spec=importlib.util.spec_from_file_location('freeze_geometry_contract',task/'tools/audit/freeze_grasp_geometry.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    frozen=tmp_path/'runtime-code';package=frozen/'src/task/wam-rl-policy-v3/src/wam_rl_policy_v3';package.mkdir(parents=True)
    for name in ('__init__.py','inspire_contract.py'):shutil.copyfile(task/'src/wam_rl_policy_v3'/name,package/name)
    module.copy_geometry_runtime(frozen)
    command=[sys.executable,'-c','import wam_rl_policy_v3.source_pd as m; print(m.__file__)']
    env=dict(os.environ,PYTHONPATH=str(package.parent),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='')
    result=subprocess.run(command,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    assert result.returncode==0,result.stdout
    assert str(package/'source_pd.py') in result.stdout
    # Deliberately restore the actual omitted-file defect: no live-tree fallback.
    (package/'source_pd.py').unlink()
    bad=subprocess.run(command,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    assert bad.returncode!=0 and 'ModuleNotFoundError' in bad.stdout
