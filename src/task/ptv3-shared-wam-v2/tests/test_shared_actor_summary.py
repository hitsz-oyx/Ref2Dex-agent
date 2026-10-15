import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest

path=Path(__file__).resolve().parents[1]/'tools/audit/summarize_shared_actor.py'
spec=importlib.util.spec_from_file_location('shared_summary',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def setup_runs(root):
    for arm in module.ARMS:
        p=root/('shared-actor-'+arm+'-20261015-r1');p.mkdir()
        manifest=dict(status='COMPLETED',config=dict(action_interface=arm,updates=12000),
            provenance={k:'same' for k in ('source','parent_checkpoint','parent_step','world_size','git_commit')})
        (p/'manifest.json').write_text(json.dumps(manifest));np.save(p/'validation.npy',np.arange(3))
        (p/'rollout_admission.json').write_text('{}')
        reduction=1 if arm=='pred_tau' else .7
        metric=lambda mean,p95:dict(mean=mean,p95=p95,count=3)
        physical=dict(PD_xyz_mm=metric(50*reduction,100*reduction),
                      hand_all24_mm=metric(20,40),scene_all24_mm=metric(10,20))
        curve=[dict(step=12000,objective=1.,physical=physical)]
        (p/'curve.json').write_text(json.dumps(curve));(p/'checkpoint_files.json').write_text('["step12000.pt"]')


def test_matched_final_comparison_and_mismatched_panel_rejected(tmp_path):
    setup_runs(tmp_path)
    report=module.summarize(tmp_path)
    assert report['matched_checks']=='PASS' and report['improvements']['shared']['signal']=='PROMISING'
    p=tmp_path/'shared-actor-shared-20261015-r1/validation.npy';np.save(p,[2,1,0])
    with pytest.raises(ValueError,match='panels'):module.summarize(tmp_path)
