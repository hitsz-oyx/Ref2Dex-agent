import importlib.util,json
from pathlib import Path
import numpy as np

def test_first_completed_A3_is_retained_even_when_it_failed_and_later_duplicate_succeeded(tmp_path):
    path=Path(__file__).resolve().parents[1]/'tools/run/prepare_v13_carry.py';spec=importlib.util.spec_from_file_location('v13carry',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    sources=[]
    for i in range(2):
        root=tmp_path/str(i);worker=root/'comparators/g4';worker.mkdir(parents=True);(worker/'manifest.json').write_text(json.dumps(dict(worker=0)))
        row=dict(id='p000-A3',trial='p000',arm='A3',seed=100,model_seed=180,sequence='source',controls=10 if i==0 else 600,final_success=bool(i),physical_abort='physical_joint_limit' if i==0 else None)
        folder=worker/row['id'];folder.mkdir();(folder/'result.json').write_text(json.dumps(row));sources.append(root)
    folder=sources[1]/'comparators/g4/p000-A4';folder.mkdir();(folder/'result.json').write_text(json.dumps(dict(arm='A4',controls=100)))
    folder=sources[1]/'comparators/g4/p001-A4';folder.mkdir();np.savez(folder/'trajectory.npz',action=np.zeros((3,12)))
    index,ledger=m.build_index(sources,dict(p000=dict(native_seed=100,model_seed=180,sequence='source',worker=0)))
    assert index['n']==1 and index['records'][0]['row']['final_success'] is False
    assert index['carried_formal_controls']==10
    assert ledger['old_total_acknowledged']==713 and ledger['acknowledged_controls']==703 and ledger['physical_controls_upper']==704
