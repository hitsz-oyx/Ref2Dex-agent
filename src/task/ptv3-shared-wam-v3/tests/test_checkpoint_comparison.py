from pathlib import Path
from types import SimpleNamespace
import importlib.util

path=Path(__file__).resolve().parents[1]/'tools/audit/compare_main3_checkpoints.py'
spec=importlib.util.spec_from_file_location('checkpoint_comparison',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_native_identity_handles_list_index_and_rollout_batches_and_gap_segments():
    rows=[dict(path='batch-a/episode.npz',source_sequence='episode'),dict(path='batch-b/episode.npz',source_sequence='episode'),
          dict(path='part0.npz',source_sequence='subject/take'),dict(path='part1.npz',source_sequence='subject/take')]
    data=SimpleNamespace(main=[1,2],native=SimpleNamespace(root=Path('/corpus'),index=[(i,3) for i in range(4)],meta={'sequences':rows}))
    assert module.recording_identity(data,0,dict(source_id=0,sample_id=[7,0,3]))=='0:7'
    assert module.recording_identity(data,2,dict(source_id=4))!=module.recording_identity(data,3,dict(source_id=4))
    assert module.recording_identity(data,4,dict(source_id=9))==module.recording_identity(data,5,dict(source_id=9))


def test_paired_errors_weight_valid_points_and_bootstrap_whole_recordings():
    def row(i,capture,n,error):return dict(index=i,capture=capture,metrics={k:dict(sum_mm=error*n,count=n) for k in module.CORE})
    old=[row(0,'take-a',1,10),row(1,'take-a',3,20),row(2,'take-b',2,30)]
    new=[row(0,'take-a',1,5),row(1,'take-a',3,10),row(2,'take-b',2,15)]
    result=module.paired(old,new)['hand_all24']
    assert abs(result['old_mm']-130/6)<1e-10
    assert abs(result['new_mm']-65/6)<1e-10
    assert result['improvement_pct']==50 and result['captures']==2 and result['windows']==3
    assert result['capture_bootstrap_ci95_new_minus_old_mm'][1]<0
