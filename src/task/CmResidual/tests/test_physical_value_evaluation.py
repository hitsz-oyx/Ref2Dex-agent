"""Exercise the fixed scientific gate and reject incomplete/mismatched panels."""
import hashlib
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[4]/"scripts"))
from cm_physical_value_evaluation import ARMS, TRAINING_SEEDS, EPOCHS, EVALUATION_SEEDS, checkpoint, summarize


def matrix(root, dropped=False):
    source = root/"source.pth"
    source.write_bytes(b"source")
    for t in TRAINING_SEEDS:
        for epoch in EPOCHS:
            for seed in EVALUATION_SEEDS:
                for arm in ARMS:
                    ckpt = checkpoint(root, arm, t, epoch, source)
                    if not ckpt.exists():
                        ckpt.parent.mkdir(parents=True, exist_ok=True)
                        ckpt.write_bytes(f"{arm}/{t}/{epoch}".encode())
                    n = 8 if epoch==160 and arm=="cm_value" else 2 if epoch==160 else 0
                    rows = [dict(env_id=i,motion_id=i%3,start_frame=0,steps=500,control_dt=1/30,
                        initial_object_height=.01,stable_success=i<n,drop_after_success=dropped and arm=="cm_value" and i<n) for i in range(96)]
                    directory = root/f"eval_{arm}_t{t}_e{epoch}_s{seed}"
                    directory.mkdir()
                    (directory/"run_manifest.json").write_text(json.dumps(dict(run_status="COMPLETED",input_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest())))
                    (directory/"results.json").write_text(json.dumps(dict(run_status="COMPLETED",mode="evaluate",per_episode=rows,
                        stable_success_count=n,drop_after_success_count=sum(r["drop_after_success"] for r in rows))))
    return source


def test_full_matrix_preserves_fixed_gate_and_curve_denominator(tmp_path):
    result=summarize(tmp_path,matrix(tmp_path))
    assert result["conclusion"]=="PROMISING"
    assert result["differences"]==dict(plain_off=.0625,direct_q=.0625)
    assert result["terminal_counts"]==dict(plain_off=8,direct_q=8,cm_value=32)
    assert result["learning_curve_rates"]["cm_value"]["160"]==32/384


def test_success_does_not_bypass_fixed_drop_risk_gate(tmp_path):
    result=summarize(tmp_path,matrix(tmp_path,dropped=True))
    assert result["conclusion"]=="UNPROMISING"
    assert result["conditional_drop_rates"]["cm_value"]==1


def test_balanced_but_unpaired_native_panel_is_rejected(tmp_path):
    source=matrix(tmp_path)
    path=tmp_path/"eval_cm_value_t286_e160_s288/results.json"
    result=json.loads(path.read_text())
    rows=result["per_episode"]
    rows[0]["motion_id"],rows[1]["motion_id"]=rows[1]["motion_id"],rows[0]["motion_id"]
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError,match="pairing mismatch"):
        summarize(tmp_path,source)


def test_missing_native_point_is_not_a_negative_result(tmp_path):
    source=matrix(tmp_path)
    (tmp_path/"eval_cm_value_t287_e160_s289/results.json").unlink()
    with pytest.raises(FileNotFoundError):
        summarize(tmp_path,source)
