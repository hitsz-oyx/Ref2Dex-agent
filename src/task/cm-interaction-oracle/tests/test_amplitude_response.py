import importlib.util
from pathlib import Path
import torch

SPEC = importlib.util.spec_from_file_location("amplitude_contract", Path(__file__).resolve().parents[1] / "src/intervention.py")
contract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(contract)


def test_joint_amplitude_draw_and_scaled_live_clipping():
    arms, alpha = contract.decode_amplitude_assignment(torch.arange(21), [1, 2, 4])
    assert arms.tolist() == list(range(7))*3
    assert alpha.tolist() == [1.]*7+[2.]*7+[4.]*7
    base = torch.zeros(21, 18)
    result = contract.apply_feedback_residual(base, arms, torch.zeros(21), torch.full((21,), 8), torch.zeros(21, dtype=torch.bool), contract.residuals(), alpha)
    assert torch.allclose(result, contract.residuals()[arms]*alpha[:, None])
    base[-2, 6] = .9
    result = contract.apply_feedback_residual(base, arms, torch.zeros(21), torch.full((21,), 8), torch.zeros(21, dtype=torch.bool), contract.residuals(), alpha)
    assert result[-2, 6] == 1


def test_force_direction_is_preserved_when_norm_is_equal():
    force = torch.tensor([[0., 0., 5.], [4., 0., 3.]])
    normal = torch.tensor([[0., 0., 1.]]).expand_as(force)
    fn, ft = contract.surface_force_projection(force, normal)
    assert fn.tolist() == [5., 3.] and ft.tolist() == [0., 4.]
    assert torch.equal(force.norm(dim=-1), torch.tensor([5., 5.]))


def load_probe():
    import sys
    path = Path(__file__).resolve().parents[1] / 'tools/audit'
    sys.path.insert(0,str(path))
    spec = importlib.util.spec_from_file_location('amplitude_probe',path/'probe_amplitude_authority.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_threshold_gate_allows_nonlinearity_but_requires_late_and_repeated_effect():
    import numpy as np
    probe = load_probe()
    b=np.zeros((18,52)); t=np.ones(52)
    b[0,0]=-.01; b[6,0]=.08; b[12,0]=-.20; b[12,1]=-.10
    t[:2]=.05
    c=probe.authority_candidates(b,t,[b,b])
    assert len(c)==1 and c[0]['alpha']==4 and c[0]['orientation']==-1
    weak=b.copy(); weak[12,0]=-.01
    assert not probe.authority_candidates(b,t,[b,weak])
    no_late=b.copy(); no_late[12,1]=0
    assert not probe.authority_candidates(no_late,t,[b,b])
    t[:2]=1; t[19:]=.001
    assert not probe.authority_candidates(b,t,[b,b])  # force-only cannot rescue


def test_distance_alternative_needs_same_arm_late_task_alignment():
    import numpy as np
    probe=load_probe(); b=np.zeros((18,52)); t=np.ones(52)
    b[6,13]=.004; b[6,2]=.15; t[[13,2]]=.025
    c=probe.authority_candidates(b,t,[b,b])
    assert len(c)==1 and c[0]['gate']=='distance' and c[0]['alpha']==2
    b[6,2]=-.15
    assert not probe.authority_candidates(b,t,[b,b])


def test_useful_direction_requires_both_task_gains_and_interaction():
    import numpy as np
    probe=load_probe(); b=np.zeros((6,52)); t=np.ones(52)
    b[0,1]=.15; b[0,2]=-.15; t[1:3]=.05
    families={k:dict(max_tail=1.) for k in probe.FAMILIES}
    assert not probe.useful_candidates(b,t,[b,b],families)
    families['signed_force']['max_tail']=.05
    assert len(probe.useful_candidates(b,t,[b,b],families))==1
    weak=b.copy(); weak[0,2]=.01
    assert not probe.useful_candidates(b,t,[b,weak],families)
    b[0,2]=.15
    assert not probe.useful_candidates(b,t,[b,b],families)
