from pathlib import Path
import numpy as np
import torch
from consequence_evaluator.contracts import HAND_LINKS
from consequence_evaluator.reference_motion import NATIVE_DOF_NAMES
from consequence_evaluator.reference_tracking import apply_coupling
from consequence_evaluator.reset_kinematics import ResetKinematics
from consequence_evaluator.tau_projection import project_tau


def test_projection_preserves_reachable_palm_and_detects_nonrigid_palm():
    urdf = Path(__file__).resolve().parents[4]/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf'
    fk = ResetKinematics(urdf, NATIVE_DOF_NAMES, HAND_LINKS, 'cpu')
    q = torch.zeros(5, 18); q[:, 0] = torch.arange(5)*.002
    q = apply_coupling(q)
    root = torch.zeros(5, 13); root[:, 6] = 1
    hand = fk.states(q, torch.zeros_like(q), root)[:, :, :3].numpy()
    future = np.stack([hand[1:], hand[1:]])
    future[1, :, 3, 0] += .03
    projection = project_tau(np.repeat(hand[:1], 2, axis=0), future,
        np.repeat(q[:1].numpy(), 2, axis=0), urdf, 'cpu', iterations=2)
    np.testing.assert_allclose(projection['points'][0, 1:], future[0], atol=2e-6)
    error = projection['points'][1, 1:, [0, 1, 3, 5, 7, 9]]-future[1, :, [0, 1, 3, 5, 7, 9]]
    assert np.sqrt(np.mean(error**2)) > .002
