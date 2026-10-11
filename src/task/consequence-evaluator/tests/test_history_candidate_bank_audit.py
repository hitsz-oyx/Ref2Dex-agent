import importlib.util
from pathlib import Path

import numpy as np


_spec = importlib.util.spec_from_file_location(
    "history_candidate_bank_audit",
    Path(__file__).resolve().parents[1] / "tools/audit/audit_history_candidate_bank.py",
)
_audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_audit)


def test_history_pairwise_rms_is_symmetric_and_zero_on_diagonal():
    value = np.asarray([[0., 0.], [3., 4.], [0., 4.]], dtype="float32")
    result = _audit.pairwise_rms(value)
    np.testing.assert_allclose(result, result.T)
    np.testing.assert_allclose(np.diag(result), 0.)
    np.testing.assert_allclose(result[0, 1], np.sqrt(12.5))


def test_summary_handles_empty_and_nonempty_values():
    assert _audit.summary([]) == {"count": 0, "mean": None, "max": None}
    assert _audit.summary([1., 3.]) == {"count": 2, "mean": 2., "max": 3.}
