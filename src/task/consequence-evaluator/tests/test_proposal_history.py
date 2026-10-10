import numpy as np
from consequence_evaluator.proposal_history import condition


def example():
    rng = np.random.default_rng(44)
    obj = np.broadcast_to(np.eye(4), (4, 4, 4)).copy()
    obj[:, :3, 3] = rng.normal(size=(4, 3))
    return [obj, rng.normal(size=(4, 11, 3)), rng.normal(size=(4, 18)),
            rng.normal(size=(4, 18)), rng.normal(size=(4, 6))]


def test_measured_history_has_no_world_translation_dependence():
    values = example(); original = condition(*values)
    shifted = [x.copy() for x in values]
    shift = np.array([100., -30., 7.])
    shifted[0][:, :3, 3] += shift; shifted[1] += shift
    # Native wrist q translation is deliberately excluded; measured geometry
    # supplies wrist state. Finger q/dq remain measured inputs.
    shifted[2][:, :3] += shift
    translated = condition(*shifted)
    np.testing.assert_allclose(original[0], translated[0], atol=1e-6)
    np.testing.assert_allclose(original[1], translated[1], atol=1e-6)


def test_query_condition_uses_only_measured_prefix():
    values = example()
    packets = [np.concatenate([x, x], axis=0) for x in values]
    original = condition(*(x[:4] for x in packets))
    for x in packets:
        x[4:] = np.nan
    poisoned = condition(*(x[:4] for x in packets))
    np.testing.assert_array_equal(original[0], poisoned[0])
    np.testing.assert_array_equal(original[1], poisoned[1])
