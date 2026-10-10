import importlib.util
from pathlib import Path

import numpy as np


_path = Path(__file__).resolve().parents[1] / "tools/audit/audit_paired_triplet_pilot.py"
_spec = importlib.util.spec_from_file_location("paired_triplet_audit", _path)
_audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_audit)


def _minimal_prefix_data():
    frames, commands, envs = 3, 2, 3
    data = {}
    for name in _audit.STATE_FIELDS:
        if name in {"pair", "done"}:
            data[name] = np.zeros((frames, envs), dtype=bool)
        elif name == "actor_observation":
            data[name] = np.zeros((frames, envs, 4), dtype="float32")
        else:
            data[name] = np.zeros((frames, envs, 2), dtype="float32")
    for name in _audit.COMMAND_FIELDS:
        if name in {"structured_mode", "clipped", "done"}:
            data[name] = np.zeros((commands, envs), dtype=np.int8 if name == "structured_mode" else bool)
        elif name == "native_pd_target":
            data[name] = np.zeros((commands, envs, 6), dtype="float32")
        else:
            data[name] = np.zeros((commands, envs, 18), dtype="float32")
    data["structured_phase"] = np.asarray([0, 1], dtype=np.int8)
    return data


def test_prefix_audit_treats_phase_as_time_global_not_env_indexed():
    data = _minimal_prefix_data()
    state, command = _audit._prefix_equal(data, (0, 1, 2), tick=1)
    assert all(state.values())
    assert all(command.values())
    phase = _audit._phase_contract(
        {"structured_phase": np.r_[np.zeros(120, np.int8),
                                    np.ones(120, np.int8),
                                    np.full(302, 2, np.int8)]},
        {"phase_boundaries": {"approach_end_tick": 120,
                               "contact_end_tick": 240}},
    )
    assert phase["exact"]
