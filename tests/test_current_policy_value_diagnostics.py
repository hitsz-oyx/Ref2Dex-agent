#!/usr/bin/env python3
"""CPU contract tests for the factual current-policy value diagnostic."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.audit_current_policy_value import (
    DiagnosticError,
    complete_mc_targets,
    frozen_critic_lambda_targets,
    evaluate_saved_pv_value,
    load_collections,
    run_diagnostic,
    synthetic_smoke,
    validate_diagnostic_schema,
)


def smoke_data():
    n = 8
    state = torch.zeros(n, 55); next_state = torch.zeros_like(state)
    state[:, 39] = 1; next_state[:, 39] = 1
    return {"state": state, "next_state": next_state,
            "reward": torch.tensor([1., 2., 3., 4., 1., 2., 3., 4.]),
            "done": torch.tensor([False, False, False, True, False, False, False, True]),
            "terminate": torch.tensor([False, False, False, False, False, False, False, True]),
            "timeout": torch.tensor([False, False, False, True, False, False, False, False]),
            "episode_id": torch.tensor([11, 11, 11, 11, 22, 22, 22, 22], dtype=torch.long),
            "step": torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long),
            "motion_id": torch.zeros(n, dtype=torch.long),
            "value_at_state": torch.tensor([10., 20., 30., 40., 10., 20., 30., 40.]),
            "global_tick": torch.tensor([30, 31, 32, 33, 80, 81, 82, 83], dtype=torch.long),
            "checkpoint_seed": torch.full((n,), 286, dtype=torch.long),
            "value_at_next_state": torch.tensor([20., 30., 40., 0., 20., 30., 40., 0.])}


class CurrentPolicyValueTests(unittest.TestCase):
    def test_hand_computable_mc_timeout_and_episode_boundary(self):
        data = smoke_data()
        contract = validate_diagnostic_schema(data)
        groups = [(key, indices) for key, indices in
                  [((0, 11), [0, 1, 2, 3]), ((0, 22), [4, 5, 6, 7])]]
        mc = complete_mc_targets(data, groups)
        self.assertTrue(torch.allclose(mc[:4], torch.tensor([9.801496, 8.8904, 6.96, 4.]), atol=1e-5))
        target, meta = frozen_critic_lambda_targets(data, groups, contract["checkpoint_id"])
        # Row 1 is the final row of rollout 0: it bootstraps V(row 2), but its
        # recursive lambda carry stops at the global horizon boundary.
        self.assertAlmostEqual(float(target[1]), 31.7, places=4)
        # Terminal timeout row has no bootstrap and does not read episode 22.
        self.assertAlmostEqual(float(target[3]), 4.0, places=5)
        self.assertEqual(meta["terminal_rows_masked"], 2)
        self.assertGreaterEqual(meta["rollout_boundary_rows"], 3)

    def test_synthetic_smoke_is_engineering_only(self):
        result = synthetic_smoke()
        self.assertEqual(result["status"], "COMPLETED")
        self.assertIn("synthetic engineering smoke only", result["scope"])

    def test_missing_nonterminal_value_is_explicit(self):
        data = smoke_data()
        data["value_at_state"][2] = float("nan")
        with self.assertRaises(DiagnosticError):
            run_diagnostic(data)

    def test_saved_v_checkpoint_inference_path_is_separate(self):
        data = smoke_data()
        data["context"] = torch.zeros(8, 435)
        data["previous_action"] = torch.zeros(8, 18)
        root = Path("/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/src/task/CmResidual/research/physical_value/output/P-20260930-cm-physical-value/r7")
        online = next((root / "train_cm_value_s286").glob("**/GRAB_00000420.pth"))
        prediction, metadata = evaluate_saved_pv_value(data, root / "models/tier_1000000.pt", {286: online})
        self.assertEqual(tuple(prediction.shape), (8,))
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertEqual(metadata["checkpoints"]["286"]["optimizer_step"], [7680.0])

    def test_episodes_loader_retains_collector_fields(self):
        n = 4
        state = torch.zeros(n, 55); next_state = torch.zeros_like(state)
        state[:, 39] = 1; next_state[:, 39] = 1
        payload = {
            "schema": "ref2dex.physical_value.v1", "gamma": .99, "control_dt": 1 / 30,
            "state": state, "next_state": next_state,
            "context": torch.zeros(n, 435), "next_context": torch.zeros(n, 435),
            "action": torch.zeros(n, 18), "previous_action": torch.zeros(n, 18),
            "reward": torch.ones(n), "reward_components": torch.cat((torch.ones(n, 1), torch.zeros(n, 4)), 1),
            "done": torch.tensor([False, False, False, True]),
            "terminate": torch.tensor([False, False, False, True]),
            "timeout": torch.zeros(n, dtype=torch.bool),
            "episode_id": torch.zeros(n, dtype=torch.long), "env_id": torch.zeros(n, dtype=torch.long),
            "step": torch.arange(n, dtype=torch.long), "progress": torch.arange(n, dtype=torch.long),
            "start_frame": torch.zeros(n, dtype=torch.long), "motion_id": torch.zeros(n, dtype=torch.long),
            "noise_std": torch.zeros(n), "value_at_state": torch.arange(n, dtype=torch.float32),
            "global_tick": torch.arange(n, dtype=torch.long),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); shard = root / "transitions_000.pt"
            torch.save(payload, shard)
            import hashlib
            digest = hashlib.sha256(shard.read_bytes()).hexdigest()
            (root / "results.json").write_text(json.dumps({"run_status": "COMPLETED", "mode": "collect",
                "shards": [{"path": str(shard), "sha256": digest, "rows": n}]}))
            loaded = load_collections([root])
            self.assertIn("value_at_state", loaded)
            self.assertIn("global_tick", loaded)
            self.assertEqual(loaded["episode_id"].dtype, torch.int64)


if __name__ == "__main__":
    unittest.main()
