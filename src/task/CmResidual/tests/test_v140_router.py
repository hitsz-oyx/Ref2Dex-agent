"""Small deterministic checks for the V1.40 preregistered route objective."""
import unittest

from src.task.CmResidual.tools.fit_s3_episode_router import _fit


class RouterFitTest(unittest.TestCase):
    def test_penalty_and_contiguous_segments(self):
        records = []
        for env_id in range(4):
            records.append((81, env_id, 0, (False, False, False, True)))
            records.append((81, env_id + 4, 1, (True, False, False, False)))
        _, segments, route, raw, achieved = _fit(records)
        self.assertEqual([segment["expert"] for segment in segments],
                         ["back260", "source"])
        self.assertEqual(route, {"0": "back260", "1": "source"})
        self.assertEqual(raw, 8)
        self.assertEqual(achieved[81], 8)

    def test_tie_prefers_one_segment_then_expert_order(self):
        records = [(81, env_id, env_id, (True, True, True, True))
                   for env_id in range(4)]
        _, segments, route, raw, _ = _fit(records)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["expert"], "source")
        self.assertEqual(raw, 4)
        self.assertEqual(set(route.values()), {"source"})


if __name__ == "__main__":
    unittest.main()
