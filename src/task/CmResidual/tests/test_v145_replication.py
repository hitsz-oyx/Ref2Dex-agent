"""Deterministic checks for V1.45's seed-cluster analysis."""

from src.task.CmResidual.tools.analyze_v145_replication import cluster_bootstrap
from src.task.CmResidual.tools.run_v145_replication import command


def test_cluster_bootstrap_constant_seed_effect():
    lower, upper = cluster_bootstrap([0.125] * 5)
    assert lower == upper == 0.125


def test_baseline_and_route_use_same_evaluator_with_distinct_map():
    route, _ = command("route", 99, 0, 5)
    baseline, _ = command("baseline", 99, 0, 6)
    assert route[1] == baseline[1]
    assert route[route.index("--router-map-file") + 1] != baseline[
        baseline.index("--router-map-file") + 1]
    assert route.count("--router-checkpoint") == 3
    assert baseline.count("--router-checkpoint") == 1
