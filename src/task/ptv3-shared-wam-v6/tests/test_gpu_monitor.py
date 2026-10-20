"""Supervisor must report unavailable telemetry without an integer parse crash."""
import importlib.util
from pathlib import Path
from unittest.mock import patch
import pytest

path=Path(__file__).resolve().parents[1]/'tools/run/launch_pretrain.py'
spec=importlib.util.spec_from_file_location('wam_v6_launch',path)
launch=importlib.util.module_from_spec(spec);spec.loader.exec_module(launch)


def test_reset_gpu_is_explicit_and_telemetry_parse_does_not_crash():
    raw='0, 100, 25000, 48\n3, [N/A], 27993, [GPU requires reset]\n'
    with patch.object(launch.subprocess,'check_output',return_value=raw):
        rows=launch.gpu_rows()
    assert rows[0]['utilization']==100
    assert rows[3]['utilization'] is None
    assert rows[3]['requires_reset'] is True
    assert rows[3]['memory_mib']==27993
    with pytest.raises(ValueError,match='healthy'):
        launch.require_released(rows,[3])


def test_missing_utilization_alone_is_not_hardware_reset():
    with patch.object(launch.subprocess,'check_output',return_value='0, [N/A], 0, 45\n'):
        row=launch.gpu_rows()[0]
    assert row['utilization'] is None
    assert row['requires_reset'] is False
    launch.require_released({0:row},[0])


def test_unknown_memory_never_counts_as_released():
    with patch.object(launch.subprocess,'check_output',return_value='0, 0, [N/A], 45\n'):
        rows=launch.gpu_rows()
    with pytest.raises(ValueError,match='released'):
        launch.require_released(rows,[0])
