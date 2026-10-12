"""Fixed startup sampling partition, independent of model/data execution."""
import importlib.util
from pathlib import Path

import numpy as np


def test_startup_sampler_uses_exact_declared_partitions():
    path=Path(__file__).resolve().parents[1]/'tools/run/calibrate_wam_robot_startup.py'
    spec=importlib.util.spec_from_file_location('wam_startup_sampler',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ticks=np.repeat(np.arange(519),16)
    rng=np.random.default_rng(294)
    total=0
    for _ in range(300):
        rows=module.sample_training_rows(rng,ticks)
        assert rows.shape==(128,)
        assert (rows>=0).all() and (rows<len(ticks)).all()
        assert (ticks[rows]<16).sum()==32
        assert (ticks[rows]>=16).sum()==96
        total+=len(rows)
    assert total==38400
