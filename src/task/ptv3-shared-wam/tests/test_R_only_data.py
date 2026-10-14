import sys
from pathlib import Path
import numpy as np
from test_rl_robot import corpus
from ptv3_shared_wam.rl_robot import RLRobotWindows
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/run'))
from r_only_data import pack


def test_packed_arrays_preserve_real_loader_semantics_order_and_repeated_indices(tmp_path):
    dataset=RLRobotWindows(corpus(tmp_path),'train')
    indices=[25,0,25,1];packed=pack(dataset,indices)
    for row,index in enumerate(indices):
        sample=dataset[index]
        for key,source in [('state','native_state'),('tau','native_tau'),('action','native_action')]:
            np.testing.assert_allclose(packed[key][row],sample[source],atol=1e-6)
        assert packed['tick'][row]==dataset.keys[index][0]
