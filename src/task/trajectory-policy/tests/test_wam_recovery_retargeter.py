"""CPU admission regressions for the pinned recovery-coverage probe."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np

TASK = Path(__file__).resolve().parents[1]
ROOT = TASK.parents[2]
spec = importlib.util.spec_from_file_location('recovery_fitter', TASK/'tools/run/fit_wam_recovery_retargeter.py')
fitter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fitter)


class AdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        native = ROOT/'outputs/trajectory-policy/wam-ref2-execution-aggcollect-20261012-r1/native'
        if not (native/'trajectory.npz').exists():
            raise unittest.SkipTest('probe source absent')
        cls.manifest, cls.audit, cls.result = [json.loads((native/k).read_text()) for k in
            ('manifest.json','root_recovery_audit.json','result.json')]
        with np.load(native/'trajectory.npz',allow_pickle=False) as stream:
            cls.data = {k:stream[k].copy() for k in stream.files}
        cls.digest = fitter.sha(native/'trajectory.npz')

    def admission(self, manifest=None, audit=None, data=None):
        return fitter.admit(manifest or self.manifest,audit or self.audit,self.result,
                            data or self.data,self.digest)

    def test_exact_source_admitted_R_only(self):
        admitted = self.admission()
        self.assertTrue(admitted['R_training_allowed'])
        self.assertFalse(admitted['actor_future_hand_training_allowed'])
        self.assertFalse(admitted['mixed_policy_future_hand_training_allowed'])

    def test_split_and_last_causal_control(self):
        train, held = fitter.recovery_split()
        self.assertEqual((len(train),len(held)),(113,24))
        self.assertFalse(set(train)&set(held))
        self.assertEqual(set(train)|set(held),set(range(32,169)))
        self.assertEqual(set(held),set(range(40,48))|set(range(96,112)))
        data = dict(self.data,expert_applied=self.data['expert_applied'].copy())
        data['expert_applied'][191,1] = False
        with self.assertRaises(ValueError):
            self.admission(data=data)

    def test_old_protocol_and_wrong_role_rejected(self):
        for key,value in [('controls',542),('takeover_tick',16),('roles',list(reversed(fitter.ROLES)))]:
            m=copy.deepcopy(self.manifest); m[key]=value
            with self.assertRaises(ValueError):
                self.admission(manifest=m)

    def test_audit_must_bind_data(self):
        audit=dict(self.audit,native_data_sha256='0'*64)
        with self.assertRaises(ValueError):
            self.admission(audit=audit)

    def test_applied_PD_and_action_must_equal_expert(self):
        for key,index in [('pd_targets',(32,3,0)),('applied',(32,3,0)),('expert_PD',(168,1,0))]:
            data=dict(self.data,**{key:self.data[key].copy()})
            data[key][index] += .01
            with self.assertRaises(ValueError):
                self.admission(data=data)

    def test_cannot_enable_actor_future_or_admit_clipped_recovery(self):
        m=dict(self.manifest,actor_future_hand_training_allowed=True)
        with self.assertRaises(ValueError):
            self.admission(manifest=m)
        data=dict(self.data,expert_clipped=self.data['expert_clipped'].copy())
        data['expert_clipped'][32,1]=True
        with self.assertRaises(ValueError):
            self.admission(data=data)


if __name__=='__main__':
    unittest.main()
