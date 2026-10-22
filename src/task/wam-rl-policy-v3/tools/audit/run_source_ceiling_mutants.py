"""Execute regressions against four deliberately broken in-memory controllers."""
import importlib.util,json,sys
from pathlib import Path
from unittest.mock import patch
TASK=Path(__file__).resolve().parents[2];sys.path.insert(0,str(TASK/'src'))
from wam_rl_policy_v3.source_ceiling import SourceCeilingConsumer,MarginAdmission
from wam_rl_policy_v3.grasp_geometry import JointAdmission


def main():
    spec=importlib.util.spec_from_file_location('source_ceiling_regressions',TASK/'tests/test_source_ceiling.py')
    tests=importlib.util.module_from_spec(spec);spec.loader.exec_module(tests)
    original=SourceCeilingConsumer.step
    def altered_held(self,**args):
        command,info=original(self,**args)
        if self.phase in ('lift','hold'):command=command.copy();command[0]+=.01
        return command,info
    cases=[('ignore-measured-pressure',patch.object(SourceCeilingConsumer,'force_limit_n',float('inf')),tests.test_measured_pressure_unloads_at_actual_approach_close_step),
        ('unlimited-finger-target-jump',patch.object(SourceCeilingConsumer,'finger_rate_radps',100.),tests.test_approach_finger_target_does_not_jump_to_reference),
        ('omit-target-margin',patch.object(MarginAdmission,'__init__',JointAdmission.__init__),tests.test_nominally_legal_near_limit_target_is_rejected_before_candidate_ranking),
        ('change-held-object-servo',patch.object(SourceCeilingConsumer,'step',altered_held),tests.test_postgrasp_command_is_identical_to_existing_controller)]
    result=[]
    for name,mutation,check in cases:
        check() # First prove the unmutated check passes.
        killed=False
        with mutation:
            try:check()
            except AssertionError:killed=True
        result.append(dict(mutant=name,killed=killed))
    print(json.dumps(dict(status='PASS' if all(r['killed'] for r in result) else 'FAIL',killed=sum(r['killed'] for r in result),total=len(result),records=result)))
    return int(not all(r['killed'] for r in result))


if __name__=='__main__':raise SystemExit(main())
