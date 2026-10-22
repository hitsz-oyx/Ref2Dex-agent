"""Execute regressions against four deliberately broken in-memory controllers."""
import importlib.util,json,sys,dataclasses
from pathlib import Path
from unittest.mock import patch
TASK=Path(__file__).resolve().parents[2];sys.path.insert(0,str(TASK/'src'))
from wam_rl_policy_v3.source_ceiling import SourceCeilingConsumer,MarginAdmission
from wam_rl_policy_v3.grasp_geometry import JointAdmission
import wam_rl_policy_v3.source_ceiling as controller_module


def main():
    spec=importlib.util.spec_from_file_location('source_ceiling_regressions',TASK/'tests/test_source_ceiling.py')
    tests=importlib.util.module_from_spec(spec);spec.loader.exec_module(tests)
    original=SourceCeilingConsumer.step
    def altered_held(self,**args):
        command,info=original(self,**args)
        if self.phase in ('lift','hold'):command=command.copy();command[0]+=.01
        return command,info
    def omit_closure_limit(value,**changes):
        changes.pop('closure_q',None)
        return dataclasses.replace(value,**changes)
    def incorrect_target_log(self,**args):
        command,info=original(self,**args)
        info['wrist_target'][0][3]+=.01
        return command,info
    cases=[('ignore-measured-pressure',patch.object(SourceCeilingConsumer,'force_limit_n',float('inf')),tests.test_measured_pressure_unloads_at_actual_approach_close_step),
        ('unlimited-finger-target-jump',patch.object(SourceCeilingConsumer,'finger_rate_radps',100.),tests.test_approach_finger_target_does_not_jump_to_reference),
        ('omit-target-margin',patch.object(MarginAdmission,'__init__',JointAdmission.__init__),tests.test_nominally_legal_near_limit_target_is_rejected_before_candidate_ranking),
        ('change-held-object-servo',patch.object(SourceCeilingConsumer,'step',altered_held),tests.test_postgrasp_command_is_identical_to_existing_controller),
        ('omit-closure-handover-slew',patch.object(controller_module,'replace',omit_closure_limit),tests.test_close_to_lift_handover_cannot_jump_to_future_closure_target),
        ('log-unexecuted-target',patch.object(SourceCeilingConsumer,'step',incorrect_target_log),tests.test_executed_target_telemetry_matches_final_admitted_command),
        ('retain-recording-world-source',patch.object(tests,'object_anchored_source_target',lambda target,*args:target),tests.test_source_approach_follows_live_object_translation_not_recording_world)]
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
