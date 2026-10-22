"""Execute regressions against deliberately broken in-memory controllers."""
import importlib.util,json,sys,dataclasses
import numpy as np
from pathlib import Path
from unittest.mock import patch
TASK=Path(__file__).resolve().parents[2];sys.path.insert(0,str(TASK/'src'))
from wam_rl_policy_v3.source_ceiling import SourceCeilingConsumer,MarginAdmission,SourceReferenceClock,SourceGraspReference,SourcePDGuard
from types import SimpleNamespace
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
    def wall_clock(self,*args,**kwargs):
        self.cursor+=1
        return dict(advanced=True)
    def cumulative_probe(self,**args):
        self.acquire_anchor=None
        return original(self,**args)
    key_intent=SourceGraspReference.intent
    def omit_finger_closure(self,*args):
        intent=key_intent(self,*args);intent.closure_q=intent.preshape_q.copy()
        return intent
    key_init=SourceGraspReference.__init__
    def measured_state_as_target(self,reference,points,offset,scale):
        key_init(self,reference,points,offset,scale)
        self.closure=reference['q'][self.grasp_frame,0,[6,8,10,12,14,15]].copy()
    def next_command_as_target(self,reference,points,offset,scale):
        key_init(self,reference,points,offset,scale)
        frame=self.grasp_frame
        self.closure=controller_module.pd_targets(reference['action'][frame],reference['q'][frame,0],offset,scale)[[6,8,10,12,14,15]]
    def measured_slew_anchor(self,**args):
        self.finger_target=np.asarray(args['q'])[[6,8,10,12,14,15]].copy()
        return original(self,**args)
    def chasing_close_goal(self,**args):
        self.closing_goal=args['object_pose']@args['intent'].grasp_pose_object
        return original(self,**args)
    def drop_motion(self,goal,**kwargs):return key_intent(self,goal)
    def drop_wrist_preload(self,goal,**kwargs):
        self.pd_grasp=self.grasp.copy()
        return key_intent(self,goal,**kwargs)
    def tick_motion(self,goal,**kwargs):
        self.motion_cursor=min(self.motion_cursor+1,self.grasp_frame)
        return key_intent(self,goal,**kwargs)
    def freeze_probe_closure(self,**args):
        command,info=original(self,**args)
        if getattr(args['feedback'],'grasp_candidate',False):
            command[6:]=2*(args['q'][[6,8,10,12,14,15]]+.0015-self.offset[[6,8,10,12,14,15]])/self.scale[[6,8,10,12,14,15]]-1
        return command,info
    def target_margin_as_pressure(self,q,dq):
        projected=np.asarray(q)+np.asarray(dq)/30
        return bool((self.mask&((projected<self.lower)|(projected>self.upper))).any())
    def anchor_wrist_slew_to_measurement(self,**args):
        self.wrist_command=np.asarray(args['palm']).copy()
        return original(self,**args)
    guard_step=SourcePDGuard.step;guard_init=SourcePDGuard.__init__
    def change_safe_guard(self,**args):
        command,info=guard_step(self,**args);command[0]+=.01
        return command,info
    def remove_guard_margin(self,*args,**kwargs):
        guard_init(self,*args,**kwargs);self.inner_admission=self.outer_admission
    def change_guard_postservo(self,**args):
        command,info=guard_step(self,**args)
        if self.phase in ('lift','hold'):command[0]+=.01
        return command,info
    def freeze_on_unused_closure(self,**args):
        command,info=guard_step(self,**args)
        info['source_ceiling']['finger_bandwidth_limited']|=info['source_ceiling'].get('closure_handover_limited',False)
        return command,info
    def jump_guard_handover(self,**args):
        command,info=guard_step(self,**args)
        if self.phase in ('lift','hold'):
            command[6:]=2*(args['intent'].closure_q-self.offset[[6,8,10,12,14,15]])/self.scale[[6,8,10,12,14,15]]-1
        return command,info
    cases=[('ignore-measured-pressure',patch.object(SourceCeilingConsumer,'force_limit_n',float('inf')),tests.test_measured_pressure_unloads_at_actual_approach_close_step),
        ('unlimited-finger-target-jump',patch.object(SourceCeilingConsumer,'finger_rate_radps',100.),tests.test_approach_finger_target_does_not_jump_to_reference),
        ('omit-target-margin',patch.object(MarginAdmission,'__init__',JointAdmission.__init__),tests.test_nominally_legal_near_limit_target_is_rejected_before_candidate_ranking),
        ('change-held-object-servo',patch.object(SourceCeilingConsumer,'step',altered_held),tests.test_postgrasp_command_is_identical_to_existing_controller),
        ('omit-closure-handover-slew',patch.object(controller_module,'replace',omit_closure_limit),tests.test_close_to_lift_handover_cannot_jump_to_future_closure_target),
        ('log-unexecuted-target',patch.object(SourceCeilingConsumer,'step',incorrect_target_log),tests.test_executed_target_telemetry_matches_final_admitted_command),
        ('retain-recording-world-source',patch.object(tests,'object_anchored_source_target',lambda target,*args:target),tests.test_source_approach_follows_live_object_translation_not_recording_world),
        ('outrun-measured-source-phase',patch.object(SourceReferenceClock,'advance',wall_clock),tests.test_source_cursor_waits_for_measured_wrist_and_is_object_frame_invariant),
        ('remove-acquisition-probe',patch.object(SourceCeilingConsumer,'acquisition_distance_m',0.),tests.test_force_bearing_candidate_gets_bounded_probe_before_motion_coupling),
        ('make-probe-cumulative',patch.object(SourceCeilingConsumer,'step',cumulative_probe),tests.test_acquisition_probe_is_not_cumulative_and_requires_valid_contact),
        ('never-close-at-measured-wrist-pose',patch.object(SourceCeilingConsumer,'wrist_angle_tolerance_rad',-1.),tests.test_measured_wrist_arrival_starts_closure_with_open_fingertips),
        ('omit-source-finger-closure-intent',patch.object(SourceGraspReference,'intent',omit_finger_closure),tests.test_source_keygrasp_preserves_preshape_closure_and_live_SE3_without_clock),
        ('count-table-force-as-object-contact',patch.object(controller_module,'InteractionConfig',lambda:SimpleNamespace(near_surface_m=float('inf'))),tests.test_source_contact_force_without_object_proximity_is_not_a_contact_label),
        ('remove-force-preload',patch.object(SourceGraspReference,'__init__',measured_state_as_target),tests.test_force_bearing_closure_uses_preceding_pd_target_not_deflected_joint_state),
        ('off-by-one-force-command',patch.object(SourceGraspReference,'__init__',next_command_as_target),tests.test_force_bearing_closure_uses_preceding_pd_target_not_deflected_joint_state),
        ('slow-free-approach',patch.object(SourceCeilingConsumer,'free_linear_rate_mps',.3),tests.test_free_approach_is_fast_and_near_object_command_is_slow),
        ('swap-force-and-native-tip-slots',patch.object(SourceCeilingConsumer,'force_to_native_tip',(0,1,2,3,4)),tests.test_same_finger_contact_authorizes_closure_before_tight_wrist_goal),
        ('anchor-slew-to-measured-q',patch.object(SourceCeilingConsumer,'step',measured_slew_anchor),tests.test_command_slew_accumulates_preload_when_measured_joint_is_blocked),
        ('unload-at-command-margin',patch.object(SourceCeilingConsumer,'measured_warning_margin',-.04),tests.test_target_margin_does_not_trigger_measured_unload_on_ordinary_deflection),
        ('miss-intermediate-link-contact',patch.object(SourceCeilingConsumer,'contact_geometry','tip'),tests.test_intermediate_link_contact_can_start_closure_before_tip_contact),
        ('use-initial-touch-instead-of-completed-grasp',patch.object(controller_module,'FeedbackObserver',lambda points:SimpleNamespace(update=lambda *args,**kwargs:SimpleNamespace(grasp_established=False))),tests.test_source_key_prefers_completed_coupled_lift_over_initial_touch),
        ('chase-live-object-during-closing',patch.object(SourceCeilingConsumer,'step',chasing_close_goal),tests.test_closing_reference_does_not_chase_object_after_real_contact),
        ('keep-stale-goal-after-contact-loss',patch.object(SourceCeilingConsumer,'contact_loss_dwell_s',float('inf')),tests.test_contact_loss_releases_captured_goal_and_returns_to_approach),
        ('discard-source-PD-motion',patch.object(SourceGraspReference,'intent',drop_motion),tests.test_source_approach_preserves_recorded_pd_motion_and_uses_measured_progress),
        ('discard-source-wrist-PD-preload',patch.object(SourceGraspReference,'intent',drop_wrist_preload),tests.test_source_key_wrist_preserves_predecessor_pd_preload_as_well_as_fingers),
        ('drive-source-path-by-wall-ticks',patch.object(SourceGraspReference,'intent',tick_motion),tests.test_source_approach_preserves_recorded_pd_motion_and_uses_measured_progress),
        ('continue-inserting-after-measured-contact',patch.object(SourceCeilingConsumer,'step',chasing_close_goal),tests.test_first_measured_contact_stops_insertion_and_closes_fingers_at_measured_wrist),
        ('freeze-closure-at-provisional-candidate',patch.object(SourceCeilingConsumer,'step',freeze_probe_closure),tests.test_acquisition_probe_does_not_freeze_partly_completed_finger_closure),
        ('use-target-margin-as-measured-collision',patch.object(MarginAdmission,'measured_pressure',target_margin_as_pressure),tests.test_inner_target_margin_is_not_a_measured_collision_or_replanning_gate),
        ('limit-PD-error-instead-of-command-velocity',patch.object(SourceCeilingConsumer,'step',anchor_wrist_slew_to_measurement),tests.test_wrist_command_slew_accumulates_pd_tracking_offset_while_measured_hand_lags),
        ('probe-with-partially-open-fingers',patch.object(SourceCeilingConsumer,'closure_ready_for_probe',lambda *args:True),tests.test_candidate_waits_for_completed_closure_and_motion_history_before_probe),
        ('skip-motion-history-after-closure',patch.object(SourceCeilingConsumer,'probe_closure_dwell_s',0.),tests.test_candidate_waits_for_completed_closure_and_motion_history_before_probe),
        ('rewrite-safe-source-PD-command',patch.object(SourcePDGuard,'step',change_safe_guard),tests.test_guard_preserves_original_safe_source_pregrasp_command),
        ('ignore-source-guard-pressure',patch.object(SourcePDGuard,'force_limit_n',float('inf')),tests.test_guard_unloads_measured_pressure_without_changing_postservo),
        ('change-source-guard-postservo',patch.object(SourcePDGuard,'step',change_guard_postservo),tests.test_guard_unloads_measured_pressure_without_changing_postservo),
        ('remove-source-guard-target-margin',patch.object(SourcePDGuard,'__init__',remove_guard_margin),tests.test_guard_projects_source_targets_before_execution_and_uses_physical_warning),
        ('omit-root-command-and-compliance-bounds',patch.object(SourcePDGuard,'limit_wrist_target',lambda self,target,*args:target),tests.test_guard_bounds_command_bandwidth_and_blocked_tracking_preload),
        ('omit-blocked-root-preload-bound',patch.object(SourcePDGuard,'far_tracking_offset_m',float('inf')),tests.test_guard_bounds_command_bandwidth_and_blocked_tracking_preload),
        ('omit-native-finger-command-bandwidth',patch.object(SourcePDGuard,'finger_rate_radps',100.),tests.test_guard_bounds_command_bandwidth_and_blocked_tracking_preload),
        ('omit-native-wrist-joint-command-rate',patch.object(SourcePDGuard,'wrist_joint_rate_radps',100.),tests.test_guard_bounds_native_wrist_joint_command_rate_and_tracking_offset),
        ('omit-native-wrist-joint-tracking-bound',patch.object(SourcePDGuard,'far_angular_offset_rad',float('inf')),tests.test_guard_bounds_native_wrist_joint_command_rate_and_tracking_offset),
        ('freeze-clock-on-unused-handover-target',patch.object(SourcePDGuard,'step',freeze_on_unused_closure),tests.test_guard_unused_handover_closure_cannot_block_executed_preshape_clock),
        ('jump-source-guard-handover-to-full-closure',patch.object(SourcePDGuard,'step',jump_guard_handover),tests.test_guard_handover_limits_actual_finger_target_without_blocking_preshape)]
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
