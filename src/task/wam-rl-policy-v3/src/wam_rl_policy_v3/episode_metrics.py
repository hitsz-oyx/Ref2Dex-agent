"""Keep official sticky success separate from final task outcomes."""
def episode_outcome(native_success, longest, streak, workspace_failure, final_near, dropped_after_hold=None):
    ever = bool(longest >= 60)
    final = bool(streak >= 60)
    return dict(native_success=bool(native_success),
                held60_lift10cm=ever,  # historical alias: never reinterpret old reports
                ever_held60=ever, final_held60=final,
                drop_after_hold=bool(ever and not final) if dropped_after_hold is None else bool(dropped_after_hold),
                workspace_failure=bool(workspace_failure),
                final_task_success=bool(final_near and not workspace_failure),
                final_stable_success=bool(final and not workspace_failure),
                termination_reason='wam_workspace' if workspace_failure else 'native_done')
