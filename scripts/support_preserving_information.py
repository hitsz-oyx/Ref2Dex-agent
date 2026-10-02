"""Prespecified information gates; utility requires a later independent source."""


def fixed_gates(reports, old, adequate):
    cm = reports['cm']['new_reused_held']
    gates = dict(supervision=bool(adequate))
    for subset in ('new_reused_held', 'new_generated_distribution'):
        gates[subset+'_old_height_improve5pct'] = reports['cm'][subset]['height_mae_mm'] <= .95*old[subset]['height_mae_mm']
    for control in ('state_only', 'shuffled'):
        for metric in ('joint_presence_brier', 'contact_loss_brier'):
            gates[control+'_'+metric+'_improve5pct'] = cm[metric] <= .95*reports[control]['new_reused_held'][metric]
    for control, other in [('frozen_old_cm', old['new_reused_held'])] + [(k, reports[k]['new_reused_held']) for k in ('state_only', 'shuffled')]:
        for metric in ('joint_support_brier', 'lift_brier', 'loss_brier', 'object_dv_rmse_mps', 'relative_position_rmse_mm'):
            gates[control+'_'+metric+'_nonworse10pct'] = cm[metric] <= 1.10*other[metric]
    for metric in ('height_mae_mm', 'joint_support_brier'):
        gates['old_replay_'+metric+'_nonworse10pct'] = reports['cm']['old_reused_held'][metric] <= 1.10*old['old_reused_held'][metric]
    gates['coherence'] = all(reports['cm'][subset][metric] <= 1e-6 for subset in
        ('new_cal', 'new_reused_held', 'old_reused_held', 'new_generated_distribution') for metric in
        ('subset_max_excess', 'support_subset_max_excess', 'contact_loss_necessary_max_excess'))
    gates['passed'] = all(gates.values())
    return gates
