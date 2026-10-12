"""End-to-end robot policy losses and exact paired checkpoint contracts."""
import hashlib
import math

import torch

JOINT_SCHEMA = 'ref2dex.wam-joint-retargeter.v1'
PATCH_SCHEMA = 'ref2dex.wam-policy-chain-patch.v1'
POLICY_MODULES = ('scene', 'goal', 'trajectory', 'hand_query')
TRAIN_MODULES = (*POLICY_MODULES, 'R')


def own_expert_admission(manifest, audit, result, data, trajectory_hash, group, reference):
    """Narrow role3 true-future scope; role2 has CURRENT expert PD only."""
    import numpy as np
    from .pd_retargeter import COUPLING
    expected_sha = '7c586a79eb8c859405cca1be9cda3b7a28505136d4464f2a24a2e8f80bf343a3'
    if (trajectory_hash != expected_sha or group.get('status') != 'COMPLETED'
            or manifest.get('status') != 'COMPLETED' or result.get('status') != 'COMPLETED'
            or manifest.get('controls') != 128 or manifest.get('takeover_tick') != 8
            or group.get('controls') != 128 or group.get('takeover_tick') != 8
            or manifest.get('training_allowed') is not True
            or manifest.get('expert_PD_training_allowed') is not True
            or manifest.get('mixed_policy_future_hand_training_allowed') is not False
            or manifest.get('actor_future_hand_training_allowed') not in (None, False)
            or manifest.get('dt') != 1/30 or audit.get('pass') is not True
            or audit.get('all_arrays_finite') is not True or audit.get('checks_max') != 0
            or audit.get('check_count') != 2316 or audit.get('controls') != 128
            or audit.get('measured_states') != 129 or audit.get('native_data_sha256') != trajectory_hash
            or manifest.get('trajectory_sha256') != trajectory_hash
            or result.get('partial_episode') is not True or result.get('calibration_pass') is not None
            or result.get('local_grasp_calibration') is not True):
        raise ValueError('completed exact128/t8 source and root audit required')
    shapes = {'q': (129,4,18), 'dq': (129,4,18), 'hand': (129,4,11,3),
        'obj': (129,4,4,4), 'velocity': (129,4,6), 'expert_PD': (128,2,18),
        'expert_raw': (128,2,18), 'expert_action': (128,2,18), 'applied': (128,4,18),
        'pd_targets': (128,4,18), 'expert_applied': (128,2), 'expert_clipped': (128,2),
        'learner_hand_history': (128,2,4,11,3), 'learner_object_history': (128,2,4,4,4),
        'learner_goal': (128,2,4,4)}
    if any(k not in data or data[k].shape != shape or not np.isfinite(data[k]).all() for k,shape in shapes.items()):
        raise ValueError('finite measured source array shapes required')
    mask = np.zeros((128,2),bool); mask[8:,1] = True
    if (data['expert_applied'].dtype != bool or data['expert_clipped'].dtype != bool
            or not np.array_equal(data['expert_applied'], mask)
            or data['expert_clipped'][8:,1].any()
            or data['expert_clipped'][:53,0].any()
            or not np.array_equal(data['applied'][8:,3], data['expert_action'][8:,1])
            or not np.array_equal(data['pd_targets'][8:,3], data['expert_PD'][8:,1])):
        raise ValueError('own recovery expert actual/unclipped controls and pure current labels required')
    native = manifest['native_controller']
    for query, role, ticks in ((0,2,np.arange(105)), (1,3,np.arange(8,128))):
        offset, scale = [np.asarray(native[k],np.float32) for k in ('offset','scale')]
        if offset.shape != (18,) or scale.shape != (18,) or (scale <= 0).any():
            raise ValueError('native controller differs')
        raw = data['expert_raw'][ticks,query]
        action = data['expert_action'][ticks,query]
        if not np.array_equal(action,np.clip(raw,-1,1)):
            raise ValueError('expert action clipping differs')
        converted = action.copy(); converted[:,6:] = (converted[:,6:] + 1)/2
        pd = offset + scale*converted; pd[:,:6] += data['q'][ticks,role,:6]
        for distal,parent,ratio in COUPLING: pd[:,distal] = pd[:,parent]*ratio
        if not np.allclose(pd,data['expert_PD'][ticks,query],atol=1e-6,rtol=0):
            raise ValueError('CURRENT-q expert PD labels differ')
        for tick in ticks:
            past = np.maximum(tick + np.arange(-3,1),0)
            if (not np.array_equal(data['learner_hand_history'][tick,query],data['hand'][past,role])
                    or not np.array_equal(data['learner_object_history'][tick,query],data['obj'][past,role])
                    or not np.array_equal(data['learner_goal'][tick,query],reference[min(tick+24,542)])):
                raise ValueError('own measured H4/planned goal differ')
    return dict(schema='ref2dex.joint-own-expert-admission.v1', admitted=True,
        source_trajectory_sha256=trajectory_hash, role=3, first_tick=8,last_tick=104,
        horizon=24,windows=97,actor_expert_future_training_allowed=True,
        mixed_policy_future_hand_training_allowed=False,pure_role=2,
        pure_current_train_first=0,pure_current_train_last=52,pure_future_training_allowed=False,
        scope='own role3 actual expert futures only; role2 CURRENT expert PD only')


def state_hash(values):
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def buffer_hash(net):
    return state_hash(dict(net.named_buffers()))


def joint_losses(predicted_tau, true_tau, trajectory_valid, predicted_pd, true_pd, target_scale):
    """Absent expert futures never supply a geometry label; PD uses every row."""
    if (predicted_tau.shape != true_tau.shape or predicted_tau.shape[1:] != (24, 33)
            or trajectory_valid.dtype != torch.bool or trajectory_valid.shape != (len(predicted_tau),)
            or predicted_pd.shape != true_pd.shape or predicted_pd.shape != (len(predicted_tau), 12)
            or target_scale.shape != (12,) or not trajectory_valid.any() or (target_scale <= 0).any()):
        raise ValueError('joint valid-right-hand/current independent PD contract differs')
    geometry = ((predicted_tau[trajectory_valid] - true_tau[trajectory_valid]) / .05).square().mean()
    action = ((predicted_pd - true_pd) / target_scale).abs().mean()
    return geometry, action


def action_gradient_evidence(loss, modules):
    """Probe the ACTION loss alone before total backward; preserve its graph."""
    if tuple(modules) != TRAIN_MODULES:
        raise ValueError('full policy chain and R modules required')
    params = [p for net in modules.values() for p in net.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    evidence = {}
    for name, net in modules.items():
        owned = {id(p) for p in net.parameters()}
        selected = [g for p, g in zip(params, grads) if id(p) in owned and g is not None]
        if not selected or any(not torch.isfinite(g).all() for g in selected):
            raise ValueError('missing/nonfinite action gradient: ' + name)
        evidence[name] = sum(float(g.detach().abs().sum()) for g in selected)
        if evidence[name] <= 0:
            raise ValueError('zero action gradient: ' + name)
    return evidence


def offline_screen(final):
    checks = {}
    for name, limit in (('nominal_report', 40.), ('startup_t0', 20.), ('startup_all', 40.)):
        checks[name + '_predicted_PD'] = final[name]['xyz_p95_mm'] <= limit
    for name, limit in (('nominal_report', 30.), ('startup_all', 20.)):
        value = final[name]
        checks[name + '_predicted_clipping'] = value['clipping_rate'] < .01
        checks[name + '_GT_PD'] = value['GT_tau_R_xyz_p95_mm'] <= limit
        checks[name + '_GT_clipping'] = value['GT_tau_R_clipping_rate'] < .01
    for prefix in ('', 'GT_tau_R_'):
        value = final['nominal_report']
        checks[prefix + 'rotation'] = value[prefix + 'rotation_p95_rad'] <= .15
        checks[prefix + 'fingers'] = value[prefix + 'finger_p95_rad'] <= .3
    return checks


def policy_patch_admission(packet, manifest, result):
    if (packet.get('schema') != JOINT_SCHEMA or manifest.get('schema') != JOINT_SCHEMA
            or packet.get('policy_patch_schema') != PATCH_SCHEMA
            or manifest.get('policy_patch_schema') != PATCH_SCHEMA
            or set(packet.get('head', {})) != {'trajectory', 'hand_query'}
            or set(packet.get('policy_patch', {})) != {'scene', 'goal'}
            or 'scene_patch' in packet or 'scene_patch_schema' in packet
            or packet.get('input_sha256') != manifest.get('input_sha256')
            or packet.get('joint_R_sha256') != result.get('checkpoint_sha256')
            or packet.get('joint_R_sha256') != manifest.get('checkpoint_sha256')
            or manifest.get('train_modules') != list(TRAIN_MODULES)):
        raise ValueError('exact jointly trained policy/R binding required')
    hashes = result.get('train_final_hash', {})
    for name, values in dict(packet['policy_patch'], **packet['head']).items():
        if (not isinstance(values, dict) or not values
                or any(not isinstance(v, torch.Tensor) or not torch.isfinite(v).all() for v in values.values())
                or state_hash(values) != hashes.get(name)):
            raise ValueError('joint policy module state/hash differs: ' + name)
    return dict(applied=True, schema=PATCH_SCHEMA,
                module_sha256={name:hashes[name] for name in POLICY_MODULES},
                WAM_buffer_sha256=result['normalization_final_hash']['WAM'],
                joint_R_sha256=packet['joint_R_sha256'])


def load_policy_patch(model, packet, identity):
    """Validate every tensor BEFORE applying any policy change."""
    if buffer_hash(model)!=identity['WAM_buffer_sha256']:
        raise ValueError('joint WAM normalization/config buffers differ from frozen base')
    states = dict(packet['policy_patch'], **packet['head'])
    for name in POLICY_MODULES:
        expected = getattr(model, name).state_dict()
        values = states[name]
        if set(values) != set(expected):
            raise ValueError('joint state keys differ: ' + name)
        for key, value in values.items():
            target = expected[key]
            if value.shape != target.shape or value.dtype != target.dtype or not torch.isfinite(value).all():
                raise ValueError('joint state tensor differs: ' + name + '.' + key)
        # Original normalization and PTv3 statistics are frozen in this recipe.
        for key, value in getattr(model, name).named_buffers():
            # Nonpersistent voxel_origin_m is reconstructed from authenticated
            # base config; its bytes are included in the whole-model hash above.
            if key not in expected:
                continue
            if state_hash({key:values[key]}) != state_hash({key:value}):
                raise ValueError('joint normalization buffer changed: ' + name + '.' + key)
    for name in POLICY_MODULES:
        getattr(model, name).load_state_dict(states[name], strict=True)
        if state_hash(getattr(model, name).state_dict()) != identity['module_sha256'][name]:
            raise ValueError('applied joint module differs: ' + name)


def paired_manifest_admission(manifest, result, head_manifest, head_sha, r_sha):
    if (head_manifest is None or head_manifest != manifest
            or manifest.get('schema') != JOINT_SCHEMA or manifest.get('status') != 'COMPLETED'
            or result.get('status') != 'COMPLETED' or result.get('offline_screen_pass') is not True
            or manifest.get('head_sha256') != head_sha or result.get('head_sha256') != head_sha
            or manifest.get('checkpoint_sha256') != r_sha or result.get('checkpoint_sha256') != r_sha
            or manifest.get('head_training_R_sha256') != r_sha
            or manifest.get('train_modules') != list(TRAIN_MODULES)
            or result.get('normalization_buffers_unchanged') is not True
            or result.get('frozen_initial_hash') != result.get('frozen_final_hash')
            or set(result.get('action_gradient_checks', {})) != set(TRAIN_MODULES)
            or not all(math.isfinite(v) and v > 0 for v in result.get('action_gradient_checks', {}).values())
            or set(result.get('train_initial_hash', {})) != set(TRAIN_MODULES)
            or set(result.get('train_final_hash', {})) != set(TRAIN_MODULES)
            or any(result.get('train_initial_hash', {}).get(k) == result.get('train_final_hash', {}).get(k)
                   for k in TRAIN_MODULES)):
        raise ValueError('completed screened exact learned WAM/R pair required')
