"""CPU SciPy descriptor/protocol audit of completed native oracle artifacts.

Does not import the production descriptor, run physics or fit a model.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('unique audit artifact')
    begin = time.monotonic()
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation

    torch.set_num_threads(2)
    source = args.source.resolve()
    manifest = json.loads((source/'run_manifest.json').read_text())
    if not args.partial and manifest['run_status'] != 'COMPLETED':
        raise ValueError('terminal scientific collection required')
    load = lambda p: torch.load(p, map_location='cpu', weights_only=False)
    baseline = load(source/'baseline/trace.pt')
    initial = load(source/'baseline/initial.pt')
    frozen = Path(manifest['source_frozen_selectors'])
    options = np.load(frozen/'options.npy')
    expected_options = np.concatenate((np.zeros((1, 12)),
                                       np.random.RandomState(805).normal(0, .25, (7, 12)))).astype(np.float32)
    if not np.array_equal(options, expected_options):
        raise ValueError('original eight options')
    packet_path = source/'selection/feature_packet.pt'
    selected = load(packet_path) if packet_path.exists() else None
    if not args.partial and selected is None:
        raise ValueError('actual selector input packet required')
    audits = []
    hashes = {}

    def sha(p):
        digest = hashlib.sha256()
        with p.open('rb') as handle:
            for block in iter(lambda: handle.read(1 << 20), b''):
                digest.update(block)
        return digest.hexdigest()

    def canonical(q):
        return np.where(q[..., 3:4] < 0, -q, q)

    def raw_apply(q, vector, inverse=False):
        # SciPy normalizes input quaternions. Production's cross-product
        # formula consumes the recorded float32 quaternion without that step.
        # Match its algebra independently: M_raw = |q|^2 R_unit + (1-|q|^2)I.
        norm2 = np.sum(q*q, axis=-1)
        rotation = Rotation.from_quat(q)
        result = (rotation.inv() if inverse else rotation).apply(vector)
        return norm2[..., None]*result+(1-norm2[..., None])*vector

    panels = sorted(p.parent for p in source.glob('queries/*/results.json'))
    if not args.partial and len(panels) != 84:
        raise ValueError('all 84 nonzero queries')
    panels = [source/'baseline'] + panels
    for panel in panels:
        target, index = (None, 0) if panel.name == 'baseline' else (int(panel.name[1:3]), int(panel.name[5:7]))
        trace = load(panel/'trace.pt')
        local_initial = load(panel/'initial.pt')
        n = len(initial['motion'])
        ticks = 202 if index == 0 else 68
        if n != 12 or len(trace['object_root']) != ticks:
            raise ValueError('fixed 12 subjects and short query')
        for key in initial:
            if torch.is_tensor(initial[key]) and not torch.equal(initial[key], local_initial[key]):
                raise ValueError('identical initial packet: '+key)
        matrix = np.zeros((n, 12), np.float32)
        if target is not None:
            matrix[target] = options[index]
        if index and not np.array_equal(matrix, np.load(source/'requests'/(panel.name+'.npy'))):
            raise ValueError('exactly one nonzero subject request')
        expected_request = np.zeros((ticks, n, 12), np.float32)
        expected_request[36:] = matrix
        if not np.array_equal(trace['request'].numpy(), expected_request):
            raise ValueError('actual native request chronology')
        background = np.ones(n, bool) if target is None else np.arange(n) != target
        background_error = float((trace['target'][36:, background] -
                                  trace['base_target'][36:, background]).abs().max())
        if background_error > 1e-5:
            raise ValueError('background controller differs from P0')
        for key in ('object_root', 'native_q', 'native_dq', 'rigid_state', 'net_force', 'target', 'action'):
            if not torch.equal(trace[key][:36], baseline[key][:36]):
                raise ValueError('entire world predecision prefix: '+key)

        root = trace['object_root'][36:68].numpy().astype(np.float64)
        body = trace['rigid_state'][36:68, :, :25].numpy().astype(np.float64)
        decision_root = baseline['object_root'][35].numpy().astype(np.float64)
        decision_quaternion = np.tile(decision_root[:, 3:7], (32, 1))
        decision_rotation = Rotation.from_quat(decision_quaternion)
        object_rotation = Rotation.from_quat(root[..., 3:7].reshape(-1, 4))
        delta = raw_apply(decision_quaternion,(root[..., :3]-decision_root[None, :, :3]).reshape(-1, 3),inverse=True).reshape(32,n,3)/.05
        quaternion_scale = (np.linalg.norm(decision_quaternion,axis=-1)*np.linalg.norm(root[...,3:7].reshape(-1,4),axis=-1)).reshape(32,n,1)
        quaternion = canonical((decision_rotation.inv()*object_rotation).as_quat().reshape(32,n,4)*quaternion_scale)
        effect = np.concatenate((delta, quaternion, root[..., 7:10]/.1, root[..., 10:13]), -1).transpose(1, 0, 2).reshape(n, -1)
        repeated_quaternion = np.repeat(root[..., None, 3:7], 25, axis=2).reshape(-1,4)
        repeated_object = Rotation.from_quat(repeated_quaternion)
        object_inverse = repeated_object.inv()
        displacement = body[..., :3]-root[..., None, :3]
        relative_velocity = body[..., 7:10]-root[..., None, 7:10]-np.cross(root[..., None, 10:13], displacement)
        relative_pose = np.concatenate((
            raw_apply(repeated_quaternion,displacement.reshape(-1,3),inverse=True).reshape(32,n,25,3)/.1,
            canonical((object_inverse*Rotation.from_quat(body[...,3:7].reshape(-1,4))).as_quat().reshape(32,n,25,4)*(np.linalg.norm(repeated_quaternion,axis=-1)*np.linalg.norm(body[...,3:7].reshape(-1,4),axis=-1)).reshape(32,n,25,1)),
            raw_apply(repeated_quaternion,relative_velocity.reshape(-1,3),inverse=True).reshape(32,n,25,3)/.1,
            raw_apply(repeated_quaternion,(body[...,10:13]-root[...,None,10:13]).reshape(-1,3),inverse=True).reshape(32,n,25,3)), -1)

        contacts = np.load(panel/'contacts.npy', mmap_mode='r')
        frames = json.loads((panel/'contact_frames.json').read_text())
        physics = load(panel/'physics_states.pt')['rigid_state'].numpy()
        metadata = json.loads((panel/'physical_metadata.json').read_text())
        weight = np.array([actors[2]['properties'][0]['mass']*9.81 for actors in metadata['actor_body_properties']])
        moments = np.zeros((32, n, 26, 15), np.float64)
        offset = 0
        hand_records = hand_positive = 0
        expected_frames = [(tick, subtick, env) for tick in range(36, 68) for subtick in range(2) for env in range(n)]
        if [(f['tick'], f['subtick'], f['env']) for f in frames] != expected_frames:
            raise ValueError('complete unique ordered contact frame coverage')
        for frame in frames:
            if frame['offset'] != offset:
                raise ValueError('contiguous raw contact storage')
            raw = contacts[offset:offset+frame['count']]
            offset += frame['count']
            raw = raw[(raw['body0'] == 26) | (raw['body1'] == 26)]
            first = raw['body0'] == 26
            partners = np.where(first, raw['body1'], raw['body0']).astype(int)
            if np.any((partners < 0) | (partners > 25)):
                raise ValueError('native attributed partner identity')
            force = np.maximum(raw['lambda'].astype(np.float64), 0)
            normals = np.column_stack([raw['normal'][axis] for axis in ('x', 'y', 'z')])
            signed_force = normals*(force*np.where(first, 1., -1.))[:, None]
            orientation = physics[frame['tick']*2+frame['subtick'],frame['env'],26,3:7].astype(np.float64)
            local_force = raw_apply(orientation,signed_force,inverse=True) if len(raw) else np.empty((0,3))
            point = np.column_stack([np.where(first, raw['localPos0'][axis], raw['localPos1'][axis]) for axis in ('x', 'y', 'z')]).astype(np.float64)
            second = np.stack([point[:, i]*point[:, j] for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))], -1)
            values = np.column_stack((np.ones(len(raw)), force > 1e-8, force,
                                      local_force, point*force[:, None], second*force[:, None]))*.5
            # Group by partner, independently of production np.add.at aggregation.
            for partner in np.unique(partners):
                moments[frame['tick']-36, frame['env'], partner] += values[partners == partner].sum(0)
            if target is None or frame['env'] == target:
                hand_records += int((partners < 25).sum())
                hand_positive += int(((partners < 25) & (force > 1e-8)).sum())
        if offset != len(contacts):
            raise ValueError('all raw contacts used')
        total = moments[..., 2:3].copy()
        moments[..., :2] = np.log1p(moments[..., :2])
        moments[..., 2:6] /= weight[None, :, None, None]
        moments[..., 6:9] /= np.maximum(total, 1e-12)*.05
        moments[..., 9:] /= np.maximum(total, 1e-12)*.05**2
        interaction = np.concatenate((relative_pose.transpose(1, 0, 2, 3).reshape(n, -1),
                                      moments.transpose(1, 0, 2, 3).reshape(n, -1)), -1)
        common = np.concatenate((baseline['context'][36].numpy(), matrix), -1)
        feature = np.concatenate((common, effect, interaction), -1).astype(np.float32)
        feature_error = None
        if selected is not None:
            difference = feature-selected['features'][:, 0] if target is None else feature[target]-selected['features'][target, index]
            feature_error = float(np.max(np.abs(difference)))
            if feature_error > 1e-4:
                raise ValueError('independent SciPy descriptor: '+panel.name+' '+str(feature_error))
        if feature.shape != (12, 23378) or not np.isfinite(feature).all():
            raise ValueError('finite full privileged feature dimensions')
        audits.append(dict(panel=panel.name,subject=target,option=index,
                           background_P0_target_max_error=background_error,
                           independent_descriptor_max_error=feature_error,
                           subject_hand_object_records=hand_records,
                           subject_positive_normal_hand_object_records=hand_positive))
        for name in ('trace.pt','initial.pt','contacts.npy','contact_frames.json','physics_states.pt','physical_metadata.json'):
            path = panel/name
            hashes[str(path)] = sha(path)
        print(json.dumps(dict(audited=panel.name,descriptor_error=feature_error)), flush=True)
        if time.monotonic()-begin > 180:
            raise TimeoutError('bounded CPU artifact audit')

    terminal_audit = None
    if not args.partial:
        observer = json.loads((source/'native_audit_manifest.json').read_text())
        outcomes = json.loads((source/'results.json').read_text())
        rows = json.loads((source/'rows.json').read_text())
        if observer['run_status'] != 'COMPLETED' or observer['scientific_run_status'] != 'COMPLETED':
            raise ValueError('terminal native artifact observer')
        actual_panels = [source/'baseline']+sorted(p.parent for p in source.glob('deploy/*/results.json'))
        labels = {}
        physics_fidelity = {}
        for panel in actual_panels:
            native = json.loads((panel/'native_audit.json').read_text())
            if native['run_status'] != 'COMPLETED' or not native['all105_verified']:
                raise ValueError('independent full105 mesh audit')
            trace = load(panel/'trace.pt');init = load(panel/'initial.pt')
            stops = init['phase_stop'][init['motion']].numpy()
            mask = (np.arange(202)[:, None] >= stops[None, :]-74) & (np.arange(202)[:, None] <= stops[None, :]+30)
            good = ((trace['object_root'][:, :, 2].numpy()-init['initial_height'].numpy()[None, :]) >= .03) & (trace['clearance'].numpy() >= .02)
            recorded = (good | ~mask).all(0).tolist()
            if not np.all(mask.sum(0) == 105) or recorded != native['episode_success']:
                raise ValueError('all episode labels match independent SciPy mesh labels')
            labels[str(panel.resolve())] = native['episode_success']
            if panel.name != 'baseline':
                actual_physics = load(panel/'physics_states.pt')
                query_physics = load(source/'queries'/panel.name/'physics_states.pt')
                errors = {key:float((actual_physics[key][:136]-query_physics[key]).abs().max())
                          for key in ('rigid_state','net_force')}
                physics_fidelity[panel.name] = errors
                if any(value != 0 for value in errors.values()):
                    raise ValueError('entire world both physics steps differ: '+panel.name)
        motion = initial['motion'].numpy();counts = {}
        baseline_labels = np.array(labels[str((source/'baseline').resolve())], bool)
        def count(y):
            return dict(successes=int(y.sum()),episodes=12,by_motion=[int(y[motion == m].sum()) for m in range(3)])
        counts['P0'] = count(baseline_labels)
        for arm in ('state','effect','interaction','joint'):
            arm_rows = sorted((r for r in rows if r['arm'] == arm),key=lambda r:r['subject'])
            if [r['subject'] for r in arm_rows] != list(range(12)):
                raise ValueError('all subjects included for '+arm)
            y = []
            for row in arm_rows:
                truth = labels[str(Path(row['actual_rollout']).resolve())][row['subject']]
                if truth != row['success'] or row['motion'] != int(motion[row['subject']]):
                    raise ValueError('actual selected outcome identity')
                y.append(truth)
            counts[arm] = count(np.array(y,bool))
        if counts != outcomes['counts']:
            raise ValueError('independent actual counts')
        gates = dict(joint_vs_state_5pp=(counts['joint']['successes']-counts['state']['successes'])/12 >= .05,
                     joint_vs_effect_5pp=(counts['joint']['successes']-counts['effect']['successes'])/12 >= .05,
                     joint_at_least_P0=counts['joint']['successes'] >= counts['P0']['successes'])
        if any(outcomes['gates'][key] != value for key,value in gates.items()):
            raise ValueError('fixed utility gates')
        expected_class = 'UNCLEAR' if not outcomes['gates']['oracle_contract_verified'] else ('PROMISING' if all(gates.values()) else 'UNPROMISING')
        if outcomes['conclusion'] != expected_class:
            raise ValueError('bounded Probe classification')
        terminal_audit = dict(all_full105_labels_agree=True,counts=counts,gates=gates,
                              conclusion=expected_class,actual_native_worlds=len(actual_panels),
                              whole_world_first136_physics_frames=physics_fidelity)
    result = dict(run_status='COMPLETED',partial=args.partial,query_panels=len(audits)-1,
                  all_zero_option_subjects_audited=True,terminal_audit=terminal_audit,
                  independent_scipy_geometry=True,independent_partner_grouping=True,
                  no_new_physics_or_model_updates=True,cpu_reason='Independent SciPy saved-artifact arithmetic and protocol verification',
                  wall_seconds=time.monotonic()-begin,panels=audits,input_sha256=hashes,
                  script_sha256=sha(Path(__file__)))
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
