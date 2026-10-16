"""Independent array audit of actual commands, PD conversion and chunk clock."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path[:0] = [str(TASK / "src"), str(ROOT / "src/task/trajectory-policy/tools/audit"),
                str(ROOT / "src/task/consequence-evaluator/src")]
from audit_competent_pd import independent_pd, independent_features, ACTIVE
from ptv3_shared_wam.checkpoint import atomic_json


def audit(path):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    result = json.loads((path / "result.json").read_text())
    if manifest["status"] != "COMPLETED" or result["status"] != "COMPLETED":
        raise ValueError("complete native execution required")
    with np.load(path / "trajectory.npz", allow_pickle=False) as stream:
        d = {key: stream[key].copy() for key in stream.files}
    controls, steps = manifest["controls"], manifest["execution_steps"]
    if len(d["q"]) != controls + 1 or len(d["applied"]) != controls:
        raise ValueError("terminal measured-state boundary differs")
    errors = {}
    def check(name, actual, expected):
        if actual.shape != expected.shape:
            raise ValueError("shape differs: " + name)
        difference = float(np.max(np.abs(actual.astype(float) - expected.astype(float))))
        errors[name] = difference
        if difference > 1e-6:
            raise ValueError("contract differs: " + name)
    if not all(np.isfinite(value).all() for value in d.values()):
        raise ValueError("nonfinite execution arrays")
    check("clamped_commands", d["applied"], d["intended"].clip(-1, 1))
    check("native_PD", d["pd_targets"], independent_pd(d["applied"], d["q"][:-1], manifest["native_controller"]))
    check("clipping_flags", d["clipped"], np.abs(d["intended"] - d["applied"]).max(-1) > 1e-6)
    student_intended=d['intended'][:,1:]
    if manifest.get('current_observation_input') is not None:
        if (manifest['current_observation_input'] not in ('masked','delta0') or steps!=1
                or not manifest.get('shared_actor') or d['native_current_obs'].shape!=(controls,3,721)):
            raise ValueError('Current shared actor execution contract differs')
        check('server_own_current_observations',d['server_current_obs'],d['native_current_obs'][:,[2,1]])
    if manifest.get('diagnostic_hybrid'):
        if (manifest.get('no_teacher_intervention') is not False
                or manifest.get('training_allowed') is not False
                or not manifest.get('shared_actor') or steps != 1
                or manifest.get('roles') != ['teacher', 'teacher_wrist_student_fingers',
                                             'student_wrist_teacher_fingers']):
            raise ValueError('Oracle component probe cannot claim autonomous student execution')
        if d['teacher_obs'].shape != (controls,3,1442):
            raise ValueError('Own-row full teacher observations required')
        mixed=d['student_intended'].copy()
        mixed[:,1,:6]=d['teacher_actions'][:,1,:6]
        mixed[:,2,list(ACTIVE)[6:]]=d['teacher_actions'][:,2,list(ACTIVE)[6:]]
        check('teacher_components_only',d['intended'],mixed)
        check('teacher_row_command',d['intended'][:,0],d['teacher_actions'][:,0])
        student_intended=d['student_intended'][:,1:]
    if manifest.get('wrist_probe') or manifest.get('wrist_shared_probe'):
        expected_roles=(['teacher','none_wrist_shared_fingers','aligned_wrist_shared_fingers'] if manifest.get('wrist_shared_probe') else
                        ['teacher', 'state_goal_wrist_shared_fingers', 'current721_wrist_shared_fingers'])
        if (not manifest.get('shared_actor') or steps != 1 or manifest.get('diagnostic_hybrid')
                or manifest.get('no_teacher_intervention') is not True
                or manifest['roles'] != expected_roles):
            raise ValueError('Current-only wrist probe roles/contract differ')
        if d['native_current_obs'].shape != (controls, 3, 721):
            raise ValueError('Own-row current721 required')
        mixed = d['pre_wrist_intended'].copy()
        mixed[:, 1:, :6] = d['wrist_prediction']
        check('supervised_wrist_only', d['intended'], mixed)
        student_intended = d['pre_wrist_intended'][:, 1:]
    ticks = np.arange(controls)
    query_ticks = ticks - ticks % steps
    check("query_tick", d["query_tick"], np.repeat(query_ticks[:, None], 2, axis=1))
    check("chunk_index", d["chunk_index"], np.repeat((ticks % steps)[:, None], 2, axis=1))
    representation=manifest.get('action_representation','native-command.v1')
    if representation=='native-command.v1':
        target=d['R_output_chunk'][np.arange(controls)[:,None],np.arange(2)[None,:],(ticks%steps)[:,None]]
        check('native_chunk_to_intended_command',student_intended[:,:,ACTIVE],target)
    if representation!='native-command.v1':
        check('physical_query_q',d['R_query_q'],d['q'][query_ticks,1:])
        target=d['R_output_chunk'][np.arange(controls)[:,None],np.arange(2)[None,:],(ticks%steps)[:,None]]
        desired=target[...,:12]+d['R_query_q'][...,ACTIVE]
        if representation in ('wrist-geometry-relative-current.v1','wrist-geometry-continuous-lift.v1'):
            from scipy.spatial.transform import Rotation
            current=Rotation.from_euler('XYZ',d['R_query_q'][...,3:6].reshape(-1,3))
            relative=Rotation.from_rotvec(target[...,3:6].reshape(-1,3))
            # Check physical orientation independently of the Euler branch used
            # by the live controller. Generalized-coordinate roundtrip is a
            # separate prerequisite for admitting this representation.
            target_rotation=(current*relative).as_matrix()
            live=d['q'][:-1,1:]
            intended_pd=independent_pd(student_intended,live,manifest['native_controller'])
            actual_rotation=Rotation.from_euler('XYZ',intended_pd[...,3:6].reshape(-1,3)).as_matrix()
            if representation != 'wrist-geometry-continuous-lift.v1':
                check('geometry_buffer_orientation',actual_rotation,target_rotation)
            if representation=='wrist-geometry-continuous-lift.v1':
                reference=target[...,12:15]+d['R_query_q'][...,3:6]
                # The executed buffer stores absolute rotvec in model dtype.
                # Reconstruct that quantization before lifting near-singular
                # Euler angles; double precision alone changes the command.
                absolute=Rotation.from_rotvec((current*relative).as_rotvec().astype(target.dtype))
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',UserWarning)
                    first=absolute.as_euler('XYZ').reshape(reference.shape)
                other=np.stack((first[...,0]+np.pi,np.pi-first[...,1],first[...,2]+np.pi),-1)
                choices=np.stack((first,other),-2)
                choices+=2*np.pi*np.round((reference[...,None,:]-choices)/(2*np.pi))
                selected=np.square(choices-reference[...,None,:]).sum(-1).argmin(-1)
                angles=np.take_along_axis(choices,selected[...,None,None],axis=-2)[...,0,:]
                singular=np.abs(np.cos(first[...,1]))<1e-7
                sign=np.where(np.sin(first[...,1])>=0,1.,-1.)
                phase=(first[...,0]-reference[...,0]-sign*reference[...,2]+np.pi)%(2*np.pi)-np.pi
                angles[...,0]=np.where(singular,reference[...,0]+phase/2,angles[...,0])
                angles[...,2]=np.where(singular,reference[...,2]+sign*phase/2,angles[...,2])
                mismatch=(Rotation.from_euler('XYZ',reference.reshape(-1,3)).inv()*absolute).magnitude().reshape(reference.shape[:-1])
                lifted=np.where((mismatch<1e-6)[...,None],reference,angles)
                check('geometry_buffer_orientation',
                      Rotation.from_euler('XYZ',lifted.reshape(-1,3)).as_matrix(),absolute.as_matrix())
                desired[...,3:6]=lifted
                # Separately report unavoidable buffer/Euler/controller float32
                # roundoff; validate executed commands against their independent
                # finite-precision reconstruction below, not ideal real numbers.
                errors['geometry_float32_pose_roundtrip']=float(np.abs(actual_rotation-absolute.as_matrix()).max())
        elif representation!='pd-relative-current.v1':raise ValueError('unknown physical target units')
        controller={key:np.asarray(value,dtype=d['q'].dtype) for key,value in manifest['native_controller'].items()};live=d['q'][:-1,1:]
        raw=(desired-np.asarray(controller['offset'])[ACTIVE])/np.asarray(controller['scale'])[ACTIVE]
        raw[...,:6]=(desired[...,:6]-live[...,:6]-np.asarray(controller['offset'])[:6])/np.asarray(controller['scale'])[:6]
        raw[...,6:]=raw[...,6:]*2-1
        if representation=='wrist-geometry-continuous-lift.v1':
            reconstructed_angles=(controller['offset'][3:6]+controller['scale'][3:6]*raw[...,3:6])+live[...,3:6]
            check('geometry_controller_orientation',actual_rotation,
                  Rotation.from_euler('XYZ',reconstructed_angles.reshape(-1,3)).as_matrix())
        if representation in ('pd-relative-current.v1','wrist-geometry-continuous-lift.v1'):check('query_frozen_target_to_live_command',student_intended[:,:,ACTIVE],raw)
        else:
            check('geometry_buffer_position',student_intended[:,:,:3],raw[...,:3])
            check('geometry_buffer_fingers',student_intended[:,:,ACTIVE][...,6:],raw[...,6:])
    past = np.maximum(query_ticks[:, None] + np.arange(-3, 1), 0)
    check("own_H4_hand_at_query", d["wam_hand_history"], d["hand"][past, 2])
    check("own_H4_object_at_query", d["wam_object_history"], d["obj"][past, 2])
    dummy_future = np.repeat(d["hand"][query_ticks, 2][:, None], 24, axis=1)
    state, _ = independent_features(d, query_ticks, np.full(controls, 2), dummy_future)
    check("own_R_state_at_query", d["R_state"], state)
    if manifest.get('shared_actor'):
        check('repeat_own_H4_hand', d['repeat_hand_history'], d['hand'][past,1])
        check('repeat_own_H4_object', d['repeat_object_history'], d['obj'][past,1])
        dummy = np.repeat(d['hand'][query_ticks,1][:,None],24,axis=1)
        repeat_state, _ = independent_features(d,query_ticks,np.full(controls,1),dummy)
        check('repeat_own_R_state', d['repeat_R_state'], repeat_state)
    if manifest.get('wrist_probe') or manifest.get('wrist_shared_probe'):
        independent_state = np.stack((repeat_state, state), 1)
        goal_delta = d['wam_goal'][:, None, :3, 3]-d['obj'][:-1, 1:, :3, 3]
        expected_features = np.concatenate((independent_state, goal_delta, d['native_current_obs'][:, 1:]), -1)
        check('wrist_own_current_features', d['wrist_features'], expected_features)
        import torch
        sys.path.insert(0, str(TASK.parent/'ptv3-shared-wam-v2/src'))
        torch.set_num_threads(2)
        arms=('none','aligned') if manifest.get('wrist_shared_probe') else ('state_goal','plus_current')
        if manifest.get('wrist_shared_probe'):
            check('pooled_shared_own_row_order',d['wrist_shared_features'],d['server_pooled_shared'][:,[1,0]])
        for row, arm in enumerate(arms):
            if manifest.get('wrist_shared_probe'):
                from ptv3_shared_wam_v2.wrist_shared_execution import load_matched_wrist,predict_matched_wrist
                probe=load_matched_wrist(Path(manifest['wrist_shared_probe'])/arm/'final.pt',arm,manifest['checkpoint'],'cpu')
                replay=predict_matched_wrist(probe,torch.from_numpy(d['wrist_features'][:,row]),torch.from_numpy(d['wrist_shared_features'][:,row])).numpy()
            else:
                from ptv3_shared_wam_v2.wrist_probe import load_wrist_probe, predict_wrist
                probe = load_wrist_probe(Path(manifest['wrist_probe'])/arm/'step06000.pt', arm, 'cpu')
                replay = predict_wrist(probe, torch.from_numpy(d['wrist_features'][:, row])).numpy()
            error = float(np.abs(replay-d['wrist_prediction'][:, row]).max())
            errors['wrist_model_replay_'+arm] = error
            # CPU/GPU matmul rounding is separate from exact command mixing/PD audits.
            if error > 1e-4:
                raise ValueError('Wrist model replay differs beyond float32 device tolerance')
    import torch
    # Metadata-only historical audit also accepts old goal contracts. It never
    # instantiates a model or reinterprets legacy weights under the new goal.
    packet = torch.load(manifest["checkpoint"], map_location="cpu", weights_only=False)
    if packet.get("schema") != manifest["schema"] or packet.get("stage") != "robot":
        raise ValueError("matching robot checkpoint metadata required")
    selected = manifest.get("native_input", {})
    if selected.get("kind") == "rl660":
        source = Path(packet["provenance"]["source"]).resolve()
        corpus = json.loads((source / "manifest.json").read_text())
        records = [r for r in corpus["records"] if r["sequence"] == selected["sequence"]]
        if len(records) != 1:
            raise ValueError("native audit requires one exact original corpus record")
        folder = (source / records[0]["path"]).resolve()
        if source not in folder.parents or str(folder / "trajectory.npz") != selected["trajectory"]:
            raise ValueError("native audit sequence path differs from original corpus")
        if selected["object"] != records[0]["object"]:
            raise ValueError("native audit object differs from selected sequence")
        with np.load(folder / "trajectory.npz", allow_pickle=False) as stream:
            reference = stream["reference_obj"].copy()
    else:
        with np.load(packet["provenance"]["reference"], allow_pickle=False) as stream:
            reference = stream["obj"].copy()
    check("planned_goal_at_query", d["wam_goal"], reference[np.minimum(query_ticks + 24, len(reference) - 1)])
    passive = [i for i in range(18) if i not in ACTIVE]
    check("passive_commands_zero", d["applied"][:, 1:, passive], np.zeros_like(d["applied"][:, 1:, passive]))
    outcomes = []
    for env, saved in enumerate(result["outcomes"]):
        height = d["obj"][:, env, 2, 3] - d["obj"][0, env, 2, 3]
        held = (d["surface_gap"][:, env] <= .01) & (height >= .03)
        held &= ~(d["table_footprint"][:, env].astype(bool) & (np.abs(d["support_gap"][:, env]) <= .02))
        held[0] = False
        run = longest = 0
        for value in held:
            run = run + 1 if value else 0
            longest = max(longest, run)
        if longest != saved["maximum_held_frames"]:
            raise ValueError("independent held count differs")
        outcomes.append(dict(role=saved["role"], maximum_held_frames=int(longest)))
    return dict(pass_=True, audit_contract="float32-buffer-controller.v2", errors=errors, outcomes=outcomes,
                scope="execution wiring only; no trained-policy utility claim")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native", type=Path, required=True)
    a = p.parse_args()
    destination = a.native / "independent_audit.json"
    if destination.exists():
        raise FileExistsError(destination)
    report = audit(a.native)
    report["pass"] = report.pop("pass_")
    atomic_json(destination, report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
