#!/usr/bin/env python3
"""Validate downloaded state/control and demonstrate fixed-identity mesh flow.

CPU engineering audit only. Saved-state forward kinematics is not control replay.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import mujoco
import numpy as np
import yaml


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    a = p.parse_args(); root = a.run_dir.resolve(); start = time.monotonic()
    dataset = root/'dataset'; reports = []; destination = root/'pointflow_smoke'
    destination.mkdir(exist_ok=True)
    checks = json.loads((root/'checksums.json').read_text())
    for path in sorted(dataset.glob('processed/*/inspire/right/*/0/trajectory_mjwp.npz')):
        task_dir = path.parent.parent
        config = yaml.safe_load((path.parent/'config.yaml').read_text())
        model = mujoco.MjModel.from_xml_path(str(task_dir/'scene.xml'))
        with np.load(path, allow_pickle=False) as data:
            arrays = {}
            for key, width in (('qpos',model.nq),('qvel',model.nv),('ctrl',model.nu)):
                arrays[key] = data[key].reshape(-1,width).copy()
                assert np.isfinite(arrays[key]).all(), key
            times = data['time'].reshape(-1).copy()
        n = len(times)
        assert (model.nq,model.nv,model.nu) == (25,24,18)
        assert n > 8 and all(len(x)==n for x in arrays.values())
        assert np.isfinite(times).all() and np.allclose(np.diff(times),config['sim_dt'],atol=1e-5)
        assert np.allclose(np.linalg.norm(arrays['qpos'][:,-4:],axis=-1),1,atol=1e-5)
        object_id = model.body('right_object').id
        assert model.joint('right_object_joint').qposadr[0] == model.nq-7
        with np.load(path.parent/'trajectory_kinematic.npz',allow_pickle=False) as reference:
            qref=reference['qpos'].copy()
            assert qref.ndim==2 and qref.shape[-1]==model.nq and np.isfinite(qref).all()
        # Each point keeps a fixed (mesh geom, vertex) identity at all frames.
        specs=[]
        for geom in range(model.ngeom):
            mesh=int(model.geom_dataid[geom]); body=int(model.geom_bodyid[geom])
            if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            if body==object_id:
                if model.geom(geom).name!='right_object_visual': continue
            elif model.geom_group[geom]!=1: continue
            if body==0 or body>object_id: continue
            offset=int(model.mesh_vertadr[mesh]); count=int(model.mesh_vertnum[mesh])
            ids=np.linspace(0,count-1,8,dtype=int)
            specs.append((geom,body,model.mesh_vert[offset+ids].copy()))
        assert any(body==object_id for _,body,_ in specs)
        data=mujoco.MjData(model); points=[]
        for tick in range(n):
            data.qpos[:]=arrays['qpos'][tick];data.qvel[:]=arrays['qvel'][tick]
            data.ctrl[:]=arrays['ctrl'][tick];data.time=times[tick]
            mujoco.mj_forward(model,data)
            points.append(np.concatenate([v@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g]
                                           for g,_,v in specs],axis=0))
        points=np.stack(points); assert np.isfinite(points).all()
        object_mask=np.repeat([b==object_id for _,b,_ in specs],8)
        flow=points[8:]-points[:-8]
        assert np.isfinite(flow).all()
        source=path.relative_to(dataset).parts[1]; task=task_dir.name
        np.savez_compressed(destination/(source+'.npz'),times=times,hand_points=points[:,~object_mask],
                            object_points=points[:,object_mask],flow_h8=flow,
                            object_point_mask=object_mask,ctrl=arrays['ctrl'],qpos=arrays['qpos'],
                            vertex_geom=np.repeat([g for g,_,_ in specs],8))
        joint_names=[model.joint(i).name for i in range(model.njnt)]
        q=lambda name:arrays['qpos'][:,int(model.joint('right_'+name).qposadr[0])]
        coupling={finger:float(np.abs(q(finger+'_intermediate_joint')-1.05*q(finger+'_proximal_joint')).max())
                  for finger in ('index','middle','ring','pinky')}
        coupling['thumb_intermediate']=float(np.abs(q('thumb_intermediate_joint')-.6*q('thumb_proximal_pitch_joint')).max())
        coupling['thumb_distal']=float(np.abs(q('thumb_distal_joint')-.8*q('thumb_proximal_pitch_joint')).max())
        metrics=json.loads((path.parent/'metrics.json').read_text())
        reports.append(dict(source=source,task=task,frames=n,qpos_width=model.nq,qvel_width=model.nv,
                            ctrl_width=model.nu,sim_dt=config['sim_dt'],ref_dt=config['ref_dt'],
                            ctrl_dt=config['ctrl_dt'],recorded_success=metrics.get('success'),
                            object_lift=float(arrays['qpos'][-1,-5]-arrays['qpos'][0,-5]),
                            hand_points=int((~object_mask).sum()),object_points=int(object_mask.sum()),
                            mean_flow_norm=float(np.linalg.norm(flow,axis=-1).mean()),
                            joint_names=joint_names,native_coupling_max_error_radians=coupling,
                            raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        assert reports[-1]['raw_sha256']==checks[path.relative_to(dataset).as_posix()]
    assert len(reports)==4 and len({r['source'] for r in reports})==4
    result=dict(status='PASS',revision=json.loads((root/'dataset_info.json').read_text())['sha'],
                mujoco_version=mujoco.__version__,trials=reports,elapsed_seconds=time.monotonic()-start,
                scope='Four-source engineering sample. FK mesh vertex flow, not surface-uniform training samples or dynamics/control replay.',
                flow_lag_frames=8,flow_lag_seconds=.08,inspire_total_trajectories=1946,
                action_semantics='MuJoCo actuator controls, not normalized Isaac Gym PPO actions',
                missing_force_contact_labels=True,isaac_control_replay='NOT_TESTED')
    (root/'sample_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
