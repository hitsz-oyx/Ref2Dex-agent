"""Independent coefficients, SciPy URDF geometry, NumPy network and gates."""
import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_contact_response_probe import sha
from src.task.CmResidual.surface_motion_prior import numpy_predict


def pose(root):
    matrices=np.broadcast_to(np.eye(4),(len(root),4,4)).copy()
    matrices[:,:3,:3]=Rotation.from_quat(root[:,3:7]).as_matrix();matrices[:,:3,3]=root[:,:3]
    return matrices


def independent_links(q,urdf):
    elements=ET.parse(urdf).getroot();children={};child_names=set();coordinate=0
    mapping=[0,1,2,3,4,5,10,11,12,13,16,17,14,15,6,7,8,9]
    for joint in elements.findall('joint'):
        parent=joint.find('parent').get('link');child=joint.find('child').get('link');child_names.add(child)
        kind=joint.get('type');origin=joint.find('origin');o=np.eye(4)
        if origin is not None:
            o[:3,:3]=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix()
            o[:3,3]=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
        axis=joint.find('axis');v=np.fromstring(axis.get('xyz') if axis is not None else '0 0 0',sep=' ')
        index=coordinate if kind!='fixed' else -1
        if index>=0:coordinate+=1
        children.setdefault(parent,[]).append((child,kind,o,v,index))
    root=next(link.get('name') for link in elements.findall('link') if link.get('name') not in child_names)
    native=np.asarray(q,np.float64)[:,np.argsort(mapping)];result={root:np.broadcast_to(np.eye(4),(len(q),4,4)).copy()}
    def visit(parent):
        for child,kind,o,axis,index in children.get(parent,[]):
            m=np.broadcast_to(np.eye(4),(len(q),4,4)).copy()
            if kind!='fixed':
                axis=axis/np.linalg.norm(axis)
                if kind=='prismatic':m[:,:3,3]=native[:,index,None]*axis
                else:m[:,:3,:3]=Rotation.from_rotvec(native[:,index,None]*axis).as_matrix()
            result[child]=result[parent]@o@m;visit(child)
    visit(root);return result


def independent_samples(urdf,names):
    import trimesh
    triangles=[];links=[]
    for link in ET.parse(urdf).getroot().findall('link'):
        for visual in link.findall('visual'):
            mesh=visual.find('./geometry/mesh')
            if mesh is None:continue
            filename=mesh.get('filename');path=Path(filename) if Path(filename).is_absolute() else urdf.parent/filename
            asset=trimesh.load(str(path.resolve()),force='mesh',process=False)
            scale=np.fromstring(mesh.get('scale','1 1 1'),sep=' ',dtype=np.float32).astype(np.float64)
            if len(scale)==1:scale=np.repeat(scale,3)
            origin=visual.find('origin');rotation=np.eye(3);translation=np.zeros(3)
            if origin is not None:
                rotation=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ',dtype=np.float32)).as_matrix()
                translation=np.fromstring(origin.get('xyz','0 0 0'),sep=' ',dtype=np.float32).astype(np.float64)
            vertices=np.asarray(asset.vertices)*scale@rotation.T+translation
            tri=vertices[np.asarray(asset.faces)];cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);valid=np.linalg.norm(cross,axis=-1)/2>1e-12
            triangles.append(tri[valid]);links.append(np.full(int(valid.sum()),names.index(link.get('name')),np.int64))
    triangles=np.concatenate(triangles);links=np.concatenate(links)
    cross=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]);area=np.linalg.norm(cross,axis=-1)/2
    generator=np.random.default_rng(2024);ids=generator.choice(len(triangles),10135,p=area/area.sum())
    u=np.sqrt(generator.random(10135));v=generator.random(10135);bary=np.stack((1-u,u*(1-v),u*v),-1)
    return triangles[ids],links[ids],bary


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--native-source',type=Path,required=True);p.add_argument('--prior-source',type=Path,required=True);a=p.parse_args()
    out=a.root/'qualified';r=json.loads((out/'results.json').read_text());d=a.native_source/'s655'
    initial=torch.load(d/'initial.pt',map_location='cpu',weights_only=False);trace=torch.load(d/'trace.pt',map_location='cpu',weights_only=False)
    metadata=json.loads((d/'physical_metadata.json').read_text());selection={};generator=np.random.default_rng(4001)
    motion=initial['motion'].numpy();arm=initial['arm_assignment'].numpy();train=[];held=[]
    for m in range(3):
        for g in range(4):
            ids=np.flatnonzero((motion==m)&(arm==g));assert len(ids)==64;ids=generator.permutation(ids);train.extend(ids[:32]);held.extend(ids[32:])
    train=np.sort(train);held=np.sort(held);assert not set(train)&set(held)
    ticks=np.linspace(1,200,16,dtype=int)
    for name,envs in (('train',train),('held',held)):
        with np.load(out/(name+'_rows.npz')) as f:rows={key:f[key] for key in f.files}
        env=np.repeat(envs,16);tick=np.tile(ticks,384);assert np.array_equal(rows['env'],env) and np.array_equal(rows['tick'],tick)
        old=np.concatenate((initial['object_root'][None].numpy(),trace['object_root'].numpy()),0)
        expected=dict(q=trace['native_q'][tick-1,env].numpy(),dq=trace['native_dq'][tick-1,env].numpy(),
                      target=trace['target'][tick,env].numpy(),next_q=trace['native_q'][tick,env].numpy(),
                      current_obj=trace['object_root'][tick-1,env].numpy(),previous_obj=old[tick-1,env],next_obj=trace['object_root'][tick,env].numpy(),
                      hand_root=trace['hand_root'][tick-1,env].numpy(),sdk_next_positions=trace['hand_body_position'][tick,env].numpy(),
                      sdk_next_quaternions=trace['hand_body_quaternion'][tick,env].numpy(),motion=motion[env],arm=arm[env])
        for key,value in expected.items():assert np.array_equal(value,rows[key]),(name,key)
        selection[name]=rows
    fit=json.loads((out/'execution_fit.json').read_text());rows=selection['train']
    signals=np.stack((rows['target']-rows['q'],rows['dq']/30),-1).astype(np.float64)
    scale=np.maximum(np.sqrt(np.mean(signals**2,axis=0)),1e-4);assert np.allclose(scale,fit['scale'],atol=1e-12,rtol=1e-10)
    coefficient_max=0.
    for mode in ('velocity_only','action_velocity'):
        x=signals/scale
        if mode=='velocity_only':x=x.copy();x[...,0]=0
        x=np.concatenate((x,np.ones_like(x[...,:1])),-1);y=(rows['next_q']-rows['q']).astype(np.float64)
        for joint in range(18):
            rebuilt=np.linalg.solve(x[:,joint].T@x[:,joint]+1e-6*np.eye(3),x[:,joint].T@y[:,joint])
            error=float(np.abs(rebuilt-np.asarray(fit['coefficients'][mode])[joint]).max());coefficient_max=max(coefficient_max,error);assert error<1e-6
    rows=selection['held'];features=np.load(out/'features.npz');joints=np.load(out/'joint_predictions.npz');errors=np.load(out/'hand_errors.npz');predictions=np.load(out/'predictions.npz')
    gap_signals=np.stack((rows['target']-rows['q'],rows['dq']/30),-1).astype(np.float64)/scale
    for mode in ('velocity_only','action_velocity'):
        x=gap_signals.copy()
        if mode=='velocity_only':x[...,0]=0
        x=np.concatenate((x,np.ones_like(x[...,:1])),-1)
        q=rows['q'].astype(np.float64)+(x*np.asarray(fit['coefficients'][mode])[None]).sum(-1)
        q[:,6:]=np.clip(q[:,6:],initial['native_lower'].numpy()[6:],initial['native_upper'].numpy()[6:])
        assert np.array_equal(q.astype(np.float32),joints[mode])
    assert np.array_equal(joints['stationary'],rows['q']) and np.array_equal(joints['pd_target'],rows['target']) and np.array_equal(joints['oracle'],rows['next_q'])
    geometry=np.load(out/'geometry.npz');tri=geometry['triangle_vertices'];bary=geometry['barycentric']
    assert np.allclose((tri*bary[...,None]).sum(1),geometry['points'],atol=1e-8)
    normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);normal/=np.linalg.norm(normal,axis=-1,keepdims=True)
    assert np.allclose(normal,geometry['normals'],atol=1e-7) and np.all(bary>=0) and np.allclose(bary.sum(1),1)
    source=json.loads((a.prior_source/'data/provenance.json').read_text());record=next(i for i,row in enumerate(source['selection']['inspire_train']) if row['object']=='airplane')
    with np.load(a.prior_source/'data/inspire_train.npz') as raw:
        old_pose=raw['pose'][record];obj=(raw['obj'][record]-old_pose[:3,3])@old_pose[:3,:3];norm=raw['normal'][record]@old_pose[:3,:3]
        assert np.array_equal(obj.astype(np.float32),geometry['object_local']) and np.array_equal(norm.astype(np.float32),geometry['object_normal'])
    # One predeclared first window per held episode:384 independent geometry
    # reconstructions; raw rows, coefficients, predictions and metrics are ALL.
    ids=np.arange(0,len(rows['env']),16);urdf=ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf'
    names=('hand_base_link','thumb_proximal_base','thumb_proximal','thumb_intermediate','thumb_distal','thumb_tip','index_proximal','index_intermediate','index_tip','middle_proximal','middle_intermediate','middle_tip','ring_proximal','ring_intermediate','ring_tip','pinky_proximal','pinky_intermediate','pinky_tip')
    expected_tri,expected_links,expected_bary=independent_samples(urdf,names)
    assert np.array_equal(expected_links,geometry['links']) and np.array_equal(expected_bary,bary)
    assert np.allclose(expected_tri,tri,atol=2e-8,rtol=1e-7), 'URDF sample provenance'
    local=geometry['points'].astype(np.float64);normal=geometry['normals'].astype(np.float64);link=geometry['links'];feature_max=0.;hand_error_max=0.;knn_max=0.;target_max=0.
    def hand_points(q,base):
        links=independent_links(q,urdf);world=np.stack([base@links[name] for name in names],1)
        poses=world[:,link];points=np.einsum('bnij,nj->bni',poses[:,:,:3,:3],local)+poses[:,:,:3,3]
        normals=np.einsum('bnij,nj->bni',poses[:,:,:3,:3],normal);return points,normals
    for begin in range(0,len(ids),16):
        select=ids[begin:begin+16];base=pose(rows['hand_root'][select]);current,hn=hand_points(rows['q'][select],base);actual,_=hand_points(rows['next_q'][select],base)
        current_pose=pose(rows['current_obj'][select]);inverse=current_pose[:,:3,:3]
        obj=geometry['object_local'][None]@inverse.transpose(0,2,1)+current_pose[:,None,:3,3]
        objnormal=geometry['object_normal'][None]@inverse.transpose(0,2,1)
        knn=features['knn_indices'][select];batch=np.arange(len(select))[:,None,None]
        nearest=current[batch,knn];nearest_normals=hn[batch,knn]
        for i in range(len(select)):
            distance,_=cKDTree(current[i]).query(obj[i],k=4);value=np.abs(np.linalg.norm(nearest[i]-obj[i,:,None],axis=-1)-distance).max();knn_max=max(knn_max,float(value));assert value<1e-5
        def object_points(key):
            po=pose(rows[key][select]);return geometry['object_local'][None]@po[:,:3,:3].transpose(0,2,1)+po[:,None,:3,3]
        target=(object_points('next_obj')-obj)@inverse/.01
        target_max=max(target_max,float(np.abs(target-features['target'][select]).max()));assert np.allclose(target,features['target'][select],atol=2e-5,rtol=2e-6)
        relative=np.einsum('b...i,bij->b...j',nearest-obj[:,:,None],inverse)
        norm_obj=np.einsum('bni,bij->bnj',objnormal,inverse);norm_obj/=np.linalg.norm(norm_obj,axis=-1,keepdims=True)
        hnorm=np.einsum('b...i,bij->b...j',nearest_normals,inverse);hnorm/=np.linalg.norm(hnorm,axis=-1,keepdims=True)
        object_local=np.einsum('bni,bij->bnj',obj-current_pose[:,None,:3,3],inverse)
        previous=(obj-object_points('previous_obj'))@inverse/.01
        for mode in joints.files:
            nxt,_=hand_points(joints[mode][select],base)
            flow=np.einsum('b...i,bij->b...j',nxt[batch,knn]-nearest,inverse).mean(2)/.01
            global_flow=((nxt-current).mean(1)[:,None]@inverse)/.01
            x=np.concatenate((object_local/.05,norm_obj,relative.mean(2)/.05,hnorm.mean(2),flow,
                              np.linalg.norm(relative,axis=-1).min(-1)[...,None]/.05,np.broadcast_to(global_flow,obj.shape),previous),-1)
            value=float(np.abs(x-features[mode][select]).max());feature_max=max(feature_max,value);assert np.allclose(x,features[mode][select],atol=2e-5,rtol=2e-6),(mode,value)
            epe=np.linalg.norm(nxt-actual,axis=-1).mean(-1)*1000
            hand_error_max=max(hand_error_max,float(np.abs(epe-errors[mode][select]).max()));assert np.allclose(epe,errors[mode][select],atol=1e-3,rtol=1e-5)
    net_max=0.;metric_max=0.
    for name in ('mano','inspire'):
        path=a.prior_source/'fit'/(name+'_7168.pt');assert sha(path)==r['frozen_prior_sha256'][name]
        state=torch.load(path,map_location='cpu',weights_only=False)['state']
        for mode in joints.files:
            stored=predictions[name+'__'+mode]
            for start in range(0,len(rows['env']),32):
                value=float(np.abs(numpy_predict(state,features[mode][start:start+32])-stored[start:start+32]).max());net_max=max(net_max,value);assert value<2e-4
            per=np.linalg.norm(stored.astype(np.float64)-features['target'].astype(np.float64),axis=-1).mean(-1)*10
            parents=np.asarray([str(env) for env in rows['env']]);groups={env:float(per[parents==env].mean()) for env in sorted(set(parents))}
            value=abs(float(np.mean(list(groups.values())))-r['prior'][name][mode]['parent_epe_mm']);metric_max=max(metric_max,value);assert value<1e-4
            for env,value in groups.items():assert abs(value-r['prior'][name][mode]['per_parent_epe_mm'][env])<1e-4
    for name,prediction in (('zero',np.zeros_like(features['target'])),('persistence',features['stationary'][...,19:22])):
        per=np.linalg.norm(prediction.astype(np.float64)-features['target'].astype(np.float64),axis=-1).mean(-1)*10
        assert abs(float(per.reshape(384,16).mean(1).mean())-r['baselines'][name]['parent_epe_mm'])<1e-4
    for mode in errors.files:
        per=errors[mode].reshape(384,16).mean(1);assert np.allclose(per,r['hand'][mode]['per_episode_epe_mm'],atol=1e-6)
        assert abs(float(per.mean())-r['hand'][mode]['episode_epe_mm'])<1e-6
    bridge=r['hand']['action_velocity']['episode_epe_mm']<=.5*r['hand']['stationary']['episode_epe_mm'] and r['hand']['action_velocity']['episode_epe_mm']<=.75*r['hand']['velocity_only']['episode_epe_mm'] and r['hand']['action_velocity']['episode_epe_mm']<=5
    gates={'execution_bridge':bridge};persistence=r['baselines']['persistence']['parent_epe_mm']
    for name,pred in r['prior'].items():
        oracle=pred['oracle']['parent_epe_mm'];causal=pred['action_velocity']['parent_epe_mm']
        gates[name+'_oracle']=oracle<=.9*persistence and oracle<=.9*pred['stationary']['parent_epe_mm']
        gates[name+'_causal']=causal<=1.1*oracle and causal<=.9*persistence and causal<=.9*pred['velocity_only']['parent_epe_mm']
    assert gates==r['gates'];promising=bridge and any(gates[n+'_causal'] and gates[n+'_oracle'] for n in ('mano','inspire'))
    label='PROMISING' if promising else ('UNCLEAR' if bridge or gates['mano_oracle'] or gates['inspire_oracle'] else 'UNPROMISING');assert label==r['label']
    audit=dict(run_status='COMPLETED',label=label,raw_rows=12288,geometry_rows=384,coefficient_max=coefficient_max,
               feature_max=feature_max,target_max=target_max,hand_error_max_mm=hand_error_max,knn_distance_max_m=knn_max,numpy_forward_max=net_max,metric_max_mm=metric_max,
               all_geometry_rows_not_independently_reconstructed=True,no_new_physics_or_optimizer=True)
    (a.root/'audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit),flush=True)


if __name__=='__main__':main()
