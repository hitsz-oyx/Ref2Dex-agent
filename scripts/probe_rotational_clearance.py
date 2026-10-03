"""Full retained trajectories; batched GPU plane clearance, no physics/NN fitting."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.task.CmResidual.static_hold_feasibility import mesh_vertices
from src.task.CmResidual.tabletop_clearance import clearance,tabletop_geometry
from src.task.CmResidual.surface_execution import episode_split
from src.task.CmResidual.rotational_clearance import report,classify


def main():
    p=argparse.ArgumentParser();p.add_argument('--execution-source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert torch.cuda.is_available() and ROOT in a.output.resolve().parents and not a.output.exists()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;device='cuda:0'
    parent=json.loads((a.execution_source/'run_manifest.json').read_text());source=Path(parent['source_native'])/'s655'
    initial=torch.load(source/'initial.pt',map_location='cpu',weights_only=False);trace=torch.load(source/'trace.pt',map_location='cpu',weights_only=False)
    assert trace['object_root'].shape==(202,768,13) and torch.equal(trace['progress'],torch.arange(1,203)[:,None].expand(202,768))
    assets=ROOT/'third_party/DExplore/dexplore/data/assets/mjcf/objects'
    vertices=mesh_vertices(assets/'airplane/airplane.obj').double().to(device);table_vertices=mesh_vertices(assets/'table/table.obj').double().to(device)
    assert len(vertices)==initial['object_vertex_count']==25002
    history=torch.cat((initial['object_root'][None],trace['object_root']),0)
    flat=history.reshape(-1,13);environments=np.tile(np.arange(768),203);table=initial['table_root'].double().to(device)
    values=[]
    with torch.no_grad():
        for start in range(0,len(flat),256):
            root=flat[start:start+256].double().to(device);tab=table[torch.from_numpy(environments[start:start+256]).to(device)]
            normal,top,axis=tabletop_geometry(table_vertices,tab);assert axis==1
            values.append(clearance(root,tab,vertices,table_vertices).cpu().numpy())
            if (start//256+1)%192==0:print(json.dumps(dict(poses=min(start+256,len(flat)),total=len(flat))),flush=True)
    clear=np.concatenate(values).reshape(203,768)
    discrepancy=float(np.abs(clear[1:]-trace['clearance'].numpy()).max());assert discrepancy<2e-6,discrepancy
    pre=history[:-1];post=history[1:]
    target=torch.cat((post[...,:3]-pre[...,:3],post[...,7:10]-pre[...,7:10]),-1)/torch.tensor([.005]*3+[.05]*3)
    assert torch.equal(target,trace['physical_transition'])
    for sl in (slice(0,3),slice(7,10)):
        assert torch.equal(pre[...,sl],trace['context'][...,36:49][...,sl])
    dz=(post[...,2].double()-pre[...,2].double()).numpy()
    current=clear[:-1];following=clear[1:];rotation=following-current-dz;translation=current+dz;rotation_only=following-dz
    rise=(pre[...,2].double()-initial['initial_height'].double()).numpy();next_rise=(post[...,2].double()-initial['initial_height'].double()).numpy()
    eligible=(rise>=.03)&(current>=.015)&(current<=.025)
    unresolved=(np.abs(following-.02)<=1e-6)|(np.abs(translation-.02)<=1e-6)
    flip=((following>=.02)!=(translation>=.02))&~unresolved
    q=history[...,3:7].double().numpy();q/=np.linalg.norm(q,axis=-1,keepdims=True)
    angle=2*np.arccos(np.clip(np.abs((q[:-1]*q[1:]).sum(-1)),0,1))
    train,held=episode_split(initial['motion'].numpy(),initial['arm_assignment'].numpy());held_flags=np.isin(np.arange(768),held)
    tick=np.broadcast_to(np.arange(202)[:,None],(202,768));env=np.broadcast_to(np.arange(768)[None],(202,768))
    motion=initial['motion'].numpy();arms=initial['arm_assignment'].numpy();stop=initial['phase_stop'].numpy()[motion];lift=initial['lift_start'].numpy()[motion]
    progress=tick+1;phase=np.where(progress<lift[None],0,np.where(progress<stop[None]-74,1,np.where(progress<=stop[None],2,3)))
    arrays=dict(env=env.ravel(),tick=tick.ravel(),held=np.broadcast_to(held_flags[None],(202,768)).ravel(),
                motion=np.broadcast_to(motion[None],(202,768)).ravel(),arm=np.broadcast_to(arms[None],(202,768)).ravel(),phase=phase.ravel(),
                current_clearance_m=current.ravel(),next_clearance_m=following.ravel(),translation_oracle_clearance_m=translation.ravel(),
                rotation_only_clearance_m=rotation_only.ravel(),translation_delta_m=dz.ravel(),rotation_delta_m=rotation.ravel(),
                current_rise_m=rise.ravel(),next_rise_m=next_rise.ravel(),rotation_angle_rad=angle.ravel(),eligible=eligible.ravel(),unresolved=unresolved.ravel(),resolved_flip=flip.ravel())
    summaries={g:report(arrays,arrays['eligible']&m) for g,m in [('all',np.ones(155136,bool)),('train',~arrays['held']),('held',arrays['held'])]}
    diagnostics={}
    for name,key,count in [('motion','motion',3),('arm','arm',4),('phase','phase',4)]:
        for i in range(count):diagnostics[name+'_'+str(i)]=report(arrays,arrays['eligible']&arrays['held']&(arrays[key]==i))
    mask=(progress>=stop[None]-74)&(progress<=stop[None]+30);assert (mask.sum(0)==105).all()
    actual=np.all(np.where(mask,(following>=.02)&(next_rise>=.03),True),axis=0)
    linear=np.all(np.where(mask,(translation>=.02)&(next_rise>=.03),True),axis=0)
    task={g:dict(episodes=int(m.sum()),actual_success=int(actual[m].sum()),perstep_translation_oracle_success=int(linear[m].sum()),discordant=int((actual[m]!=linear[m]).sum())) for g,m in [('all',np.ones(768,bool)),('train',~held_flags),('held',held_flags)]}
    gates,label=classify(summaries['held']);a.output.mkdir();np.savez(a.output/'rows.npz',**arrays);np.savez(a.output/'pose_clearance.npz',clearance=clear)
    result=dict(run_status='COMPLETED',label=label,gates=gates,summaries=summaries,diagnostics=diagnostics,task=task,
                poses=155904,transitions=155136,mesh_vertices=25002,source_trace_clearance_max_m=discrepancy,
                new_optimizer_updates=0,new_native_ticks=0,oracle_uses_future_translation=True,observed_rotation_not_causal_intervention=True,
                sparse_prior_probe_windows_not_reused_as_entire_task=True,current_pointflow_already_has_rotation=True)
    (a.output/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
