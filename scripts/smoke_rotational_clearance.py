"""Explicit hybrid poses verify separable actual thin-table clearance algebra."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.task.CmResidual.static_hold_feasibility import mesh_vertices
from src.task.CmResidual.tabletop_clearance import clearance
from scripts.audit_rotational_clearance import independent_clearance,read_vertices


def main():
    torch.set_num_threads(2);assets=ROOT/'third_party/DExplore/dexplore/data/assets/mjcf/objects'
    vertices=mesh_vertices(assets/'airplane/airplane.obj').double();table_vertices=mesh_vertices(assets/'table/table.obj').double()
    current=torch.zeros(4,13,dtype=torch.float64);current[:,2]=torch.tensor([.02,.03,.04,.05]);current[:,6]=1
    next_pose=current.clone();next_pose[:,2]+=.003;next_pose[:,3:7]=torch.tensor([[0,.1,0,.99498743710662],[.1,0,0,.99498743710662],[0,0,.1,.99498743710662],[0,0,0,1]],dtype=torch.float64)
    table=torch.zeros_like(current);table[:,3]=-2**-.5;table[:,6]=2**-.5
    translation=current.clone();translation[:,:3]=next_pose[:,:3]
    rotation=current.clone();rotation[:,3:7]=next_pose[:,3:7]
    c=clearance(current,table,vertices,table_vertices);n=clearance(next_pose,table,vertices,table_vertices);t=clearance(translation,table,vertices,table_vertices);r=clearance(rotation,table,vertices,table_vertices)
    dz=next_pose[:,2]-current[:,2];error=max(float((t-c-dz).abs().max()),float((r-n+dz).abs().max()),float((n-c-(t-c)-(r-c)).abs().max()));assert error<1e-12
    independent=independent_clearance(next_pose.numpy(),table.numpy(),read_vertices(assets/'airplane/airplane.obj'),read_vertices(assets/'table/table.obj'))
    discrepancy=float(np.abs(independent-n.numpy()).max());assert discrepancy<1e-10
    print(json.dumps(dict(engineering_smoke='PASS',rows=4,explicit_hybrid_algebra_max_m=error,independent_mesh_max_m=discrepancy,cpu_reason='4poses tiny engineering cheaper than GPU startup')),flush=True)


if __name__=='__main__':main()
