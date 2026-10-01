import math
import torch
from pathlib import Path
from src.task.CmResidual.static_hold_feasibility import mesh_vertices
from src.task.CmResidual.tabletop_clearance import tabletop_geometry,clearance


def test_actual_table_mesh_thin_axis_and_native_pose_face_world_up():
    root=Path(__file__).resolve().parents[4]
    table_vertices=mesh_vertices(root/'third_party/DExplore/dexplore/data/assets/mjcf/objects/table/table.obj')
    table=torch.zeros(1,13);table[0,4]=math.sqrt(.5);table[0,5]=-math.sqrt(.5)
    normal,top,axis=tabletop_geometry(table_vertices,table)
    assert axis==1 and normal.tolist()==[[0.,-1.,0.]]
    obj=torch.zeros(1,13);obj[0,6]=1;obj[0,2]=.2
    vertices=torch.tensor([[x,y,z] for x in (-.1,.1) for y in (-.05,.05) for z in (-.02,.02)])
    result=clearance(obj,table,vertices,table_vertices)
    torch.testing.assert_close(result,.18-top,atol=1e-7,rtol=1e-6)


def test_table_translation_changes_world_support_height():
    vertices=torch.tensor([[x,y,z] for x in (-.1,.1) for y in (-.002,.002) for z in (-.2,.2)])
    table=torch.zeros(1,13);table[0,4]=math.sqrt(.5);table[0,5]=-math.sqrt(.5);table[0,2]=.7
    obj=torch.zeros(1,13);obj[0,6]=1;obj[0,2]=.8
    value=clearance(obj,table,torch.tensor([[0.,0.,-.02],[0.,0.,.02]]),vertices)
    torch.testing.assert_close(value,torch.tensor([.078]),atol=1e-7,rtol=1e-6)
