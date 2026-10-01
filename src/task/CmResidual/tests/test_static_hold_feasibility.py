import math
import torch
from src.task.CmResidual.static_hold_feasibility import table_plane_clearance


def test_full_mesh_support_uses_object_rotation_and_table_top():
    vertices=torch.tensor([[x,y,z] for x in (-.1,.1) for y in (-.05,.05) for z in (-.02,.02)])
    obj=torch.zeros(2,13);obj[:,6]=1;obj[:,2]=.25
    # Rotate a rectangular object 90deg around y: its x extent becomes z.
    obj[1,4]=math.sqrt(.5);obj[1,6]=math.sqrt(.5)
    table=torch.zeros(2,13);table[:,6]=1
    clearance=table_plane_clearance(obj,table,vertices,.15)
    torch.testing.assert_close(clearance,torch.tensor([.08,0.]),atol=1e-7,rtol=1e-6)


def test_clearance_is_in_table_local_frame_including_translation():
    vertices=torch.tensor([[x,y,z] for x in (-.05,.05) for y in (-.05,.05) for z in (-.05,.05)])
    obj=torch.zeros(1,13);obj[:,6]=1;obj[:,0]=.7;obj[:,2]=1.2
    table=torch.zeros(1,13);table[:,0]=.5;table[:,4]=math.sqrt(.5);table[:,6]=math.sqrt(.5)
    clearance=table_plane_clearance(obj,table,vertices,.1)
    torch.testing.assert_close(clearance,torch.tensor([.05]),atol=2e-7,rtol=1e-6)
