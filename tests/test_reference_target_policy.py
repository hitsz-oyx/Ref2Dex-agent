import torch
from src.task.CmResidual.reference_target_policy import target_from_residual,residual_labels

def test_stationary_reference_target_does_not_integrate_previous_command():
    reference=torch.zeros(2,18);reference[:,0]=.5
    residual=torch.zeros(2,18);residual[:,0]=.25;residual[:,6]=.75
    lower=torch.zeros(18);upper=torch.ones(18)*1.6
    goal=target_from_residual(reference,residual,lower,upper)
    torch.testing.assert_close(goal[:,0],torch.ones(2)*.505)
    torch.testing.assert_close(goal[:,6],torch.ones(2)*.30)
    torch.testing.assert_close(goal[:,7],torch.ones(2)*.315)
    current=torch.tensor([.4,.6]);command=goal[:,0]-current
    torch.testing.assert_close(current+command,goal[:,0])
    torch.testing.assert_close(residual_labels(goal,reference)[:,:7],residual[:,:7])
