import torch
from scripts.fit_observation_hold_aggregation import corrective_labels

def test_corrective_query_uses_visited_wrist_without_future_state():
    lower=torch.zeros(18);upper=torch.ones(18)*2
    initial=dict(motion=torch.zeros(1,dtype=torch.long),phase_stop=torch.tensor([202]),
        native_reference_q=torch.ones(1,203,18)*.4,native_lower=lower,native_upper=upper,
        lift_start=torch.tensor([100]),pd_offset=torch.zeros(18),pd_scale=torch.ones(18)*2)
    context=torch.zeros(202,1,70);context[...,:3]=.4
    first=corrective_labels(initial,dict(context=context))
    context[...,0]+=.02
    second=corrective_labels(initial,dict(context=context))
    torch.testing.assert_close(second[...,0]-first[...,0],torch.full((202,1),-.01))
    torch.testing.assert_close(second[...,1:],first[...,1:])
    assert torch.equal(second[...,[7,9,11,13,16,17]],torch.zeros(202,1,6))
