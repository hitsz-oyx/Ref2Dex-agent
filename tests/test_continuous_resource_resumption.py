import pytest
from scripts.resume_continuous_critic_after_resource_failure import remaining_order,verify_gpu,GPU_INDEX,GPU_UUID,PRIOR_SECONDS

def test_resource_resume_does_not_repeat_valid_native_or_optimizer_prefix():
    order=remaining_order()
    assert order[0]==('collect',3,550)
    assert [s for k,u,s in order if k=='collect']==list(range(550,567))
    assert [u for k,u,s in order if k=='update']==list(range(4,21))
    assert [(u,s) for k,u,s in order if k=='evaluate']==[(20,568),(20,569)]
    assert order[-1]==('analyze',20,None)
    assert PRIOR_SECONDS>=400+167.81858176924288+31.031380785629153

def test_migration_never_silently_uses_original_busy_or_wrong_gpu():
    verify_gpu(dict(index=GPU_INDEX,uuid=GPU_UUID))
    for gpu in [dict(index=1,uuid=GPU_UUID),dict(index=GPU_INDEX,uuid='GPU-33a9c1c9-cccb-61c2-afeb-4ec98be09963')]:
        with pytest.raises(ValueError):verify_gpu(gpu)
