import copy,json
from pathlib import Path
import pytest
from wam_rl_policy_v3.executor_qualification import require_qualification,SCHEMA


def test_source_pd_alone_cannot_certify_the_gt_tau_executor(tmp_path):
    checkpoint=tmp_path/'bc.pt';receipt=dict(schema=SCHEMA,task_contract='lift8cm',executor_checkpoint=str(checkpoint.resolve()),
        real_physics=True,independent_success_recomputation=True,chains={k:dict(attempts=1,successes=1) for k in ('source_pd','gt_tau_executor','wam_tau_executor')})
    receipt['chains']['wam_tau_executor']['successes']=0
    file=tmp_path/'qualification.json';file.write_text(json.dumps(receipt))
    require_qualification(file,task_contract='lift8cm',executor_checkpoint=checkpoint)
    receipt['chains']['gt_tau_executor']['successes']=0;file.write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='gt_tau_executor'):require_qualification(file,task_contract='lift8cm',executor_checkpoint=checkpoint)
    with pytest.raises(ValueError,match='task contract'):require_qualification(file,task_contract='full-release',executor_checkpoint=checkpoint)
    with pytest.raises(ValueError,match='formal RL requires'):require_qualification(None,task_contract='lift8cm',executor_checkpoint=checkpoint)


def test_single_sequence_witness_does_not_open_an_unseen_native_scope():
    from wam_rl_policy_v3.executor_qualification import require_qualified_membership
    receipt=dict(qualified_sequences=['seen'])
    require_qualified_membership(['seen'],receipt)
    with pytest.raises(ValueError,match='scope exceeds'):require_qualified_membership(['seen','unseen'],receipt)
