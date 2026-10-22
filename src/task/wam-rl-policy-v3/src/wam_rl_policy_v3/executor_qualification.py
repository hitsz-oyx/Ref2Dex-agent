"""A formal RL comparison cannot substitute source-PD for the tau executor."""
import json
from pathlib import Path

SCHEMA='ref2dex.physical-executor-qualification.v2'


def require_qualification(path,*,task_contract,executor_checkpoint,policies=None):
    if path is None:raise ValueError('formal RL requires physical source-PD and same-tau executor qualification')
    receipt=json.loads(Path(path).read_text())
    if receipt.get('schema')!=SCHEMA or receipt.get('task_contract')!=task_contract:
        raise ValueError('qualification task contract differs; narrow lift and full RELEASE are distinct')
    if receipt.get('executor_checkpoint')!=str(Path(executor_checkpoint).resolve()):raise ValueError('qualification executor differs')
    if not receipt.get('real_physics') or not receipt.get('independent_success_recomputation'):raise ValueError('actual physical trace and independent scoring required')
    for key,value in (policies or {}).items():
        if receipt.get('policies',{}).get(key)!=value:raise ValueError('qualification execution policy differs: '+key)
    for name in ('source_pd','gt_tau_executor','wam_tau_executor'):
        row=receipt.get('chains',{}).get(name,{})
        n=row.get('attempts',0);successes=row.get('successes',-1)
        if n<1 or not 0<=successes<=n:raise ValueError('separate physical chain evidence required: '+name)
        if name!='wam_tau_executor' and successes/n<.95:raise ValueError('teacher qualification below95%: '+name)
    # A WAM prediction need not already be successful to study its benefit;
    # its observed result must be reported separately, never inherited from GT.
    return receipt


def require_qualified_membership(names,receipt):
    allowed=receipt.get('qualified_sequences',[])
    if not names or not allowed or set(names)-set(allowed):raise ValueError('native initialization scope exceeds physically qualified sequences')
