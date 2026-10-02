"""Unchanged scalar physics/P0 checks plus independent holding-reference alignment."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def build_source():
    from scripts.audit_self_trained_teacher_panel import build_source as original
    s=original().replace('(587,588)','(589,590)')
    old='        initial=torch.load(directory/\'initial.pt\',map_location=\'cpu\',weights_only=False);data=torch.load(directory/\'trace.pt\',map_location=\'cpu\',weights_only=False)'
    new=old+'''
        contract=directory/'reference_goal_contract.pt'
        if not r['reference_goal_aligned'] or sha(contract)!=r['reference_goal_contract_sha256']:raise ValueError('goal contract drift')
        reference=torch.load(contract,map_location='cpu',weights_only=False)
        expected=reference['before'].clone();assert torch.equal(reference['stops'],initial['phase_stop'])
        for mo in range(3):
            stop=int(reference['stops'][mo]);expected[mo,stop+1:]=expected[mo,stop:stop+1].clone()
            ts=torch.minimum(torch.arange(expected.shape[1]),reference['stops'][mo])
            if not torch.equal(reference['after'][mo,:,-36:-18],reference['original_reference_q'][mo,ts]):raise ValueError('actual P0 and native expert joint plans differ')
        if not torch.equal(expected,reference['after']):raise ValueError('only goal suffix may change')
'''
    if s.count(old)!=1:raise ValueError('independent reference goal marker drift')
    s=s.replace(old,new).replace('expert_replay_separate_gpu=True,','reference_goal_contract_independently_rebuilt=True,expert_replay_separate_gpu=True,')
    return s

def main():
    exec(compile(build_source(),str(ROOT/'scripts/audit_continuous_critic_panel.py'),'exec'),{'__name__':'__main__','__file__':str(ROOT/'scripts/audit_continuous_critic_panel.py')})
if __name__=='__main__':main()
