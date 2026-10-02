"""Same teacher comparison with the P0 holding goal aligned in native reference features."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def build_source():
    from scripts.run_self_trained_teacher_environment import build_source as original
    s=original()
    old='            ids=torch.arange(192,device=device);obs=self.env_reset(ids);self.get_batch_size(obs[\'obs\'],1)'
    new='''            original_native_reference=task.hoi_data.detach().cpu().clone()
            original_reference_q=task.hoi_refs[:,0,:,119:137].detach().cpu().clone()
            task.hoi_data=task.hoi_data.clone()
            for mo in range(3):
                stop=int(stops[mo]);task.hoi_data[mo,stop+1:]=task.hoi_data[mo,stop:stop+1].clone()
            if not torch.equal(task.hoi_data[...,-36:-18],torch.stack([task.hoi_refs[mo,0,torch.minimum(torch.arange(task.hoi_data.shape[1],device=device),stops[mo]),119:137] for mo in range(3)])):
                raise ValueError('expert reference joint goal differs from P0 stop clamp')
            torch.save(dict(before=original_native_reference,after=task.hoi_data.detach().cpu(),original_reference_q=original_reference_q,stops=stops.cpu(),semantics='in-memory reference features only; unchanged native progress/physics/files; same P0 stop-clamped hold goal'),args.run_dir/'reference_goal_contract.pt')
            ids=torch.arange(192,device=device);obs=self.env_reset(ids);self.get_batch_size(obs['obs'],1)'''
    if s.count(old)!=1:raise ValueError('reference alignment marker drift')
    s=s.replace(old,new)
    s=s.replace('result=dict(deterministic=args.deterministic,',"result=dict(reference_goal_aligned=True,reference_goal_contract_sha256=sha(args.run_dir/'reference_goal_contract.pt'),deterministic=args.deterministic,")
    return s

def main():
    exec(compile(build_source(),str(ROOT/'scripts/run_continuous_critic_environment_v2.py'),'exec'),{'__name__':'__main__','__file__':str(ROOT/'scripts/run_continuous_critic_environment_v2.py')})
if __name__=='__main__':main()
