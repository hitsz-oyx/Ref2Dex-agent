"""Independent actual actor observation/forward and full native execution replay."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def build_source():
    from scripts.audit_statistical_option_panel import build_source as original
    source=original().replace('a.panel not in (603,604)','a.panel not in (611,612)')
    old="options=torch.randn((192,12),generator=g).numpy()";assert source.count(old)==1;source=source.replace(old,'options=np.zeros((192,12),np.float32)')
    old="r['statistical_option_collection']";assert source.count(old)==1;source=source.replace(old,"r['trained_option_policies']")
    source=source.replace("r['option_sampling_std']!=1.","r['option_sampling_std']!=0.")
    marker='        active=(np.arange(202)'
    assert source.count(marker)==1
    block='''        from scripts.audit_option_feature_contract import independent_points
        from src.task.CmResidual.option_model_policy import VARIANTS,numpy_forward,SCHEMA as OPTION_SCHEMA
        actor_file=root/'fit/actors.pt'
        if sha(actor_file)!=m['trained_actor_sha256'] or r['option_actor_sha256']!=m['trained_actor_sha256'] or not r['no_model_selector_calls']:raise ValueError('trained actor identity/inference contract')
        actor_bundle=torch.load(actor_file,map_location='cpu',weights_only=False)
        if actor_bundle['schema']!=OPTION_SCHEMA or actor_bundle['actor_updates_each']!=1000 or actor_bundle['deploy_model_or_future_input']:raise ValueError('actual trained actor provenance')
        decision=torch.load(directory/'option_decisions.pt',map_location='cpu',weights_only=False)
        current,extra=independent_points(initial,data,physical,base_checkpoint,steps)
        observation=np.concatenate((current,np.clip((extra-actor_bundle['extra_mean'])/actor_bundle['extra_std'],-10,10)),-1).astype(np.float32)
        actor_input_error=float(np.abs(observation-decision['observation152'].numpy()).max())
        if actor_input_error>2e-5:raise ValueError(('actual PRE actor observation',actor_input_error))
        actor_forward_error=0.
        if np.any(decision['raw12'].numpy()[arm==0]) or decision['actor_sha256']!=m['trained_actor_sha256']:raise ValueError('P0/actor identity')
        for group,name in enumerate(VARIANTS,1):
            mask=arm==group;predicted=numpy_forward(actor_bundle['actors'][name],decision['observation152'].numpy()[mask],'tanh')
            actor_forward_error=max(actor_forward_error,float(np.abs(predicted-decision['raw12'].numpy()[mask]).max()))
        if actor_forward_error>2e-5:raise ValueError(('all trained actor NumPy forwards',actor_forward_error))
        option_raw=decision['raw12'].numpy()
        if np.max(np.abs(option_raw))>1+1e-6:raise ValueError('bounded trained actor means')
'''
    source=source.replace(marker,block+marker)
    marker='report=dict(paired_option_draw_initialization_and_execution_verified=True,'
    assert source.count(marker)==1
    source=source.replace(marker,'report=dict(all_actual_actor_observations_rebuilt=True,actor_observation_maximum=actor_input_error,all_actor_numpy_forward_maximum=actor_forward_error,no_model_at_inference=True,')
    return source


def main():
    exec(compile(build_source(),str(ROOT/'scripts/audit_truth_successor_native_panel.py'),'exec'),
         {'__name__':'__main__','__file__':str(ROOT/'scripts/audit_truth_successor_native_panel.py')})


if __name__=='__main__':
    import sys
    sys.path.insert(0,str(ROOT))
    main()
