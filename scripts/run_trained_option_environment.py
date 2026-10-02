"""Deploy three trained option actors; no physical model/value at inference."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def build_source():
    from scripts.run_statistical_option_environment import build_source as original
    source=original()
    changes={
        '    from isaacgym import gymapi,gymtorch':
        '    from src.task.CmResidual.option_model_policy import SCHEMA as OPTION_SCHEMA,VARIANTS,initialized_network,support_extra\n    import numpy as np\n    from isaacgym import gymapi,gymtorch',
        '    args,remaining=parser.parse_known_args()':
        "    parser.add_argument('--option-actors',type=Path,required=True);parser.add_argument('--option-actors-sha256',required=True)\n    args,remaining=parser.parse_known_args()",
        'statistical_option_collection=True':'trained_option_policies=True,no_model_selector_calls=True,option_actor_deterministic=True,option_actor_sha256=args.option_actors_sha256',
        'option_sampling_std=1.':'option_sampling_std=0.',
    }
    for old,new in changes.items():
        assert source.count(old)==1,old
        source=source.replace(old,new)
    marker='            trace={key:[]'
    assert source.count(marker)==1
    block='''            if sha(args.option_actors)!=args.option_actors_sha256:raise ValueError('trained option actors changed')
            option_bundle=torch.load(args.option_actors,map_location='cpu',weights_only=False)
            if option_bundle['schema']!=OPTION_SCHEMA or option_bundle['actor_updates_each']!=1000 or option_bundle['source_policy_sha256']!=args.policy_sha256 or option_bundle['deploy_model_or_future_input']:raise ValueError('frozen offline actor provenance')
            option_actors={name:initialized_network('actor').to(device).eval() for name in VARIANTS}
            for name in VARIANTS:option_actors[name].load_state_dict(option_bundle['actors'][name])
            option_fingerprints={name:fingerprint(actor.state_dict()) for name,actor in option_actors.items()}
            option_observations=torch.full((768,152),float('nan'))
            option_raw12.zero_()
'''
    source=source.replace(marker,block+marker)
    marker='                for group in (1,2,3):'
    assert source.count(marker)==1
    block='''                if at_decision.any():
                    if tick<2:raise ValueError('two observed physics steps required')
                    root_np=task._target_states.cpu().numpy();older_np=trace['object_root'][-2].numpy()
                    forces_np=torch.cat((task._tar_contact_forces[:,None],task._contact_forces[:,task._contact_body_ids]),1).cpu().numpy()
                    weight_np=np.array([p['mass'] for p in body_properties])*np.linalg.norm([gravity.x,gravity.y,gravity.z])
                    extra=support_extra(root_np,older_np,task._rigid_body_pos[:,task._contact_body_ids].cpu().numpy(),trace['hand_body_position'][-2].numpy(),task._rigid_body_rot[:,task._contact_body_ids].cpu().numpy(),forces_np,weight_np,trace['clearance'][-1].numpy()-trace['clearance'][-2].numpy())
                    compact=np.concatenate((normalized_context.cpu().numpy(),np.ones((768,1),np.float32),((stops[motion]+30-tick).float()/202).cpu().numpy()[:,None]),-1)
                    observation=np.concatenate((compact,np.clip((extra-option_bundle['extra_mean'])/option_bundle['extra_std'],-10,10)),-1).astype(np.float32)
                    inputs=torch.from_numpy(observation).to(device)
                    option_observations[at_decision.cpu()]=inputs[at_decision].cpu()
                    for group,name in enumerate(VARIANTS,1):
                        selected=at_decision&(assignment==group)
                        if selected.any():option_raw12[selected]=option_actors[name](inputs[selected])
'''
    source=source.replace(marker,block+marker)
    marker='            result=dict('
    assert source.count(marker)==1
    block='''            if not torch.isfinite(option_observations).all() or option_raw12.abs().max()>1+1e-6:raise ValueError('all causal decisions recorded')
            for name,actor in option_actors.items():
                if fingerprint(actor.state_dict())!=option_fingerprints[name]:raise ValueError('deployed actor updated')
            if sha(args.option_actors)!=args.option_actors_sha256:raise ValueError('actor checkpoint changed during evaluation')
            torch.save(dict(observation152=option_observations,raw12=option_raw12.cpu(),actor_sha256=args.option_actors_sha256),args.run_dir/'option_decisions.pt')
'''
    source=source.replace(marker,block+marker)
    return source


def main():
    exec(compile(build_source(),str(ROOT/'scripts/run_continuous_critic_environment_v2.py'),'exec'),
         {'__name__':'__main__','__file__':str(ROOT/'scripts/run_continuous_critic_environment_v2.py')})


if __name__=='__main__':
    import sys
    sys.path.insert(0,str(ROOT))
    main()
