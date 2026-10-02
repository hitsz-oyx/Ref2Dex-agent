"""Actual four-tick Gaussian requests with measured SDK geometry and unchanged P0."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def build_source():
    source=(ROOT/'scripts/run_continuous_critic_environment_v2.py').read_text()
    changes={
        "    from isaacgym import gymapi,gymtorch":"    from src.task.CmResidual.coherent_motor_request import CoherentMotorRequest\n    from isaacgym import gymapi,gymtorch",
        'heads={};head_fingerprints={};generators={}':"heads={};head_fingerprints={};generators={};request_caches={name:CoherentMotorRequest() for name in ('cm','state_only','none')}",
        'physical_metadata=dict(native_dof_names=names,':'physical_metadata=dict(native_body_names=task.gym.get_actor_rigid_body_names(task.envs[0],task.humanoid_handles[0]),contact_body_ids=task._contact_body_ids.cpu().tolist(),hand_body_com=[[p.com.x,p.com.y,p.com.z] for p in task.gym.get_actor_rigid_body_properties(task.envs[0],task.humanoid_handles[0])],native_dof_names=names,',
        "'normalized_context','request_noise')}":"'normalized_context','request_noise','hand_root','hand_body_position','hand_body_quaternion','request_logstd','request_decision_tick','request_is_decision')}",
        '                critic_value=torch.zeros(768,device=device);aux_prediction=torch.zeros((768,6),device=device)':'                request_logstd=torch.zeros_like(request);request_decision_tick=torch.full((768,),-1,device=device,dtype=torch.long);request_is_decision=torch.zeros(768,device=device,dtype=torch.bool)\n                critic_value=torch.zeros(768,device=device);aux_prediction=torch.zeros((768,6),device=device)',
        "                    noise=torch.zeros_like(mu) if args.deterministic else torch.randn(mu.shape,device=device,generator=generators[variant])\n                    raw=mu+logstd.exp()*noise":"                    packet,is_decision=request_caches[variant].choose(tick,mu,logstd,generators[variant])\n                    mu=packet['request_mean'];logstd=packet['request_logstd'];noise=packet['request_noise'];raw=packet['request']",
        '                    request_logprob[selected]=request_log_probability(mu,logstd,raw)':"                    request_logstd[selected]=logstd;request_decision_tick[selected]=packet['decision_tick'];request_is_decision[selected]=is_decision\n                    request_logprob[selected]=request_log_probability(mu,logstd,raw)",
        'normalized_context=normalized_context,request_noise=request_noise)':'normalized_context=normalized_context,request_noise=request_noise,hand_root=task._humanoid_root_states,hand_body_position=task._rigid_body_pos[:,task._contact_body_ids],hand_body_quaternion=task._rigid_body_rot[:,task._contact_body_ids],request_logstd=request_logstd,request_decision_tick=request_decision_tick,request_is_decision=request_is_decision)',
        'result=dict(deterministic=args.deterministic,':'result=dict(coherent_period=4,request_decisions_each=51,full_request_blocks=50,final_partial_block_ticks=2,request_likelihood_semantics="Gaussian at cached decision state",deterministic=args.deterministic,',
        "    args,remaining=parser.parse_known_args()":"    args,remaining=parser.parse_known_args()\n    if args.deterministic:raise ValueError('only frozen stochastic coherent execution contract')",
    }
    for old,new in changes.items():
        if source.count(old)!=1:raise ValueError('exact coherent controller marker drift '+old)
        source=source.replace(old,new)
    return source

def main():
    exec(compile(build_source(),str(ROOT/'scripts/run_continuous_critic_environment_v2.py'),'exec'),{'__name__':'__main__','__file__':str(ROOT/'scripts/run_continuous_critic_environment_v2.py')})

if __name__=='__main__':main()
