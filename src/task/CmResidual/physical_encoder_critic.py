"""Reuse a physical encoder as a trainable measured-return critic initializer."""
from src.task.CmResidual.option_model_policy import initialized_network

SCHEMA='physical-encoder-measured-return-task-Q-v1'
ENCODER_KEYS=('0.weight','0.bias','2.weight','2.bias')


def transferred_critic(physical_parameters):
    critic=initialized_network('value')
    params=critic.state_dict()
    for key in ENCODER_KEYS:
        params[key]=physical_parameters[key].clone()
    critic.load_state_dict(params)
    assert critic[4].out_features==1
    return critic
