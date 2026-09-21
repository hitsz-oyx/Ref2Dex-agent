from rl_games.algos_torch.models import ModelA2CContinuousLogStd


class ModelDexploreContinuous(ModelA2CContinuousLogStd):
    def __init__(self, network):
        super().__init__(network)

    def build(self, config):
        # rl_games BaseModelNetwork now requires normalization and shape
        # options supplied by BaseModel.build.
        return super().build(config)

    class Network(ModelA2CContinuousLogStd.Network):
        def __init__(self, a2c_network, **kwargs):
            super().__init__(a2c_network, **kwargs)

        def forward(self, input_dict):
            return super().forward(input_dict)
