"""Robot action adapter on top of the pretrained v2 shared WAM."""
import torch
from torch import nn

from . import paths as _paths
from .model import SharedWAMV2
from .objectives import forecast_loss
from ptv3_shared_wam.action_targets import NATIVE, target_tensor
from ptv3_shared_wam.robot import native_trajectory
from ptv3_shared_wam.target_retargeter import make_retargeter


class SharedWAMV2Robot(nn.Module):
    """V2 forecaster plus the existing Chunk-R action head.

    The action head sees measured q_real/dq and the forecast hand trajectory.
    Its labels are `action_target`, which for the selected geometry-lift
    representation are encoded from q_pd.  q_ref and q_real stay in the batch
    as explicit audit fields and never get implicitly swapped into the loss.
    """

    def __init__(self, stats, config, action_representation=NATIVE):
        super().__init__()
        self.forecaster = SharedWAMV2(stats, config)
        self.action_representation = action_representation
        self.action_interface = config.get("action_interface")
        self.anchor_weight = config.get("anchor_weight", 1.)
        if self.action_interface is None:
            self.retargeter = make_retargeter(action_representation)
        else:
            from .shared_actor import SharedQueryActor
            from ptv3_shared_wam.action_targets import GEOMETRY_LIFTED, action_dimension
            if action_representation not in (NATIVE,GEOMETRY_LIFTED) or self.anchor_weight != 0:
                raise ValueError("matched shared actor requires native/geometry-lift and anchor_weight=0")
            self.retargeter = SharedQueryActor(self.action_interface, config['shared_width'],
                action_dim=action_dimension(action_representation),
                current_observation=config.get('current_observation_input'))

    def forward(self, batch):
        prediction = self.forecaster(batch, return_shared=True) if self.action_interface else self.forecaster(batch)
        has_action = batch.get("has_action")
        if has_action is None or not bool(has_action.all()):
            raise ValueError("v2 robot fine-tune requires action labels on every row")
        tau = native_trajectory(prediction["trajectory"], batch["start_hand_valid"])
        if self.action_interface:
            prediction["action"] = self.retargeter(batch["native_state"], tau,
                prediction.pop("shared_tokens"), prediction.pop("shared_valid"),batch.get('native_current_obs'))
        else:
            prediction["action"] = self.retargeter(batch["native_state"], tau)
        # Anchor R to the measured future hand trajectory.  It updates only R;
        # observed future hand labels are detached from the v2 forecaster.
        if self.anchor_weight:
            prediction["action_anchor"] = self.retargeter(
                batch["native_state"], batch["native_tau"].detach())
        return prediction

    def parameter_groups(self, config):
        groups = self.forecaster.parameter_groups(config["geometry_lr"], config["shared_lr"])
        groups.append(dict(name="retargeter", params=list(self.retargeter.parameters()),
                           lr=config.get("robot_lr", config.get("shared_lr", 1e-4))))
        return groups


def robot_loss(prediction, batch, config, model):
    total, terms = forecast_loss(prediction, batch, config, model.forecaster.normalizer)
    target = target_tensor(batch, model.action_representation)
    scale = model.retargeter.action_scale
    action_error = ((prediction["action"] - target) / scale).abs()
    supervise_steps = int(config.get("action_supervision_steps", action_error.shape[1]))
    if not 1 <= supervise_steps <= action_error.shape[1]:
        raise ValueError("action_supervision_steps must be within the predicted horizon")
    # The policy always predicts the complete action chunk.  This mask only
    # changes which query positions train the action head, while forecast loss
    # remains full-horizon for both matched arms.
    action = action_error[:, :supervise_steps].mean()
    anchor = (((prediction["action_anchor"] - target) / scale).abs()[:, :supervise_steps].mean()
              if config.get("anchor_weight", 1.) else action.new_zeros(()))
    total = total + config.get("action_weight", 1.) * action + config.get("anchor_weight", 1.) * anchor
    return total, dict(terms, action=action, anchor=anchor)


__all__ = ["SharedWAMV2Robot", "robot_loss"]
