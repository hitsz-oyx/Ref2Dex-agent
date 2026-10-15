"""Admit a matched shared7k weight-only continuation with a new current-input branch."""
from pathlib import Path


def initialize_from_robot(model,packet,config,stats,source):
    if (packet.get('schema')!=config['schema'] or packet.get('stage')!='robot'
            or packet.get('step')!=7000 or packet.get('stats')!=stats
            or packet.get('provenance',{}).get('engineering_only') is not False
            or Path(packet['provenance']['source']).resolve()!=Path(source).resolve()):
        raise ValueError('Scientific shared7k with same source/stats required')
    previous=packet['config']
    fields=('goal_encoding','task_conditioning','point_width','shared_width','shared_heads','shared_layers',
            'head_width','head_heads','patch_size','scene_tokens','ptv3','fps','history','object_points','horizon',
            'voxel_m','voxel_origin_m','position_scale_m','velocity_scale_mps','acceleration_scale_mps2',
            'hand_in_ptv3','explicit_hand','horizon_normalization','action_interface','action_representation',
            'anchor_weight','encoder_workspace_admission','rollout_admission')
    if any(previous.get(k)!=config.get(k) for k in fields) or config.get('action_interface')!='shared':
        raise ValueError('Parent shared architecture/data/target contract differs')
    result=model.load_state_dict(packet['model'],strict=False)
    allowed={'retargeter.current_mean','retargeter.current_scale',
             'retargeter.current_encoder.0.weight','retargeter.current_encoder.0.bias',
             'retargeter.current_encoder.2.weight','retargeter.current_encoder.2.bias'}
    if set(result.missing_keys)!=allowed or result.unexpected_keys:
        raise ValueError('Only new current-input branch may be absent from robot parent')
