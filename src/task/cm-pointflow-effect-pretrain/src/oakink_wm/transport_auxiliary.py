"""ContactPose rigid-transport auxiliary; never a future-hand dynamics source."""
import math

HISTORY_INPUTS = ('xyz','features','point_valid','object_valid','scene_object','object_features')


def history_inputs(batch):
    """Only current/history geometry reaches the auxiliary forward pass.

    Omit future hands, their validity, effects, quality and all audit metadata.
    Targets remain in the separate loss batch.
    """
    if any(key not in batch for key in HISTORY_INPUTS):
        raise ValueError('complete current/history geometry required for transport auxiliary')
    return {key:batch[key] for key in HISTORY_INPUTS}


def validate_transport_source(meta, descriptor):
    if (descriptor.get('name') != 'contactpose' or descriptor.get('kind') != 'native'
            or meta.get('source') != 'contactpose'
            or meta.get('schema') != 'ref2dex.native-wm30.v1'
            or meta.get('status') != 'COMPLETED' or meta.get('training_allowed') is not True
            or (meta.get('fps'),meta.get('history'),meta.get('horizon'),meta.get('units')) != (30,4,24,'m')
            or meta.get('hand_order') != ['right','left']
            or meta.get('hand_label') != 'fixed_articulation_native_21_joints_with_per_frame_rigid_transforms'
            or meta.get('supervision_scope') != 'rigid_grasp_transport; no dynamic finger or time-varying contact GT'):
        raise ValueError('explicit native ContactPose rigid-transport provenance required')


def validate_auxiliary_config(config):
    for key, lo, hi in (('updates',1,500),('main_batch',1,64),('aux_batch',1,32),
                        ('seconds',1,1800),('validation_microbatch',1,8)):
        value = config.get(key)
        if isinstance(value,bool) or not isinstance(value,int) or not lo <= value <= hi:
            raise ValueError('bounded auxiliary configuration required: '+key)
    for key, lo, hi in (('auxiliary_weight',0.,.2),('learning_rate',0.,1e-4)):
        value = config.get(key)
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not lo < value <= hi:
            raise ValueError('invalid auxiliary configuration: '+key)
    if not isinstance(config.get('seed'),int) or not 100 <= config['seed'] <= 299:
        raise ValueError('registered Probe seed required')
    if config.get('aux_validation_samples') != 64:
        raise ValueError('fixed64 subject-held-out transport validation windows required')


def classify_auxiliary(control, auxiliary):
    """Predeclared single-seed screen, not a scientific validation conclusion."""
    if auxiliary['main_macro_mm'] > control['main_macro_mm']*1.05:
        return 'UNPROMISING'
    if (auxiliary['main_macro_mm'] <= control['main_macro_mm']*.97
            and auxiliary['oakink2_mm'] <= control['oakink2_mm']*1.05
            and auxiliary['transport_mm'] <= control['transport_mm']*.90):
        return 'PROMISING'
    return 'UNCLEAR'


def initial_primary_deltas(first, second):
    """Check matching metric initialization without bitwise CUDA acos equality.

    Fixed inputs/RNG/model hashes are exact. Primary EPE roundoff may differ
    by at most one micrometre, far below the predeclared 3% Probe threshold.
    Ancillary rotation errors near acos(1) do not define matching eligibility.
    """
    delta = {key:abs(first[key]-second[key])
             for key in ('main_macro_mm','oakink2_mm','transport_mm')}
    for source in ('oakink2','grab','arctic'):
        key='model/anchor/cat0/h24/point_epe'
        delta[source+'_h24_mm']=abs(first['main_metrics'][source][key]-second['main_metrics'][source][key])*1000
    if any(not math.isfinite(v) or v > .001 for v in delta.values()):
        raise ValueError('matched initial primary EPE differs by more than1micrometre')
    return delta
