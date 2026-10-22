"""Current-geometry admission, never synthetic history for WAM inference."""
import numpy as np

DOMAIN_POLICY='initial-geometry-admission.mask-invalid-online-tau.v1'


def current_v4_domain_sample(pose,canonical_hand,prepared_geometry):
    """Only fields read by V4Runtime.workspace_valid; no dynamics are fabricated."""
    surface=prepared_geometry[0];hand=np.zeros((2,11,3),np.float32)
    hand[0]=(np.asarray(canonical_hand)-pose[:3,3])@pose[:3,:3]
    valid=np.zeros((2,11),bool);valid[0]=True
    return dict(xyz=np.asarray(surface,np.float32),features=np.zeros((len(surface),18),np.float32),
        point_valid=np.ones(len(surface),bool),start_hand=hand,start_hand_valid=valid)
