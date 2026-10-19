"""Version6 requires an explicit planned-event input and independent recipe."""
from ptv3_shared_wam_v5.contracts import HORIZON, C_DIM, LABEL_SEMANTICS
from ptv3_shared_wam_v5.contracts import INPUT_SEMANTICS as BASE_INPUT_SEMANTICS

SCHEMA = 'ref2dex.ptv3-shared-wam.v6'
DATA_SCHEMA = 'ref2dex.wam-v6-event-recipe.v1'
INPUT_SEMANTICS = BASE_INPUT_SEMANTICS+'.next-planned-contact-release.frame30.v1'
MIGRATION_SCHEMA = 'ref2dex.wam-v5-to-v6-explicit-expansion.v1'
