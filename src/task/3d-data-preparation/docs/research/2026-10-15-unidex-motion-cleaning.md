# UniDex simulator motion screening

User requested screening before remote nine-source pretraining. This is data-quality engineering, not evidence that a method works or all remaining simulator transitions are physically correct.

## Decision Note

The known jar trajectory row14909 moves approximately7.4m in0.8s. Raw60Hz captures and compact30Hz agree exactly, and world-coordinate label reconstruction agrees within2.4e-7m. Thus label-frame conversion is not the observed cause; the recording itself contains a sudden contact ejection. Solver mechanism remains unproven.

Scan all33,056 source4 sequences on CPU (array and file diagnostics; no model computation), preserving simulator success/failure labels only for reporting. Threshold selection uses TRAIN diagnostics, never DEV outcome tuning. Use `configs/unidex_motion_v1.json` to isolate suspect motion: joint large translation/speed/acceleration outrunning the hand; more severe shared hand-object impulses; unusually fast initial capture transitions; and extreme rotation. Rotation and initialization flags mean suspect isolation, not physically impossible motion. Initial-flight rule applies only to these reset-start simulator captures. Normal smooth falling is not rejected solely for speed, net displacement, or distance from the hand.

Quarantine every H4+K24 window ending at or after the first flagged transition, including its subsequent trajectory tail; preserve complete earlier windows. Do not clip targets, interpolate bad motion, stitch tails, filter grasp failures, or edit source payloads. Leave other eight sources unchanged. Recompute exact native TRAIN statistics by subtracting only excluded TRAIN windows; reuse unchanged original-three-source sampled moments and recompute nine-source mixture weights and all24 horizon scales.

Budget:4CPU workers, full audit≤20min/300MB; derived metadata≤20min/200MB; free space≥20GiB. No new GPU training. Preserve old views and annotate quarantined rows so the viewer can still display their full raw trajectory. Stop on identity/moment mismatch or budget breach.

Independent review checked first-event window boundaries, quaternion-sign invariance, gravity drops, and identified shared-impulse and already-fast initial-state blind spots; both are covered by explicit additional rules. Remaining false positives/negatives are possible; these thresholds are conservative operational filters, not universal physical laws.

## Artifacts

- Full completed audit: `outputs/3d-data-preparation/unidex-motion-audit-20261015-r2/`
- Derived view: `outputs/3d-data-preparation/native3d-clean-motion-20261015-r1/`
- Derived nine-source recipe: `outputs/ptv3-shared-wam-v3/nine-source-motion-recipe-20261015-r1/`
- Commands: `tools/audit/audit_unidex_motion.py`, `tools/run/prepare_motion_view.py`; v3 `tools/run/refresh_nine_recipe.py`.

The earlier audit r1 was stopped for uncompressed artifact growth and is not used. Audit r2 retains per-frame diagnostics in gzip; event decisions are applied by the separately versioned policy, preserving the scan's identity.

## Completed result

Policy v1 isolates25 captures (TRAIN21/DEV4), excluding114 TRAIN/22 DEV windows. Ten captures have no remaining full windows;15 retain their earlier complete prefix. Four flagged TRAIN captures are marked successful and17 unsuccessful: success is not an admission condition. Source4 retains264,325 TRAIN and66,052 DEV windows. Other eight source window sets are unchanged. Nine-source total is4,806,030 TRAIN/821,613 DEV.

All114 excluded TRAIN windows were independently replayed through `CleanPoseMeshWindows`; adding their moments back to the new statistics reconstructs the parent moments within7.28e-12 maximum absolute difference. The original three-source TRAIN moment sample is unchanged; new mixture weights and24×3 hand/scene statistics are bound to the new frozen view. Known row14909 retains ticks3/11/19/27/35/43/51 and excludes59/67/75.

Important residual cases: independent review identified rows9674/17266 with later fast falling/sliding that do not meet this conservative impulse rule. They remain explicit visual-review candidates, not certified valid; do not turn a few-metres displacement or floor contact into automatic failure. Further physical adjudication would need raw simulator contact/velocity replay. No training was started during cleaning.

## Remote and visualization verification

Remote `.../ptv3-shared-wam-v3-nine-source-20261015-r1/data-motion-r1` passed the full dependency check (37,591 sequence payloads/2,638 CAD),36 mixed TRAIN samples across9 sources and32 mixed DEV samples across8 sources. Numeric horizon arrays equal the new local recipe exactly; no payload/CAD recopied. Only3,240,669bytes of compressed metadata were transferred. Known-bad ticks59/67/75 are absent in the new index and remain readable in the retained raw parent. Evidence: `tmp/nine-remote-20261015-r1/motion-r1-evidence/`. No GPU training was launched.

Viser8769 now uses the new recipe; its real-control check loaded all10 fully quarantined sequences and the original7.387m sample, showing the correct admission state. Old annotations are preserved. A watchlist for uncertain rows9674/17266 is stored as `visual-review-watchlist.json` in the new local native view.
