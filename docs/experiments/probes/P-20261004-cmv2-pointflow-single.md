# P-20261004-cmv2-pointflow-single

## Question

Does the original `ObjectInteractionCmv2` spatial point-stream encoder provide a useful one-step E-only signal on held-out object-interaction sequences, before adding an action chunk?

## Design

- Data: read-only GRAB/MANO Cmv2 v114d index with v114c lean geometry.
- Split: eight training sequences and two held-out validation sequences, selected by sequence rather than by transition.
- Inputs: current object points/normals (1024), bilateral MANO points/normals (4096), and current-to-next hand flow.
- Target: next-frame object point flow plus rigid translation/rotation labels. No future object geometry, object flow, or contact labels are provided as inputs.
- Model: current V1.3 swept local interaction, contact tokens, cross-attention, and rigid SE(3) flow head; width 64, 8 tokens, kNN 16, no residual head.
- Probe budget: 256 train transitions, 64 held-out transitions, four epochs, one GPU.

The threshold was fixed before the run: continue to a chunk probe only if held-out point-flow EPE improves over zero flow by at least 10 percent.

## Result

Run artifact: `tmp/P-20261004-cmv2-pointflow-single.json`.

| Metric | Zero flow | Point-stream model |
|---|---:|---:|
| Point-flow EPE | 5.764 mm | 4.830 mm |
| Root translation error | 38.256 mm | 36.659 mm |

Point-flow EPE reduction is 16.2 percent. This is a `PROMISING` feasibility signal for the spatial E-only route, not a Gate 2 or policy-utility conclusion. The model was trained only on the short probe subset and has one seed.

## Decision

Proceed to one matched chunk extension using the same point encoder and sequence split. Keep the single-step result as the control, and do not compare it directly with the action/value predictor until a common geometry-bearing Gate 2 data contract exists.
