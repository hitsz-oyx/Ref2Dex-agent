# P-20261004-cmv2-pointflow-chunk-k4

## Question

After the single-step point-stream probe is promising, does temporal aggregation over a short spatial interaction chunk improve longer-horizon object flow prediction?

## Design

- Same GRAB/MANO sequence split and Cmv2 spatial configuration as `P-20261004-cmv2-pointflow-single`.
- Chunk length: four horizons; 128 train chunks and 32 held-out chunks.
- Each horizon uses the current object geometry and cumulative recorded MANO hand flow from the chunk start. Per-horizon spatial fused features are aggregated by a GRU and decoded to a rigid SE(3) flow.
- The matched baseline is the trained single-step Cmv2 checkpoint applied independently to each cumulative hand-flow horizon.
- No future object geometry, object-flow target, or contact label is provided as an input.

The predeclared continuation threshold was a mean held-out EPE improvement of at least 5 percent over the matched single-step baseline.

## Result

Run artifact: `tmp/P-20261004-cmv2-pointflow-chunk-k4.json`.

| Horizon | Single-step baseline | Chunk-GRU |
|---:|---:|---:|
| 1 | 5.03 mm | 7.88 mm |
| 2 | 10.77 mm | 8.26 mm |
| 3 | 17.53 mm | 13.25 mm |
| 4 | 24.38 mm | 19.13 mm |

Mean EPE decreases from 14.42 mm to 12.13 mm (15.9 percent). The chunk model loses accuracy at the first horizon but improves horizons two through four.

## Decision

The short chunk route is `PROMISING` as a spatial point-flow diagnostic. The next useful experiment is a matched action-to-hand-flow interface or geometry-bearing Gate 2 collector, so the point-flow model can be tested without feeding recorded future hand flow directly. This result is not a Gate 2 policy-utility conclusion.
