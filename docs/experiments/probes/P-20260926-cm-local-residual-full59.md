# P-20260926-cm-local-residual-full59

- Classification: Decision Probe.
- Code commit: `4c638e299b5fbb961a7e3f3314b6a1a6f42209ac`.
- Question: after five consecutive hand-object contact steps, can the
  existing Cm checkpoint choose a small local wrist/finger residual that
  improves held-lift on the canonical 59-motion six-expert route?
- Decision rule: continue this exact residual family only if the matched
  Cm-on arm gains a meaningful held-lift margin over Cm-off. A zero delta
  does not justify a coefficient or seed sweep.

## Frozen substrate

- Route: `outputs/CmResidual/grab59_six_expert_route.json`, SHA256
  `637d0e7cfef1ea2b6578bbff468afefc461e98e82f20bdfdf56bc9f892319e9a`.
- Motion spec: SHA256
  `8ce7b92b2551af02e928a68c424a806e54722f0d588c65b4321a2e8b8da5c3b4`.
- Actor checkpoint: `source_e260`, SHA256
  `16fd261b4b2de4cbdb257b09f1c7b363b384153103901ff831c825cf47d6a78f`.
- Cm checkpoint: `outputs/CmLite/V1.37/balanced_s1x24_s3_e160_e180/best.pt`,
  SHA256 `396f5e0c92ecf67b2a096568c1a538b7c86d40779a594729f786df6c64ed204a`.
- Both arms used seed `254`, 64 environments, stable contact gate,
  `--cm-candidate-mode local`, and disabled early termination. The route
  remains simulator-object-ID based; no ten-expert or 660-motion result is
  included here.

## Matched result

| arm | lift success | lift rate | contact fraction | max contact lift | mean reward |
|---|---:|---:|---:|---:|---:|
| Cm-off | 5/64 | 7.8125% | 0.19712 | 0.04564 m | 93.65 |
| Cm-on local residual | 5/64 | 7.8125% | 0.18433 | 0.04337 m | 92.40 |
| difference | 0 | 0 pp | -0.01279 | -0.00227 m | -1.25 |

The Cm-on selector was active for 1,615 of 66,944 scored steps (2.41%); its
selection histogram was `[65329, 257, 766, 405, 187]`. Thus this is a valid
matched implementation run, but it provides no held-lift gain and slightly
reduces contact-support metrics. Classification: **UNPROMISING** for this
local residual and checkpoint. Do not spend another seed or coefficient sweep
on this exact接法.

## Reproduction and artifacts

The authoritative commands, route SHA, seed, GPU, and completion summaries
are in the two manifests:

- `outputs/CmResidual/agent_grab59_cm_local_probe_off_s254_r2/run_manifest.json`
  (SHA256 `055ff49544664e6e15e7878b11e4521e82055d28c6adee11d72cf6d6b9ed1fc6`)
- `outputs/CmResidual/agent_grab59_cm_local_probe_on_s254_r2/run_manifest.json`
  (SHA256 `87ee01f840a2b617bd27f02f58f63112ad5ac6cc0d4b8005a34a1fda3ec0ab7a`)

Result JSON hashes are `01a26be2802a4dda24c49037dec4266cec608410ca802d99f981a1a14d817b17`
(off) and `ff78b43f828f4e31315dd25f5d52aeb4ecc173215a36b456c11d6e0480990c23`
(on). The output directories are ignored runtime artifacts; this card and the
code commit are the versioned provenance.
