# P-20260926-cm-gate-followup

## Decision target

Test whether the current Cm checkpoint can improve the frozen object-ID expert route when the contact gate is made less reactive. The predeclared continuation rule was: continue only if matched Cm-on gains at least 5 percentage points of lift success; otherwise freeze this checkpoint for policy-utility work.

This is an exploratory Probe, not a Validation. The route, actor checkpoint, Cm checkpoint, seed, and motion inventory are pinned in [`HF02_CM_GATE_FOLLOWUP_20260926.json`](../../handoffs/HF02_CM_GATE_FOLLOWUP_20260926.json).

## Frozen substrate

- Route: six experts over 59 filtered lift motions, simulator-object-ID routing. Route SHA256: `637d0e7cfef1ea2b6578bbff468afefc461e98e82f20bdfdf56bc9f892319e9a`.
- Motion spec SHA256: `8ce7b92b2551af02e928a68c424a806e54722f0d588c65b4321a2e8b8da5c3b4`.
- Actor checkpoint SHA256: `16fd261b4b2de4cbdb257b09f1c7b363b384153103901ff831c825cf47d6a78f`.
- Cm checkpoint SHA256: `396f5e0c92ecf67b2a096568c1a538b7c86d40779a594729f786df6c64ed204a`.
- Each pair used the same seed, 64 environments, `--cm-candidate-mode experts`, and `--disable-early-termination`.

## Results

| gate | seed | Cm-off | Cm-on | delta |
|---|---:|---:|---:|---:|
| instant | 250 | 4/64 (6.25%) | 2/64 (3.125%) | -3.125 pp |
| stable | 251 | 2/64 (3.125%) | 3/64 (4.6875%) | +1.5625 pp |
| stable | 252 | 3/64 (4.6875%) | 5/64 (7.8125%) | +3.125 pp |
| stable pooled | 251–252 | 5/128 (3.90625%) | 8/128 (6.25%) | +2.34375 pp |

The instant pair lost three matched successes and recovered one. The stable pairs show a weak exploratory positive direction, but the pooled gain is below the 5 pp continuation gate and the absolute success rate remains low. Classification: **UNPROMISING for current policy utility; weak positive signal retained as research debt**. No training or coefficient sweep is justified by this Probe.

## 660-motion follow-up blocker

The same evaluator was attempted on the pinned 660-motion route with `--num_envs 64`. Both runs stopped before simulation because `inspire_object_balanced.yaml` leaves `objectMotionSampling` disabled and the task rejects `660 motions exceed 64 environments`. Increasing to 660 environments has an unreviewed memory cost; enabling `objectMotionSampling` changes the fixed per-motion evaluation semantics. The failed manifests are retained:

- `outputs/CmResidual/agent_grab660_cm_stable_probe_off_s253/run_manifest.json` (SHA256 `ed4816b91eea0f19997bea6165594bfaa120599eb7e4a67781dabe9b9f1e7fb3`)
- `outputs/CmResidual/agent_grab660_cm_stable_probe_on_s253/run_manifest.json` (SHA256 `304c686156abac6ce1bbbcd602154d44ee035071c83643c2126ff07f61983429`)

This is a configuration blocker, not a 660-motion scientific result. A future full-pool Probe must choose explicitly between a memory-budgeted `num_envs >= 660` evaluator and an object-sampling evaluator with a different estimand.

## Reproduction

Run from `third_party/DExplore` with the commands recorded in the six run manifests. The two stable matched pairs are:

```bash
env CUDA_VISIBLE_DEVICES=0 $PYTHON dexplore/evaluate_object_router.py \
  --route-config "$ROOT/outputs/CmResidual/grab59_six_expert_route.json" \
  --cm-mode off --cm-candidate-mode experts --cm-contact-gate stable \
  --output "$ROOT/outputs/CmResidual/agent_grab59_cm_stable_probe_off_s251/results.json" \
  ... --seed 251

env CUDA_VISIBLE_DEVICES=1 $PYTHON dexplore/evaluate_object_router.py \
  --route-config "$ROOT/outputs/CmResidual/grab59_six_expert_route.json" \
  --cm-checkpoint "$ROOT/outputs/CmLite/V1.37/balanced_s1x24_s3_e160_e180/best.pt" \
  --cm-sha256 396f5e0c92ecf67b2a096568c1a538b7c86d40779a594729f786df6c64ed204a \
  --cm-mode cm --cm-candidate-mode experts --cm-contact-gate stable \
  --output "$ROOT/outputs/CmResidual/agent_grab59_cm_stable_probe_on_s251/results.json" \
  ... --seed 251
```

The `...` arguments are identical to the corresponding manifests (task, configs, motion root, actor checkpoint, headless mode, and 64 environments); the manifests are the authoritative command records.
