---
schema: ref2dex.validation.v2
validation_id: VAL-20260926-observation-six-expert-c1
date: 2026-09-26
branch: agent/observation-router-reliability
git_commit: 7187a1159d5b65a494dd476eb150be0ecd48b809
claim_id: C1
hypothesis_family: HB01
frozen_method_commit: 32926392c915d2420140675781182e0d52c533f5
development_seed_pool: validation.development
validation_seed_pool: validation.holdout
matched_control: "Frozen simulator_object_id six-expert route, paired by simulator seed, motion and start frame."
status: COMPLETED
---

# Validation: frozen observation-driven six-expert C1 route

## Claim and decision

Test the narrow C1 claim that the frozen initial-observation router preserves the self-trained six-expert choices and repeatedly produces nontrivial held-lift on the specified 12-motion task. A `SUPPORTED` result makes this hierarchy a fixed, nonprivileged routing substrate for the next high-level Cm policy-utility design. A failed expert-choice gate points to the router representation; failed held-lift gates point to the expert substrate or its physical robustness. This does not test whether observation routing beats fixed routing.

The seed233 and seed260 Decision Probes motivated this Validation and are excluded from its holdout matrix. No parameters, thresholds, expert checkpoints or model are selected using holdout outcomes.

## Frozen method and controls

- Holdout simulator seeds **400, 401, 402, 403, 404**, all reserved in `docs/SEED_LEDGER.yaml`; no development runs are needed because the evaluator completed the seed260 Probe unchanged.
- For each seed, run the fixed `simulator_object_id` route and frozen observation route sequentially, 64 environments each, disabled early termination, first completed full episode per environment. Both use the same 12 motions, environment/train configs, source checkpoint, six self-trained expert checkpoints, reward and metric. Cm is off.
- Frozen route config SHA256 `6e3205a8b5fe2be3ab46308678e673d10e62745a7eda8196a7fe0e0d365bb7a8`, evaluator SHA256 `4f4feddb3bbad7d5b0dbd67848f4a9a88fa1923bdfcdeb316d5916640c6b6320`, observation model SHA256 `1fa84c94891d63824be0100b5eb78f4aa1e37825c6d71fff65e517c0c7f2fc14`. The seed260 preflight contains the complete checkpoint, motion and config hashes; copy those exact inputs into the Validation preflight and check them before each seed.
- No checkpoint selection or training. Use the unchanged evaluator at `third_party/DExplore/dexplore/evaluate_object_router.py`. Change only `--seed`, output path and presence of the frozen observation-model arguments between arms.

## Validity contract and primary metrics

All 10 native run manifests must say `COMPLETED` and record the same execution commit, route/input hashes, Cm disabled and the planned command. The evaluator hash must equal the frozen method's hash. Each arm must contain 64 unique environment IDs and finite metrics. Within each seed, motion ID, start frame and episode length must match by environment ID. The observation arm must save exactly 64 initial expert choices. A technical failure, input drift or incomplete arm is `INVALID_IMPLEMENTATION`; do not replace a seed based on its result.

The primary metrics are initial expert-choice agreement with the fixed route and `lift_success` held-lift count, defined by the evaluator as object rise of at least 0.03 m with hand/object contact for at least five consecutive steps. The fixed route is a matched physical coverage reference. Report per-seed counts, object-wise expert confusion, paired lift gain/loss counts and total observation-minus-fixed difference. A seed-level bootstrap with 20,000 draws, RNG seed `20260926`, reports a descriptive 95% percentile interval for the physical difference; it is not a superiority test or a general-population guarantee.

## Predeclared conclusion rule

If all runs are valid, `SUPPORTED` requires **all** of:

1. at least **60/64** initial expert choices match the fixed route in **each** of the five seeds;
2. every cup environment chooses `cup_e340` (30/30 if six cup environments occur per seed);
3. observation-route held-lifts are at least **16/64 in each seed** and **100/320 in total**;
4. total observation held-lifts are no more than **30/320 below** the fixed route (the Probe's -6/64 tolerance aggregated over five seeds).

If the complete valid matrix fails any of these fixed conditions, mark this *joint, task-bound claim* `REFUTED`, identifying whether the router or physical substrate failed. Partial or ambiguous analysis without a valid matrix is `INCONCLUSIVE`; technical failure is `INVALID_IMPLEMENTATION`. Do not tune and rerun holdout seeds after seeing their outcomes.

## Budget and stop conditions

One idle GPU, sequential arms, expected 36 minutes, hard wall budget **60 minutes** for the matrix, new output under **100 MB**. Check GPU ownership before starting and between arms. Stop on occupied GPU, drift from frozen hashes/commands, missing files, incomplete first episodes, nonfinite outputs, evaluator failure, wall/storage limit or any unknown process collision. Keep every native result and a parent manifest, with no video or training output.

## Interpretation boundary

Even a `SUPPORTED` result applies only to this frozen six-expert hierarchy, these 12 motions and the five held-out seeds. It does not establish a single shared GRAB actor, unseen-object generalization, an improvement over fixed routing, or Cm policy utility. Before promoting the Validation result to a formal scientific conclusion, use the `AGENTS.md` Decision Checkpoint and submit a short Decision Memo.

## Result

run_status: `COMPLETED`

conclusion: `SUPPORTED`（仅限冻结六专家观测路由在这 12 条 motion、五个 holdout seed 上的预注册联合门槛）

decision_status: `ACCEPTED`（2026-09-27）

Root 提议 Option A；用户回复“你是主agent有最高权限，按照你的想法来”，将该
Decision Checkpoint 的路线选择交由 root。Root 随后接受窄范围 `SUPPORTED`
结论；决策与边界见
[D-20260927-c1-observation-route-validation.md](../../archive/2026-10-04-research-governance/decisions/D-20260927-c1-observation-route-validation.md)。
下述机器可读结果索引是在此决策**之前**生成的执行证据，其中
`promotion_status: PENDING_DECISION_CHECKPOINT` 和
`proposed_terminal_label: SUPPORTED` 是当时的历史状态；为保留原始分析指纹，
不修改索引，也不以该字段代表当前决策状态。

### Frozen-matrix evidence

The parent manifest and all ten native arms completed on execution commit
`3697ddd2b37339990c97aaa025bd37924a673c3c`. The preflight input contract,
route/evaluator/router hashes, checkpoint and motion hashes, planned commands,
Cm-off settings, unique 64-environment episode sets, finite metrics and all
fixed/observation episode pairings passed. The matrix used GPU 1 for 1,553.770
seconds (25.90 minutes) and produced 460,588 bytes of native output, below the
3,600-second and 104,857,600-byte budgets. The parent manifest recorded 460,507
bytes before its final manifest write.

| Seed | Choice agreement | Cup route | Fixed held-lift | Observation held-lift | Observation − fixed |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 400 | 63/64 | 6/6 | 24/64 | 21/64 | −3 |
| 401 | 62/64 | 6/6 | 24/64 | 28/64 | +4 |
| 402 | 64/64 | 6/6 | 23/64 | 23/64 | 0 |
| 403 | 62/64 | 6/6 | 23/64 | 26/64 | +3 |
| 404 | 60/64 | 6/6 | 24/64 | 25/64 | +1 |
| **Total** | **311/320** | **30/30** | **118/320** | **123/320** | **+5** |

The paired lift outcomes were 99 both-success, 178 neither, 24 observation
gains and 19 observation losses. Object-wise confusion had 30/30 correct cup
choices; the nine non-matches were `source_e260 → cup_e340` on seven cubesmall
and two waterbottle environments. All other object groups were exact.

The predeclared five-seed bootstrap used 20,000 draws and RNG seed `20260926`,
resampling seed-level observation-minus-fixed rates. Its descriptive 95%
percentile interval is **+1.5625 pp [−1.8750, +4.6875] pp**. This interval is
not a superiority test or a general-population guarantee.

All predeclared gates pass: every seed has at least 60/64 choices, every cup
chooses `cup_e340`, every observation arm has at least 16/64 held-lifts, the
total observation count is at least 100/320, and observation is no more than
30/320 below fixed. The exact machine-readable index and reproducible CPU
analysis are [VAL-20260926-observation-six-expert-c1-results.json](VAL-20260926-observation-six-expert-c1-results.json)
and [VAL-20260926-observation-six-expert-c1-analysis.py](VAL-20260926-observation-six-expert-c1-analysis.py).

### Decision Memo and disposition

**Decision:** whether to promote the completed valid matrix to the predeclared
`SUPPORTED` terminal label for the narrow C1 claim. Root accepted Option A at
the Decision Checkpoint on 2026-09-27.

**Key evidence:** all technical gates pass; 311/320 choices agree with the
fixed route, cup routing is 30/30, observation held-lift is 123/320 versus
118/320 fixed, and paired gains/losses are 24/19. The descriptive seed-level
interval crosses zero but is not a predeclared superiority gate.

**Option A — promote `SUPPORTED` at the checkpoint.** No new compute. Retain
the frozen observation-driven six-expert hierarchy as the C1 substrate for a
separately authorized next design, with the scope limits below.

**Option B — keep the label pending review.** No new compute. Preserve this
valid matrix and defer formal promotion; no holdout rerun or tuning is needed.

**Disposition:** Option A was accepted. This task-bound route reliability
result does not establish a shared GRAB actor, unseen-object generalization,
improvement over fixed routing, or Cm policy utility.
