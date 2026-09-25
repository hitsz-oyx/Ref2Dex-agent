---
schema: ref2dex.validation.v2
validation_id: VAL-YYYYMMDD-topic
date: YYYY-MM-DD
branch: agent/<hypothesis-family>
git_commit: <card-creation-commit>
claim_id: C3
hypothesis_family: HF02
frozen_method_commit: <immutable-method-commit>
development_seed_pool: validation.development
validation_seed_pool: validation.holdout
matched_control: <precisely matched Cm-off/control definition>
status: PLANNED
---

# Validation: <formal claim>

## Claim

准备验证的正式 claim：

## Hypothesis

H1:

H0:

## Treatment

Cm-on:

## Matched control

Cm-off:

## Frozen variables

必须相同：

* initialization:
* data:
* environment:
* reward except tested variable:
* training budget:
* checkpoint selection:
* evaluation protocol:

## Primary metric

Metric:

## Predefined success criterion

在看到正式结果前固定：

## Runs

Development seeds:

Validation holdout seeds:

Repeats:

Environment count:

Training budget:

## Stop conditions

工程失败：

资源异常：

scientific early stop:

## Result

run_status:

`COMPLETED` | `FAILED` | `STOPPED`

conclusion:

`SUPPORTED` | `REFUTED` | `INCONCLUSIVE` | `INVALID_IMPLEMENTATION`

## Evidence

核心结果：

## Limitations

该实验不能说明什么？

## Artifacts

checkpoints:

metrics:

logs:

analysis:
