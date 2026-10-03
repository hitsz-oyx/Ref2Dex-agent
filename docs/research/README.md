- [Exact resume implementation and current resource blocker](../activities/20261002-continuous-critic-resume-ready.md)
Current independent continuation: `/tmp/Ref2Dex-contact-response-continuation`,
same research branch/history with separate local Git metadata; old worktree read-only.
- [Completed selective-finger negative](20261002-selective-finger-feasibility-results.md)
- [Continuous method boundary](20261002-continuous-method-boundary.md)
- [Actual training progress and GPU/runtime interruption](../activities/20261002-continuous-critic-runtime-change.md)
- [Paper10source](../../paper/manuscript-v10.tex) / [reviewPDF](../../paper/manuscript-v10.pdf)

# Research 文档

- [Independently supervised implementation review and confirmed native collision-filter bug](20261003-implementation-review.md).
- [Completed fixed-policy sensitivity to corrected collision filters](20261003-inspire-filter-impact-results.md).
- [Cm knowledge-prior data scale and coverage hypothesis](20261003-cm-data-scale-hypothesis.md): unresolved; proposed controlled design, no new run.
- [Actual scale and MANO–Inspire prior transfer result](20261003-cm-scale-cross-hand-results.md):15600updates, all audits pass; complete gates fail, offline-input limitation remains.
- [Raw hand points and actual query granularity](20261003-surface-granularity-results.md):2048/10135raw hand points;64query coverage audit completed, no demonstrated accuracy cause.
- [Fixed-data query/neighbor granularity result](20261003-cm-granularity-results.md): eight matched fits/12000updates, all audits pass; primary error changes<0.3%, all six gates fail.

## 当前独立工作树研究

`agent/contact-response-cm` 的现状见 [STATE](../STATE.md)，首轮实际结果见
[contact-response results](20261001-contact-response-results.md)，文献和 novelty
边界见 [literature](20261001-contact-response-literature.md)。论文工作稿位于
[paper](../../paper/README.md)。此分支的实验、写作与产物独立于原 paired-evaluator
工作树；旧冻结指令和状态记录用于解释历史证据。

后续实际结果：[actuation/effect screen](20261001-actuation-effect-results.md)，
[exact-null recovery screen](20261001-null-action-results.md)。新的候选机制与
已有研究的边界见 [action-recovery boundary](20261001-action-recovery-boundary.md)。
全部失败门槛保留；当前活跃实验以本工作树 STATE 和对应 run manifest 为准。
新的 [prospective randomized risk result](20261001-randomized-effect-risk-results.md)
完整采集3072窗口，未支持历史模型的因果效果优势；不能当作政策收益。
[direct randomized response](20261001-direct-randomized-response-results.md)
在另一批3072新窗口上未通过门槛；其事实模型对照的事后信号只用于选择下一轮
[抓取保持/掉落实验](../experiments/probes/P-20261001-randomized-task-selection.md)。
该 [任务实验](20261001-randomized-task-selection-results.md) 已完整运行3072首回合，
固定门槛失败；发现参考抬升仅25--36帧且会正常放回桌面，与45帧保持及整回合不回落
要求冲突。下一步先验证单独标注的合成保持任务底座，不重写旧结果。

新完成的 [holding curriculum](20261001-hold-plateau-curriculum-results.md)0/384，
[static PD](20261002-static-hold-feasibility-results.md)0/192必要保持条件；其错误桌面轴
完整保留，单独离线修正。 [bounded closure](20261002-finger-preload-feasibility-results.md)
有部分机械保持实例，但整体门槛仍失败，停止原静态剂量族；论文第五版报告全部结果。

这里按用途索引研究事实；具体的当前文件暂时保留在 `docs/` 根目录，因为
`tools/verify.py`、实验卡和已有交接把它们作为稳定入口。

| 类别 | 入口 |
| --- | --- |
| 研究问题 | [`MISSION.md`](../MISSION.md) |
| 当前事实与 North-star | [`STATE.md`](../STATE.md) |
| 机器和资源边界 | [`CAMPAIGN.md`](../CAMPAIGN.md) |
| 当前可消费路线 | [`RESEARCH_QUEUE.yaml`](../RESEARCH_QUEUE.yaml) |
| seed 归属 | [`SEED_LEDGER.yaml`](../SEED_LEDGER.yaml) |
| 论文后补实验 | [`RESEARCH_DEBT.md`](../RESEARCH_DEBT.md) |
| 项目全局概览 | [`项目总览.md`](../项目总览.md) |

研究记录已经按目录分开：

* `decisions/`：Decision Checkpoint 和路线选择；
* `experiments/probes/`、`experiments/validations/`：实验卡和正式验证；
* `handoffs/`：研究路线交接和证据审计；
* `activities/`：跨 Task 研究活动；
* `archive/`、`logs/`：只读历史资料，不是默认上下文。

旧的根级 `plan/`、`指导/` 和多代理 workflow 规范已删除。Task 内部仍可能保留与历史
实验绑定的计划或指导文件；它们只用于解释对应证据，不是仓库级运行规则。

## Independent contact-response update, 2 October 2026

- [Support-removal witness](20261002-support-removal-witness-results.md): PROMISING mechanical Probe.
- [Scratch observation-policy fit](20261002-observation-hold-baseline-results.md): UNPROMISING0/192.
- [One fresh aggregation](20261002-observation-hold-aggregation-results.md): UNPROMISING0/192.
- [Absolute-target initializer](20261002-reference-target-policy-results.md): PROMISING56/64on primary motion1; other motions0.
- [Updated primary-source novelty constraints](20261002-contact-response-novelty-update.md).

Paper revision9incorporates these and the newer results below without a Cm utility
or journal-readiness claim. Earlier revisions remain preserved.

- [Executable-support forecasting](20261002-support-response-information-results.md):6144complete trajectories, strong-global primary gate fails.
- [Gradient / positional headroom](20261002-support-gradient-and-headroom-results.md): two explicitly reused-data screens, both UNPROMISING.
- [Post-lift disturbance feasibility](20261002-support-disturbance-feasibility-results.md):1536complete fresh trajectories, no eligible fixed load.
- [Control-variate primary sources](20261002-physical-control-variate-literature.md) and [recovery novelty boundary](20261002-disturbance-recovery-literature.md).
- [First actual matched policy-training result](20261002-support-feature-policy-results.md):9216training/1536evaluation, UNPROMISING; all three tested deterministic decision rules coincide.
- [Full-text method boundaries](20261002-fulltext-method-boundary.md): generic contact prediction and frozen-feature adaptation are already occupied.
- [Temporal decision review](../decisions/D-20261002-natural-retention-headroom.md): reused training-only natural loss counts, no recovery or policy-utility claim.

- [Natural feedback failure](20261002-natural-retention-feedback-results.md):1536fresh trajectories, both event arms trail unchanged; UNPROMISING.
- [Predictive/reactive method boundaries](20261002-feedback-method-boundary.md): primary full-text sources, no novelty from generic slip alarms.
- [Action representation review](../decisions/D-20261002-contact-action-representation-review.md): selective effective finger directions, next design pending.

- [Complete continuous policy comparison](20261002-continuous-critic-policy-results.md):15360training/1536evaluation, Cm129/state129/no-aux147/384; UNPROMISING. Original failures, same-model GPU migration and separate witnessed ReLU audit correction retained; native manuscriptv11preserves all prior evidence.

- [Training-only request execution headroom](20261002-request-execution-headroom-results.md):403200command transitions,100%meaningful in all three arms; not physical response or policy benefit. Projection repair is not the main next route.

- [Fixed delayed-response UNCLEAR](20261002-delayed-request-response-results.md)
- [Response-field prior-art boundary](20261002-response-field-method-boundary.md)

- [Fixed state-conditioned response field negative](20261002-state-response-field-results.md)

- [Independent rotation-only retention negative](20261002-rotation-retention-feasibility-results.md)
- [Impulse and model-to-actor prior-art boundary](20261002-impulse-gradient-method-boundary.md)

- [Force-aware non-gravity impulse forecast negative](20261002-force-aware-impulse-results.md)

- [Measured geometry binding and failed FP64 recovery](20261002-object-frame-binding-results.md).

- [Measured geometry barrier result](20261002-measured-geometry-barrier-results.md).
- [Geometric contact model primary-source boundary](20261002-measured-geometry-method-boundary.md).

- [Actual coherent four-tick consequence negative](20261002-coherent-barrier-information-results.md).

- [Self-trained source qualification negative](20261002-self-trained-teacher-qualification-results.md).
- [Existing model/value interface boundary](20261002-model-value-interface-boundary.md).

- [Object-relative feedback negative](20261002-object-relative-transport-results.md).

- [Task determinacy diagnostic](20261002-task-outcome-determinacy-results.md).

- [Fixed-policy truth-successor task-value result](20261002-truth-successor-task-value-results.md): UNPROMISING; oracle gain over direct Q0.9354% fails the frozen1% gate. This track is being finalized at the user's request before a new route.

- [Resumed observed-support task-value result](20261002-observed-support-task-value-results.md): UNPROMISING;1.6088% descriptive gain but paired interval crosses zero.

- [Paired joint-option opportunity and engineering checks](20261002-paired-option-task-opportunity-results.md): UNCLEAR; pre-intervention matching fails despite identical reset states. No same-state oracle utility inferred.

- [Actual option-model policy learning](20261002-option-model-policy-results.md): UNPROMISING; Cm172/off174/directQ222/P0129 per384. Complete3072native trajectories and9000optimizer steps, original JSON failure and same-data correction preserved.
- [Return-corrected gradient prior-art and algebra](20261002-return-corrected-gradient-boundary.md).
- [Fresh complete actor-gradient control](20261002-physical-gradient-control-results.md): UNPROMISING; Cm32.58 versus baseline8.25/off11.94/directQ14.05 covariance trace; actual14732-parameter gradients independently audited, no new training.
- [Reused actual-successor/value compatibility](20261002-successor-value-compatibility-results.md): PROMISING for a different representation only; oracleBrier.08441/directQ.12246/meanCm.12047, no policy or Jensen claim.

- [Observed successor law actual policy result](20261002-empirical-successor-policy-results.md): UNPROMISING; Cm188/off188/directQ222/P0126 per384, all native/input/value checks pass.
- [Empirical successor prior-art boundary](20261002-empirical-successor-method-boundary.md).

- [Physical encoder transfer primary-source boundary](20261002-physical-encoder-transfer-boundary.md).

- [Physical encoder into measured-return task critic](20261002-physical-encoder-critic-results.md): UNPROMISING; Cm219/off218/directQ225/P0143 per384, all checks pass.

- [Training-only late-option support review and cohort correction](20261002-late-option-support-results.md): motion0zero complete/transient labels in256Gaussian plus256zero trajectories; choose earlier preparation data.

- [Finite prelift contact opportunity](20261003-prelift-contact-opportunity-results.md): UNPROMISING; randommotion0zero/256, full control/P0-restoration/native checks pass.

- [Joint physical/Q prior-art boundary](20261003-budgeted-physical-critic-boundary.md).

- [Equal-budget joint physical critic](20261003-budgeted-physical-critic-results.md):
  UNPROMISING, Cm383/off399 per768; equal-budget block Cm183/labelQ220 per384.
  All native/input/model/actor audits pass; pre-fit normalizer correction and
  first valid short-panel inheritance preserved without duplicate collection.
