# PointWAM robot joint fine-tuning audit

Checked 2026-10-12; reuses the existing schema and architecture/source audits.

| Published contract | Primary location |
| --- | --- |
| Human-pretrained backbone and both trajectory heads are robot-finetuned; retargeter starts random. Frozen: Mosaic3D/Recap-CLIP, not the forecasting backbone. | [§3.4/Eq.7](https://arxiv.org/html/2610.02840v1#S3.SS4), [C.1/Table4](https://arxiv.org/html/2610.02840v1#A3.SS1) |
| Objective: scene MSE + hand MSE + recorded-action L1; coefficients one, coordinate means, valid-hand masking, motion-weighted scene targets. | [Eq.7](https://arxiv.org/html/2610.02840v1#S3.SS4), [B.5](https://arxiv.org/html/2610.02840v1#A2.SS5) |
| Retargeter takes predicted hand displacements/current end-effector-and-finger state; two decoder blocks/four heads output 30 parallel actions, not scene forecasts. | [§3.3/Eq.6](https://arxiv.org/html/2610.02840v1#S3.SS3), [B.4](https://arxiv.org/html/2610.02840v1#A2.SS4) |
| Trajectories retain human per-horizon normalization. Actions use robot per-horizon statistics: state-relative position/fingers, orientation rotation-vector; 22 coordinates/arm. | [B.5](https://arxiv.org/html/2610.02840v1#A2.SS5) |
| DexJoCo: 60k updates, batch192, AdamW LR1e-4 constant, decay.01, clipping5, no warmup. | [C.1/Table4](https://arxiv.org/html/2610.02840v1#A3.SS1) |
| RoboDojo lacks scene labels: hand+action supervision, scene head frozen. | [B.2](https://arxiv.org/html/2610.02840v1#A2.SS2) |

**Inference:** the differentiable published composition implies action→R→predicted hand trajectory→forecaster/backbone gradients without detach. Actual autograd/parameter groups and any GT-trajectory mixing remain source-unverified. [Equations2/6/7](https://arxiv.org/html/2610.02840v1#S3.SS3)

Official project still says Code soon; no released author training implementation was linked. [Project](https://chrockey.github.io/PointWAM/)

本地应用：联合微调可直接借鉴。保留已训练 R、当前步12D PD目标与24步11点手轨迹是本地变体；不能称作论文完整复现。直接轨迹标签约束 τ，动作损失同时更新 R 和 WAM；需要本地核验两者梯度及实际闭环效果。


## 2026-10-13 ref3_2 assessment against current implementation

Primary HTML rechecked: §3.3–3.4 supports paired forecaster/retargeter robot
training, FK hand targets and direct scene supervision; C.2 describes an async
30-action buffer refilled below24 remaining. These facts motivate a local
Probe, not a causal explanation for our failed grasp.

The proposal omits prior local evidence: Probe ref2-joint-retargeter already
trained the policy/R end-to-end with real robot trajectory supervision, without
frozen-World effect loss. Nominal GT-tau R PDp95 worsened10.59→43.96mm; mapping
drift prevented physics admission. Joint updates alone do not preserve physical
tau semantics. Subsequent ref22/ref23 experiments added GT-tau R anchoring and
changed gradient ownership. Current actor training already includes true robot
hand trajectory and recorded-action losses; it is not an unconstrained latent.
Current ref2_8 actor additionally sees q/dq/previous command and planned object
poses at+8/+16/+24, so the one-endpoint description is stale. Negative world-gap
is a potential exploitation signal, not proof that effect loss caused failure.

Decision: retain ongoing same-row visited-state collection and the bounded
fixed-R control. Next useful comparison holds those data, initialization,
draws and execution fixed: effect-on/fixed-R versus effect-off/fixed-R, then
(if still decision-relevant) effect-off/joint-R with direct GT-tau→R anchoring.
Require both geometric and action improvements, nominal/startup retention,
whole-prefix development holdout and actual autonomous hold45/drop checks.
Save/admit actor and R as one bound pair. Do not change action chunk execution
in this comparison. Our frozen action-conditioned World cannot become a
PointWAM shared-backbone auxiliary scene head merely by renaming its loss;
that would be a separate implementation and research variable.

This is a recommended Probe, not launched by the user's question about value.
It supports MISSION baseline-first priority; eventual matched Cm-on/off policy
training evidence remains required. No claim that PointWAM guarantees our
embodiment, data coverage or controller will grasp.
