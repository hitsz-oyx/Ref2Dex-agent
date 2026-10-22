---
schema: ref2dex.probe.v2
probe_id: P-20261022-single-tau-calibration
experiment_id: P-20261022-single-tau-calibration
date: 2026-10-22
task: ptv3-shared-wam-v6
branch: main
git_commit: 1f41bcdbd613335b11d7accca9b2ec1035ec6e24
claim_id: C1
hypothesis_family: HF-single-trajectory-tau-feasibility
probe_index_in_family: 1
seed_pool: probe
seeds: [195, 196]
classification: Blocker
status: COMPLETED
run_id: single-tau-calibration-20261022-r1
decision_changed_if_positive: close the narrow implementation audit with a physical positive witness before new RL
decision_changed_if_negative: keep RL HOLD and identify the failed same-channel executor boundary without rejecting WAM
---

# 单轨迹同τ通道可实现性校准（执行前注册）

采纳机主v2(E)。区分“该τ输入/执行链在一个见过的可行任务上仍不可实现”与“同架构局部适配
可以给出物理阳性见证”。这是最便宜的Blocker检查；不测泛化、τ必要性或样本效率。
完整RELEASE任务和合成8cm精确终点任务的成功率都不由本卡替代。

## 固定任务、链路与目标数值

- 固定`s1_cylindersmall_pass_1`：上一批6例按native顺序的首条，选择不依赖本轮拟合结果。
  一只Inspire/一个物体/一条轨迹；此序列从本轮起是训练见过，不能再作未见序列验证。
- **本卡独立任务**：初始物体上升≥5cm、真实force≥0.1N的至少两指、真实运动耦合、
  未失握且离开桌面，连续真实elapsed≥1s。物理安全abort否决当前成功；不把30状态算1s。
  同时保存ever_success/final_success。它不同于三臂当前“精确8cm endpoint HOLD1s”，报告分开。
- source-PD：读取已保存official PD targets，以live q反解合法动作、continuous rpy采用最近
  2π等价代表，记录clipping；重新采集这次真实物理轨迹和实际执行动作。预期本链1/1可行。
- 基线R：原Scheduled686 step50000。校准R：复制同一架构/权重/归一化，冻结shape encoder，
  只对该条新source-PD实录的measured347+task10+GTτ8→实际canonical12微拟合。
  目标训练action RMSE至少下降80%；GTτ物理成功目标1/1，失败不改门槛。
- 同τ通道：τ[0]=第一未来1/30s，8步=8/30s；GT参考未来−参考当前，转native11顺序；
  初始3帧与末端缺8未来的帧τ invalid。部署不使用新的实测未来，不做额外DAgger。
  无τ后备模式预先固定为同executor的τ=0/valid=false；不替换PPO action制造错误logprob。
- 分开列出source-PD、原R/校准R的GTτ、原R/校准R的WAMτ(step11000)。WAM已有adapter
  保留；不能以source-PD成功认证GTτ→R，不能以微拟合结果认证泛化。

## 固定预算与停止

GPU使用前nvidia-smi，native GPU4、模型GPU5；0–3/7不动。优化器Adam lr1e-4、batch64、
seed196；**最多2000updates或1800s（30min），先到先停**，不按成功挑checkpoint。
保存最后权重及完整loss曲线、源码diff/snapshot；基线不修改，不启动v6训练或RL。
物理每条最多300controls（10s），source1次、GT基线/校准各2次、WAM基线/校准各1次，
合计最多2100envcontrols；重复为同单轨迹复跑，不当独立样本。全部链用共同初态逐数组核对。
总wall上限3600s、产物上限2GiB、磁盘余量20GiB；错误只停止自有进程。首次成功或物理
abort结束该回合，encoder失败只屏蔽τ并继续；teacher不达标则不把后续R失败作为方法否决。

编译/接口smoke为纯工程，可先运行。完整RL三臂保持HOLD；本卡结束即以一页结论收口审计，
原任务意图、完整RELEASE和广泛泛化问题列future evidence，不继续无界扩大检查。

## 结果：执行完成，GTτ executor资格未通过

运行代码为上列base commit加本轮dirty diff，run内`source.diff`与`code/`保留实际代码。
GPU4物理、GPU5拟合/推理；使用前均核到6MiB。实际数据80controls/81states，监督拟合
2000updates、5.3758s，action RMSE0.103540→0.001849，下降98.21%；固定预算已用尽，未加训。

| 链 | 窄任务成功/重复 | 最长elapsed持握 | 实际controls |
|---|---:|---:|---:|
| source-PD |1/1|1.000s|80|
| 原R+GTτ |0/2|0 /0s|300+300|
| 校准R+GTτ |0/2|0.633 /0.467s|126+300|
| 原R+WAM11000 |0/1|0.9667s|232|
| 校准R+WAM11000 |0/1|0s|150|

合计1488envcontrols、总wall118.24s；source动作clipping0，共同初态9数组最大差0。
有3条R链触发真实limited-joint safety代理；其余到达注册时域。真实物理后缀有差异，
两次复跑不是两条独立轨迹。WAM累计推理34.558s；checkpoint只读，无v6/PPO训练。

独立保存状态重算7/7 verdict一致，按连续整数区间≥31状态认证1秒；FeedbackObserver仍
复用，因此不是独立接触真值。GT参考只有81状态，有效GTτ70controls，之后进入预注册
state-only fallback，限制后段失败归因；校准R已在64/66帧出现首次失持，当时GT仍有效。
没有延长原参考、伪造未来、降低持握/安全门槛或增加监督样本/updates。

结论：source-PD证明本窄物理任务可行；当前GTτ→R未获阳性，不能把source1/1移交给它，
不能把动作拟合好解释成闭环资格PASS。正式RL继续HOLD；不据此否决τ/预训练路线。

产物：`outputs/ptv3-shared-wam-v6/single-tau-calibration-20261022-r1/`：manifest、
calibrated.pt、7条trajectory、independent-score.json、两份NOT_QUALIFIED receipt、退出记录。
两份receipt均被正式资格检查明确拒绝`gt_tau_executor below95%`。一页结算与复现入口：
[negative-result-audit-one-page](../../audit/negative-result-audit-one-page-20261022.md)。
