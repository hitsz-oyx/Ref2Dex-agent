---
schema: ref2dex.probe.v2
probe_id: P-20261023-tau-fixed-panel-metric
experiment_id: P-20261023-tau-fixed-panel-metric
date: 2026-10-23
task: ptv3-shared-wam-v6
branch: main
git_commit: f30e8d7ba803d1d1b64270636dd807bf1461a5fb
classification: Decision
claim_id: C3
hypothesis_family: wam-v6-fixed-train-tau-quality
probe_index_in_family: 1
status: COMPLETED
run_id: tau-fixed-panel-baseline-20261023-r1
seed_pool: probe
seeds: [170, 172]
decision_changed_if_positive: use the fixed panel as the τ improvement gate for the next checkpoint
decision_changed_if_negative: keep the checkpoint from being reported as an overall τ improvement and inspect the failing phase
---

# P-20261023-tau-fixed-panel-metric

## 2026-10-23 四格条件归因补测

run_id `tau-condition-ab-20261023-r1`，实现预注册 `708b6e9b`、实际评估源码
`baac96c7`，同一512锚点/124母序列，无训练或重采样。新增 init@B 为
9.357853/19.834800/4.816335mm，trained@B 为12.131092/45.561236/7.861320mm
（前8手/接触窗手/前8物体）。旧模型 B 比 A 误差下降42.13/66.51/57.20%，
新模型 B 比 A 误差上升17.91/43.46/20.86%，配对cluster95%CI均排除0。
旧条件有效、继续训练适应A并损失B下表现；不能将+36~46%解释为无意图天然更强，
也不能只用这两格将条件与数据贡献因果拆分。B联合改变goal与event，且端点来自
未来观测，仅诊断、禁止作为部署配方；不支持直接给新权重接回端点。
提高合法且训推一致goal消费的优先级，数据审计仍未被排除。
协议、四格表、配对差及资源见[归因报告](../../audit/tau-condition-attribution-20261023.md)。
GPU7串行两格各约29s，新增约21MiB，已释放；两格A均直接引用。

## 2026-10-23 CI 修正与 event 合同补审

主 CI 已改为124母序列整组重采样2000次；原帧级 CI 留作对照。overall、first8、
contact-window 的 B1−B0a 配对帧中位差聚类95% CI分别为
`[-50.816599,-35.525611]`、`[-21.198197,-13.684390]`、`[-36.579623,-26.001958]`mm。
8个子集两种口径均不跨0，旧TRAIN面板优势未反转；下方历史 bootstrap 表为帧级口径。

新增实际入口审计证明训练 E_next 的 type/delta/validity 来自未来 proximity teacher，
部署 TaskScheduler 使用已知任务计划与当前反馈，合同仍未统一。此前“优先买覆盖”
的建议须服从此阻塞，不是训练放行依据，也不能据旧面板宣称部署优势。
实现9eeb3fac；完整字段来源、主CI及采样选择项见
[报告](../../audit/tau-event-contract-ci-sampling-20261023.md)，产物
`outputs/ptv3-shared-wam-v6/tau-contract-20261023-r1/`。没有训练、config状态变更或新基线。

| 可能结果 | 含义 | 会改变哪个决定 |
|---|---|---|
| 新 checkpoint 在全面板前进、但接触窗口/前 8 帧退步 | 指标掩盖了关键相位 | 判为**未改进**，别报喜 |
| 全面板前进、接触窗口与前 8 帧同时前进 | τ 真改进 | 写进 8 点报告并排下一轮训练 |
| 只有个别相位动、其它不动 | 只改到局部 | 写明是哪个相位、别当整体改进 |

## Decision and scope

This Decision Probe establishes the fixed TRAIN anchor score that will decide
whether a future checkpoint improved τ. It does not train, replay physics, or
make a generalization claim. The requested distinction is between a true τ
improvement and a whole-panel average that hides early or contact-phase damage.

## Duplicate check and reuse

The following existing artifacts were checked and are retained as separate
contracts:

* [`train-anchor.json`](../../../../../../outputs/ptv3-shared-wam-v6/tau-curated-20261023-r2/train-anchor.json)
  is the equivalent fixed sample set, so the scorer consumes its 512 immutable
  `(sequence_id, slot, tick)` rows and anchor ID directly.
* [`evaluate_staged_progress.py`](../../../tools/audit/evaluate_staged_progress.py)
  scores synthetic or archived executor stage transitions; it has no
  checkpoint prediction input or fixed TRAIN anchor.
* [`prepare_executor_panel.py`](../../../tools/audit/prepare_executor_panel.py)
  selects measured robot rollout prefixes and explicitly does not replay or
  score a checkpoint.
* [`evaluate_executor.py`](../../../tools/run/evaluate_executor.py) runs a native
  closed-loop executor arm. Its physics state, policy arm, and held-out robot
  panel are not equivalent to a checkpoint-independent offline TRAIN score.
* [`run_single_tau_calibration.py`](../../../tools/audit/run_single_tau_calibration.py)
  fits and evaluates one `s1_cylindersmall_pass_1` trajectory; its
  `single-tau-calibration-20261022-r1/manifest.json` is a bounded executor
  calibration receipt, not a multi-phase fixed panel.
* [`staged-eval-panel-v1.json`](../../../configs/staged-eval-panel-v1.json)
  describes six archived native 600-control episodes, not the 512 curated
  TRAIN windows.

No existing scorer was checkpoint-independent and equivalent. The only new
scorer is [`evaluate_tau_fixed_panel.py`](../../../tools/run/evaluate_tau_fixed_panel.py),
with its frozen definition in
[`tau-fixed-panel-v1.json`](../../../configs/tau-fixed-panel-v1.json).

## Fixed panel contract

The panel uses the 512 rows and anchor ID
`94d9287bb6c650cc80e4345515e0589472786bd749981a0b25ea7475118ebc37` from the
curated TRAIN anchor. It uses seed `170`, v6 `V6Windows` H4/K24 preprocessing,
the frozen event recipe with dropout disabled, and flow evaluation seed `172`.
The order and rows do not depend on the checkpoint. A deterministic CUDA mode
(`CUBLAS_WORKSPACE_CONFIG=:4096:8` and
`torch.use_deterministic_algorithms(True)`) is part of the command contract.

The primary metric is hand future-point endpoint error in millimetres, pooled
over valid hand points and frames. The JSON reports mean, median, p95, and
count for the full panel, `approach`, `close`, `contact`, `hold`, and
`unknown` phase groups. It also reports the separate first eight future frames
and contact-window subset. A contact-window frame is one with any valid surface
contact target above confidence `0.25`; this is independent of the phase label.

## Baseline result

Command (repository root):

```bash
python3 src/task/ptv3-shared-wam-v6/tools/run/evaluate_tau_fixed_panel.py \
  --checkpoint outputs/ptv3-shared-wam-v6/flowmatching-overnight-20261022-r2/training/step011000.pt \
  --output outputs/ptv3-shared-wam-v6/tau-fixed-panel-baseline-20261023-r1 \
  --batch-size 8 --gpu 7 --max-seconds 1140
```

The machine-readable receipt is
[`baseline.json`](../../../../../../outputs/ptv3-shared-wam-v6/tau-fixed-panel-baseline-20261023-r1/baseline.json),
and the one-line summary is
[`summary.txt`](../../../../../../outputs/ptv3-shared-wam-v6/tau-fixed-panel-baseline-20261023-r1/summary.txt).

| subset | valid hand values | mean mm | median mm | p95 mm |
|---|---:|---:|---:|---:|
| full panel | 467,208 | 23.409973 | 18.326221 | 60.462026 |
| approach | 105,840 | 29.897741 | 22.739629 | 80.631971 |
| close | 101,808 | 21.873276 | 18.141441 | 53.408049 |
| contact | 105,840 | 22.194105 | 16.589462 | 61.445312 |
| hold | 103,320 | 19.982422 | 16.610634 | 49.644929 |
| unknown | 50,400 | 22.469606 | 18.194866 | 53.755281 |
| first 8 frames | 156,135 | 9.357826 | 7.325660 | 24.244514 |
| contact window | 288,960 | 19.834665 | 16.518689 | 48.541784 |

The contact window contains 7,416 of 12,288 panel sample-frames. The run used
GPU7, torch `2.4.1+cu121`, completed 64 batches in 17.5 seconds, wrote about
12 KiB, and passed the deterministic repeat check. No optimizer update,
checkpoint write, training, or physics replay occurred; `/home2` remained
above the 20 GiB safety floor.

## Interpretation boundary

These are fixed TRAIN prediction-quality numbers for checkpoint comparison.
For a future checkpoint, overall improvement counts only when the first-eight
and contact-window means also move in the favorable direction. A phase-only
change must be reported as local. The panel is not a validation/generalization
or executor-success qualification.

`git_commit` for the scored implementation is
`f30e8d7ba803d1d1b64270636dd807bf1461a5fb`.

## B0 补做：固定面板上的非学习基线（2026-10-23）

| 可能结果 | 含义 | 会改变哪个决定 |
|---|---|---|
| B1 全面板 > B0a，且 contact_window 与 first_8_frames 的 win rate 明显 > 0.5 | WAM 在两个关键子集确有可用信号 | 唯一一次封顶训练优先买有效覆盖（curated coverage），预算值得花 |
| 全面板均值 > B0a，但 contact_window / first_8_frames 的 win rate ≈ 0.5 或 < 0.5 | 只在好走的相位占优，接触区没有可用信号 | 不把预算全押扩覆盖；先改条件/语义（goal、scheduler-conditioned），并写进 8 点报告 |
| B1 ≤ B0a（任一子集） | 现有 τ 模型在该面板上不如“不动” | 立刻升级为阻塞项，不开训 |
| B0b（常速）已明显优于 B1 | 目标里“运动”成分占主导，模型欠拟合动力学 | 指向模型/优化问题，而不是数据规模 |
| 无法在不改默认口径的前提下补齐 | 打分器结构不支持对比 | 只报结构问题，不改口径，转人工 |

### 协议、查重与复现

查重覆盖 `tools/`、`outputs/` 和本 Task 实验卡中的 `hold`、`zero-motion`、
`persistence`、`const-velocity`、`trivial`、`naive`。没有发现与本 512 行、固定
TRAIN 锚点、v6 H4/K24 预测质量口径等价的既有打分。已有的
`tools/audit/evaluate_staged_progress.py`、`tools/audit/prepare_executor_panel.py`、
`tools/run/evaluate_executor.py` 和 `tools/audit/run_single_tau_calibration.py` 分别是
executor stage、测量 rollout 选择、闭环 executor、单轨迹 calibration；它们没有本面板的
checkpoint-independent offline hand EPE 合同，故不复用其数值。

新增入口
[`score_tau_panel_trivial_baselines.py`](../../../tools/run/score_tau_panel_trivial_baselines.py)
直接导入 [`evaluate_tau_fixed_panel.py`](../../../tools/run/evaluate_tau_fixed_panel.py) 的
`Accumulator`、panel 校验、固定索引和 SHA 函数。B1、B0a、B0b 在同一批次使用同一
`trajectory_target`、`trajectory_valid & start_hand_valid`、phase label、前 8 帧掩码、
contact confidence `>0.25` 和聚合实现。B0a 是零位移（未来位置保持历史最后帧）；B0b 是
最后两帧差分的 1…24 帧线性外推。两者都是位移合同下的 analytic predictor，没有 Flow
Matching 积分。

复现命令（仓库根目录；GPU4 为空闲卡）为：

```bash
python3 src/task/ptv3-shared-wam-v6/tools/run/score_tau_panel_trivial_baselines.py \
  --checkpoint outputs/ptv3-shared-wam-v6/flowmatching-overnight-20261022-r2/training/step011000.pt \
  --baseline-json outputs/ptv3-shared-wam-v6/tau-fixed-panel-baseline-20261023-r1/baseline.json \
  --output outputs/ptv3-shared-wam-v6/tau-fixed-panel-b0-20261023-r2 \
  --batch-size 8 --gpu 4 --max-seconds 900
```

结果 receipt 是
[`baseline-b0.json`](../../../../../../outputs/ptv3-shared-wam-v6/tau-fixed-panel-b0-20261023-r2/baseline-b0.json)。
原有 [`baseline.json`](../../../../../../outputs/ptv3-shared-wam-v6/tau-fixed-panel-baseline-20261023-r1/baseline.json)
只追加 `baselines`、`inference_budget`、`paired_frame_analysis`、边缘差和 receipt 引用；
原 B1 顶层数值未改动，并已在 `baselines.B1` 原样保留。
补做实现运行于 `main` 的 git commit `de22c0ba6007841e7f6ed56b23803d3852dd1373`。

### 边缘度量

| subset | B1 mean mm | B0a mean mm | B0b mean mm | B1−B0a mm |
|---|---:|---:|---:|---:|
| full panel | 23.409973 | 143.425247 | 162.054779 | -120.015274 |
| first 8 frames | 9.357826 | 52.334846 | 35.036430 | -42.977020 |
| contact window | 19.834665 | 78.880424 | 116.079155 | -59.045759 |

按 phase 的 B1/B0a mean（mm）为：approach `29.897742/318.441132`、close
`21.873276/81.554558`、contact `22.194106/75.537521`、hold
`19.982422/102.124969`、unknown `22.469606/128.100525`。B0b 只有 first-8
优于 B0a（35.036430 vs 52.334846 mm）；在全面板和 contact window 它分别为
162.054779 和 116.079155 mm，均不如 B1。

### 逐帧配对（B1−B0a，负值表示 B1 胜）

帧误差是在同一 sample-frame 的同一 valid hand 点集合上取均值；严格小于才记为
win，tie 单独报告。中位数 CI 是 2,000 次 deterministic bootstrap 的 percentile
95% CI。

| subset | valid frames | win rate | paired median mm | bootstrap 95% CI mm |
|---|---:|---:|---:|---:|
| full panel | 12,216 | 0.920105 | -42.213406 | [-43.877944, -40.660145] |
| first 8 frames | 4,091 | 0.917868 | -17.135708 | [-18.337874, -15.854436] |
| contact window | 7,416 | 0.909385 | -31.158743 | [-32.657995, -29.597768] |

按 phase 的 win rate / paired median / 95% CI 为：approach
`0.928175/-214.396160/[-232.056117,-193.484809]`、close
`0.989683/-51.274138/[-53.081296,-50.049997]`、contact
`0.881746/-21.236401/[-22.786819,-19.428128]`、hold
`0.894841/-29.834136/[-34.098859,-26.425013]`、unknown
`0.903558/-24.584451/[-26.237650,-22.170077]`。

两种口径在本面板上结论一致：关键子集边缘均值和逐帧 win rate 都支持 B1。边缘均值
差的绝对值大于逐帧中位差，说明少量大误差帧拉宽了均值；最明显的是 approach，B0a
均值为 `318.441132` mm 而 B1 为 `29.897742` mm，且该相位逐帧均值差为
`-288.543375` mm、中位差为 `-214.396160` mm。contact-window 仍保持 `0.909385`
win rate 和 `-59.045759` mm 边缘差，因此不是“只靠好走相位占优、接触区无信号”。

### 预算与决定

receipt 明确记录推理预算不等：B1 `flow_steps=4`；B0a/B0b `flow_steps=0`，只做
解析预测。运行使用 torch `2.4.1+cu121`、GPU4、64 批，约 `21.7 s`，通过 deterministic
repeat check；没有训练、optimizer update、checkpoint 或 physics replay。

结果落入第一行：B1 在全面板、first-8、contact-window 都优于 B0a，且关键子集 win
rate 明显高于 0.5；B0b 也没有超过 B1。因此本 τ 面板支持把下一次封顶训练的优先购买
放在有效 curated coverage，而不是把当前问题归因于“模型只会不动”或先开一个长训练。
这仍是固定 TRAIN 预测质量证据，不升级为 validation、泛化或 executor claim。
