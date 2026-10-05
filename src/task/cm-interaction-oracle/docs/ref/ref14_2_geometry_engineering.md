# ref14_2 P2/P3/P4：有限工程准备（2026-10-05）

本页是工程记录，不是新 Probe 卡，不改变 ref14_1 正在执行的 Oracle 判定、Y 定义或候选集。目的：复用已有物理回放，检查后续 flow-space 与 ranking 工具能否在严格合同下运作。没有新增仿真、模型训练、noise 扫描或调参。

## P2：固定同初始状态 flow 审计

入口：`src/task/cm-interaction-oracle/tools/audit/audit_candidate_flow_geometry.py`。
输入：`outputs/cm-interaction-oracle/oracle-y-utility-sync-s263-s264/paired_actual_flow.pt`，已有 ref13 s263/s264 冻结队列，共 32 anchor × 7 candidate。归档保存 120 对应手表面点的 actual 0→4 / 4→8 flow（同 current object frame），原始值按归档合同乘 .02 转回米。本工具检查归档声明的原始输入 SHA256，并对各 panel 的 before/history/tip/hand-base 逐项核对同初始状态。以后 rolling 时刻的 branch H 不同，不可用本报告拼接反事实。

输出：`outputs/cm-interaction-oracle/ref14-2-geometry-prep-20261005/candidate_flow_geometry-v3.json`。保留先前 JSON，不覆盖。报告的距离是所有对应坐标/两段 chunk 的 RMS 分量距离，不是 mean point Euclidean distance。world fingertip、moving-hand-frame fingertip、world wrist translation 分开列出；后者没有声称完整 wrist rotation decomposition。surface 同时列每根手指的对应点距离、baseline contrasts 的代数 rank 和谱熵有效 rank，不能据此直接认定有多少种有价值的候选。

描述性观察：两个 batch 的六个 surface baseline contrast 代数 rank 都是 6，谱熵有效 rank 中位约 2.93/2.89；baseline 距离均值约 2.4–5.6 mm。wrist-z+ 的 world fingertip 距离约 4.1–4.6 mm，但 moving-hand frame 下只有约 .18–.44 mm，说明手腕刚体运动与相对手指运动需要分别检查。这些值不证明 selector 或新 candidate 有控制收益。

## P3：最多四状态 Jacobian smoke

`--tiny-fk` 使用固定 Inspire URDF、已有 Torch FK 和原始 native q，针对 12 独立 driver 的局部扰动做中心差分。6 个 follower 的增量按 native PD 耦合矩阵联动（1.05、.6、.8）。condition 使用固定列单位：腕平移 .001 m，其他 .01 rad；数值依赖此单位选择。

报告包含整体/wrist/finger singular values、可达局部 rank，以及 synthetic 1 mm wrist-z 的固定 DLS 求解；在所有 native joint limits 上选择统一缩放，不独立截断 follower。初始 q 出界时返回 infeasible，而非继续伪造可行结果。耦合矩阵规定增量方向，没有把观测初始 follower 状态强行改成精确 mimic manifold。

审查修正：首版和 v2 的 P3 复用了 shared `native_joint_limits` 的 continuous-joint 默认 ±π，标记为 **INVALID ENGINEERING LIMITS**，不能作为通用可达性证据；P2 数据统计不受此错误影响。v3 在本 audit 内解析 URDF/NATIVE_TO_URDF，对 continuous joint 用无穷位置边界；有限 q/step 与无 NaN 的 ordered bounds 仍强制检查。没有改 shared mapping 或任何运行中 collector。新增测试覆盖 continuous q>π 仍可行。

4 个状态的整体 rank 为 12，非零 condition 约 35–376；synthetic wrist-z 的 nonlinear FK RMS residual 约 .003–.009 mm，limit scale 为 1。仅说明几何 smoke 可计算，且此简单目标在这些状态可几何近似。未包含接触、动力学、PD tracking、真实 desired→actual execution 或 planner gain。

CPU 原因：主运算为文件/NumPy 统计，FK 仅四个状态的极小工程 smoke；没有重复神经模型计算，不占用主实验 GPU。

## P4：ranking/noise evaluator 合同准备

`src/task/cm-interaction-oracle/src/ranking_tolerance.py` 提供：

- `same_state_ranking`：严格 `[same_states,candidates]` 合同及显式缺失 mask。GT tie 单独计数；GT strict pair 上 prediction tie 得 .5 credit；pairwise accuracy 为所有严格 pair 的 micro mean。top1 第一候选索引优先，regret 只在至少两候选可观察的状态计算。missing/空状态不作科学比较。
- `within_state_shuffle`：只在当前状态的 observed candidate 内 shuffle，不跨状态/环境，不填补缺失反事实。
- `assert_environment_oof`：拒绝 train/test environment ID 重叠；跨 seed 必须使用 `(seed,env)` 的 tuple/list/array-row 复合身份（整数 env 只在单 seed 内唯一），不能把不同 candidate/state 分到两侧。
- `inject_y_noise`：通道单位 iid Gaussian，仅修改 mask 内的 Y；没有裁剪、noise 扫描或数据驱动调参。当前只用合成数据测试它的实现。
- `closed_loop_gain_retention`：接收真实 paired rolling 各臂 binary Z 与唯一 evaluation IDs。Oracle gain≤0 时 retention 未定义。函数无法从数字鉴别其来源，调用方须保证是实际闭环执行，同 cohort/order/protocol；禁止把固定 candidate lookup/offline ranking metric 当成闭环 Z。

当前不把 evaluator 接入主实验，且不对真实 Y 做 noise sweep。Gate B 的 noise 强度、真实 rolling 协议和停止条件需在 ref14_1 判定后由 root 固定。

## 复现

```bash
/home2/wyy/miniconda3/envs/dexplore_repro_py38_torch222_cu121/bin/python \
  src/task/cm-interaction-oracle/tools/audit/audit_candidate_flow_geometry.py \
  --archive outputs/cm-interaction-oracle/oracle-y-utility-sync-s263-s264/paired_actual_flow.pt \
  --tiny-fk --output outputs/cm-interaction-oracle/<unique_run_id>/candidate_flow_geometry.json
/home2/wyy/miniconda3/envs/dexplore_repro_py38_torch222_cu121/bin/python -m pytest -q \
  src/task/cm-interaction-oracle/tests/test_candidate_flow_geometry.py \
  src/task/cm-interaction-oracle/tests/test_ranking_tolerance.py
```

验证：10 个合同测试通过，实际冻结归档读取与修正版四状态 FK smoke 完成。三个结果 JSON 总计约 312 KiB。未修改正在运行的源码、root STATE/index/ledger；未提交、切分支或 push。
