# WAM 预测学习为何尚未转化为控制收益

2026-10-16。复查预训练、机器人联合微调、冻结特征腕部对照和新隔离片段实验的代码与既有产物。属于跨实验机制审查，不是新的训练或 Validation。保留所有既有实验及其 Probe 标签，不把候选机制写成已证实根因。

## 当前判断

已有离线预测学习信号，但不能自动推出反馈控制充分性。当前最可靠的负证据限于：**v2 三源预训练后又经过 Inspire 联合微调的 WAM，经三组均值池化接入当前腕部 MLP，没有改善固定 DEV 指标**。这没有分别隔离原预训练质量、联合微调变化、特征读出和控制泛化。

来源：[预训练卡](../experiments/probes/P-20261014-ptv3-shared-v2-pretrain.md)、[匹配表征卡](../experiments/probes/P-20261016-wrist-shared-matched.md)、[行为卡](../experiments/probes/P-20261016-wrist-shared-grasp.md)。

## 1. 冻结的是联合微调后的表征，并非原预训练表征

[实际特征 manifest](../../../../../outputs/ptv3-shared-wam-v2/wrist-shared-matched-20261016-r1/features/manifest.json) 指向 `shared-native-grasp-20261015-r1/inputs/control-step7000.pt`。其母运行是 `shared-actor-shared-20261015-r1`，父模型为 v2 三源 `remote-parent-20261015-r1/step80000.pt`，不是新九源 v3。

[robot_model.py](../../src/ptv3_shared_wam_v2/robot_model.py) 的 shared tokens 未 detach，全部 forecaster 参数进入 optimizer；[train_robot.py](../../tools/run/train_robot.py) 的学习循环只有 robot batch，没有人类 replay。原配置 geometry/shared/robot LR 为 5e-6/1e-5/5e-5，forecast 与 action 联合训练。

因此“联合微调改变了迁移能力”是合理候选，但尚无前后匹配评估证明发生遗忘。最直接的缺失对照是同一控制读出下的 **80k 原父权重 vs 7k 机器人权重**。

## 2. 预测目标与控制所需信息不同

[预训练目标构造](../../../goal-point-wam/src/goal_point_wam/data.py) 使用目标物体实测 `effect[-1]` 生成终点条件；[机器人 loader](../../../ptv3-shared-wam/src/ptv3_shared_wam/rl_robot.py) 则用 `reference_obj[t+24]` 给目标，标签仍来自实际执行。

这让预训练主要学习“已知实现终点时，手与场景如何共同运动”。实际机器人需要处理参考未达、误差恢复和控制反馈。WAM 编码器读几何、H4 运动和目标终点，不读 native 执行器状态、完整参考误差或候选动作；[模型](../../src/ptv3_shared_wam_v2/model.py) 的损失是 horizon-normalized 手/场景点位移 MSE。

这不是标签泄漏 bug：目标条件是设计的一部分，且已有预测指标优于简单基线。但这类监督没有保证 tokens 足以解教师的反馈控制命令。raw811 又已经包含 state87、goal3、current721，WAM 新增信息主要是几何和短历史先验，不能仅因增加特征就预期即时腕部误差下降。

## 3. 均值读出与预训练逐点读出不一致

[pool_shared](../../src/ptv3_shared_wam_v2/wrist_shared_probe.py) 对 scene、hand、goal 各自均值池化，拼成2304维。机器人只有右手有效，因此11个有效手点 token 被压成一个768维向量。原 WAM 却分别对每个手点 token 预测其未来位置。

均值后的控制器失去了直接寻址 wrist/tip token 的接口；上下文可能把身份信息编码进均值，不能断言信息全部丢失。工程复现中，交换已编码手点 token 后均值变化仅1.19e-7，说明该读出确实对这种排列不敏感。**这项机制只约束均值池化实验，不能解释所有保留完整 tokens 的 query-head 结果。**

最便宜的读出判别是保持2304维、支路、初始化和训练不变，只把 hand 平均替换为预训练可寻址的 wrist token；不同时扩大网络。

## 4. 匹配参数总数没有匹配有效容量

同一 [WristSharedMLP](../../src/ptv3_shared_wam_v2/wrist_shared_probe.py) 包含 raw MLP 3,458,566 参数，以及2304→1024支路2,359,296参数。none 输入 z 恒零，该支路梯度恒零；aligned/shuffled 启用它，相对 raw 增加68.2%有效容量。合成合同复现确认了死支路与非零输入支路的梯度区别。

`condition_rows` 的 shuffled 是各 split 一次性固定排列，不是每步重采样噪声；每个训练样本获得稳定额外编码。shuffled 也改善 TRAIN，说明不能把 TRAIN 增益直接归功于物理知识。它仍是有用的错配对照，但未来应加入有效容量或冻结随机特征控制。

## 5. 线性残差负结果的范围有限

[冻结残差实验](../../tools/run/probe_frozen_wrist_residual.py) 只用 `Linear(z2304→6)`，没有 raw 状态条件。WAM 输入锚定当前物体坐标，而输出包含世界坐标腕部控制；全局线性映射不能一般性完成依赖物体姿态的坐标转换。其负结果不能否定状态条件残差或其他 token readout。

对象坐标还带来物理信息边界：调用实际 [robot_goal_sample](../../../trajectory-policy/src/trajectory_policy/wam_shadow.py)，把手、物体、目标整体绕世界 X 轴旋转90度后，WAM 的 xyz/features/start_hand/history/goal 输入逐值相同。这意味着编码器自身看不到重力在物体局部坐标中的方向；下游 raw state 仍含 world pose，所以这是表示边界，不是执行旋转错误，也不是已证实失效原因。

## 6. 新隔离实验改变了控制泛化判断

[新卡](../experiments/probes/P-20261016-teacher-sequence-generalization.md) 的实际预测已独立复算：

| 组 | 序列等权 PD XYZ mm | 逐帧 mm |
|---|---:|---:|
| seen | 0.564 | 1.334 |
| 同序列隔离 | 32.652 | 30.371 |
| unseen | 42.436 | 41.623 |

不能再只归因于新序列缺失。另一方面，隔离到最近 seen 的中位距离37帧、最大65帧（30Hz下约1.23–2.17秒），时间中间段不等于输入空间局部邻域。只沿各序列 seen 的相对 XYZ 命令做时间线性插值，held 逐帧误差125.06mm、序列等权133.90mm；学生30.37mm好于这个粗基线。它仍有明显控制泛化缺口，但不能称完全没学到规律或已证明只记住序列。

腕部 relative PD 与 native action 的代数关系也已复算：seen6849行中，XYZ `q_PD-q-action` 最大5.96e-8，旋转与 `pi*action` 最大约9.80e-7。**仅把 relativePD 换成 native action 名字不会引入新的控制规律。**

该实验没有 WAM，所以不能直接说明 WAM 能否解决这一缺口。新的隔离面板可用作后续匹配表征诊断，但须先确保特征窗口、workspace准入与各臂样本集合完全一致。

## 已核对、暂未发现的问题

- 特征提取按明确的(sequence,tick)重建顺序，merge按indices覆盖全部行；未发现缓存行错配。
- 归一化只用 TRAIN。每 split 均匀抽2048行，三组标准化特征 RMS 约1，std最小2.439；没有小std导致的整体放大。
- 已存推理重放误差远小于DEV劣化，不支持把约4.8mm的差距归为一般GPU数值误差。
- 原始手序、物体坐标变换、位移与PD标签合同未发现新的确定性错误。
- 未测单项梯度夹角，不能由 hand/scene/action 标量权重推断多任务梯度冲突。
- 原完整token query头也有弱结果，不能把全部失败推给pooling；其8拍全动作目标、当前状态融合、优化曝光等仍需匹配区分。

## 下一个最有判别力的问题

优先检验 **原80k表征 vs 机器人7k表征**，保持 raw 通路、读出、数据、优化和参数一致；若原父权重更好，定位联合适配。若两者都弱，再做同维度 wrist-token vs hand-mean，定位读出。随后才区分表征信息不足与控制学习归纳偏置，不无条件追加同配方 joint/PPO 训练。

以上为代码审查阶段的候选设计；随后按用户自主探索授权完成了下述独立Probe。

## 后续探针：来源与腕部读出均未救回当前接法

[冻结来源六臂](../experiments/probes/P-20261016-wam-source-control-transfer.md)沿用新teacher
划分和全部raw/优化设置。none隔离/unseen为32.652/42.436mm；parent80k mean为
36.925/48.588、wrist37.311/48.887；robot7k mean36.636/51.777、wrist37.333/52.225。
raw固定随机投影为30.816/41.652，未达双组10%门。真实WAM更好拟合seen，却在两留出组
macro/p95都恶化。本接法UNPROMISING；原始预训练与wrist读出也弱，因此单用“机器人
微调破坏表征”或“平均池化丢掉wrist”解释全部结果不再充分。没有正式证明遗忘与过拟合机制。

none终点raw权重与旧24k逐tensor相等；六臂原标签/成员/指标复算通过，未发现新的实现错误。
独立审查另确认robot7k旧TRAIN已覆盖隔离1670/1707条，旧DEV包含unseen800/987条；
严格隔离仅针对新控制头，不能称整个系统未见泛化。结论限于三域v2与当前2304D接法，
不延伸到九源v3、跨手型、物理成功或PPO收益。下一步优先判别BC状态覆盖和控制学习
归纳偏置；保留WAM核心假设，不继续同类来源/聚合搜索。

## 可复算工程与数组统计

脚本：[tmp/audit-wam-control-chain-20261016.py](../../../../../tmp/audit-wam-control-chain-20261016.py)；输出：[contracts.json](../../../../../tmp/wam-control-chain-audit-20261016-r1/contracts.json)。仅4个合成网络输入和既有数组统计，CPU约2秒，无模型训练、无批量WAM推理。

```bash
TMPDIR="$PWD/tmp" PYTHONDONTWRITEBYTECODE=1 /home2/wyy/miniconda3/envs/graspenv/bin/python tmp/audit-wam-control-chain-20261016.py
```
