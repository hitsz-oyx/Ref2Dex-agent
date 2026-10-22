# v10：同 R 消融、接近可达性、精确到位修复

## 结果解读表

| 冻结验证结果（每臂每子集40） | 结论 | 下一步决定 |
|---|---|---|
| seen A1 ≥32/40，且满足 A1>A2 的 Fisher/CI 双门 | 本面板工程目标通过，τ 内容对同一 R 链有贡献 | 保留修复；新对象资格仍单独验收 |
| seen A1 ≥32/40，但 A2 相当 | 控制修复有效；不能把成功归功于 τ | 保留更简单的控制方案，停止 τ 独有收益主张 |
| 限位减少，但 seen A1 <32/40 或接触/到位不达标 | 可达性局部修好，完整窄任务仍未通过 | 保留失败，按实际分布修剩余项；RL 继续 HOLD |
| 原安全 gate 仍高比例中止 | 目标过滤不足，不能靠放宽 gate 过关 | 报告关节/相位/构型，继续修控制而非重训 WAM |

状态：实施与工程校验中。零训练更新；不重训 v6，不运行 RL。

冻结目标：seen 严格成功由15/40提高到至少32/40；v9未见面板 A1 关节中止40/40下降，目标≤4/40。
严格成功仍为完整600控制/20秒末，S2c实测接触耦合、离桌抬升≥5cm且连续持握≥1s，物体XYZ目标误差≤15mm，无物理中止。
HOLD任务 release 不适用。不能把持握31/40当作精确成功。

A2 保留同一个冻结 R、同一个 intent reducer、相位机和伺服，只把全部24帧未来替换为当前手点（即位移τ=0），共同状态的valid不变；不留 WAM keypose 泄漏，不再旁路 R。
两臂都计算 WAM 可用性，参考内容干预之后再进入同一路径；闭环运行后的状态差异是干预结果，不要求不同轨迹的观测相等。

开发上限24个工程触发样本/14400控制，无成功率推断；随后冻结实现，在原v9同初态、同seed、同预算上两臂各40 seen/40面板复验（160回合，≤96000控制）。
八个v9未见物体已用于本轮诊断，因此本轮只称“原未见面板复验”，不能再称未调参的新holdout。不得根据结果补抽/删除物理失败。

已保存触发：`outputs/wam-rl-policy-v3/grasp-v9-stat40-20261022-r1/`；临时诊断 `tmp/v10-diagnosis/first-failures.json`。
DOF15/16与10的具体原生名称将从新host回执核对，禁止按XML文本顺序猜。

复现预注册：
```sh
/home2/wyy/miniconda3/envs/graspenv/bin/python src/task/wam-rl-policy-v3/tools/run/register_grasp_repair.py --engineering --output outputs/wam-rl-policy-v3/grasp-v10-engineering-20261022-r1/registration.json
# 正式验证去掉 --engineering；工程运行不是n<40的效果对照。
```
所有运行从新目录启动，保存代码快照、初态、实测PD、原始状态、阶段和物理中止；原安全gate/奖励评分阈值保持。

工程r1回退与修正：统一25mm/0.10rad限速使已知可行seen触发样本无接触，主动SIGINT停止三个自有worker，保存3个完成+3个partial（2813 controls）。R输出是PD偏置，不能当作立即到达的运动学位姿。新过滤仅在**实测**手点/掌mesh进入CAD桌面有限支撑范围且向下侵入超过15mm时介入，正常R命令逐字节保留。原物理安全中止不变。此开发数据不进入正式n40统计。
新host原生DOF名称已确认；旧未见面板：31/40 approach、9/40 close中止；thumb_proximal_pitch_joint 11、thumb_intermediate_joint 25、pinky_proximal_joint 6（42个关节事件/40回合）。39/40不是腕pitch近奇异构型，故不能用Euler奇异性解释大多数中止。
