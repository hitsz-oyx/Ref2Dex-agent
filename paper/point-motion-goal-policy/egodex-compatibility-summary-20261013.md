# 克隆源码后：这些策略能否直接用于我们的 EgoDex？

日期：2026-10-13。回答用户对“方法相关是否等于能在EgoDex运行”的质疑。
**目前没有一个完整公开策略被验证能直接读取我们的EgoDex并训练。**
此前调研证明相关接口存在，不能扩展为数据/机器人/环境已兼容。

## 已完成的检查

已将五个官方仓库克隆到 `third_party/*-egodex-audit-20261013`，总约16MB。
源码/配置/文本稀疏检出，不下载checkpoint、mesh/media、不递归submodules，
不修改原始数据或远端训练。版本见同目录 [schema.json](schema.json)。

| 代码 | 原版直接用EgoDex训练 | 源码与实际smoke支持的可复用部分 |
|---|---|---|
| Dex4D-Simulation | 否：XArm6+LEAP 22D机器人状态/action，Isaac Gym PPO→DAgger | 原6D paired PointNet类在真实场景点上隔离前向通过；未加载完整policy |
| Dex4D-Vision | 普通视频入口可适配；原米制3D流程不可原样照搬 | 原UV+Z+K反投影在真实窗口通过；完整tracker/depth推理未执行 |
| COIL | 否：UniEnv机器人replay、Panda末端/夹爪状态和7/10D动作 | 原时空goal encoder与PointNet++在真实窗口CPU前向通过；需新手状态/输出/loss/time适配 |
| Im2Flow2Act | 下游否：Zarr机器人action/proprioception；上游生成另需格式转换和语言 | 原轨迹采点helper、backprojection、位置编码实际执行；完整dataset缺依赖且原数据合同不匹配 |
| SimToolReal | 否：IIWA+SHARPA 29D机器人、140D obs、对象SE(3)/抓取bbox | 可借鉴goal推进和抓持训练；未跑完整policy/FK |

完整审计与文件行号：

- [Dex4D/SimToolReal](dex4d-simtoolreal-egodex-source-audit-20261013.md)
- [COIL](coil-egodex-source-audit-20261013.md)
- [Im2Flow2Act](im2flow2act-egodex-source-audit-20261013.md)

## 两个需要分清的问题

**EgoDex作为人类视频预训练源：**它可以提供当前手与未来手、已有几何教师产生的
scene tracks。若加入明确对象/部件mask及可信目标点，可改编goal编码器并训练
未来人手预测。需要保留双手/逐点mask、30Hz共同参考坐标、goal身份和物理时间，
这属于我们的实现，不是原作者robot policy直接训练。

**EgoDex作为机器人策略动作监督：**人体手XYZ不是作者Panda、XArm+LEAP或
IIWA+SHARPA的执行命令。需要我们自身机器人state/action数据、仿真RL、教师
蒸馏或验证过的retargeting/执行映射。给缺失qpos/action补零，只会掩盖缺监督。

## 决策变化

1. 不用“代码已开源”作为启动完整移植训练的理由。当前可核实、最低成本的方向是
   在现有EgoDex/WAM loader上只借鉴配对goal或时空goal编码，并保留已有手监督。
2. 点数少并非唯一问题。COIL默认位置编码是有效槽顺序，不是秒；Im2Flow默认
   位置编码忽略visibility；PointNet++还要求有效点前缀排列。不能仅对齐shape。
3. EgoDex移动相机必须补偿。22窗中遗漏相机变换误差的各窗median平均12.6mm，
   最大窗口p9598.5mm。该差来自坐标解释，不证明原几何教师准确，也不说明原训练
   代码漏了补偿；我们已有point_target本来就在共同当前相机参考系。
4. 原版完整依赖尚不齐。安装环境能解决import问题，不能补机器人标签、目标身份或
   传感器标定。为本轮源码可用性判断无需安装整个仿真/视频生成环境。

本次只完成工程兼容性审计，没有形成方法性能结论、训练收益或Cm收益结论。
下一科学Probe仍需明确motivation、matched control与固定预算。本轮不替换模型，
不启动训练；此前“先做配对编码小对照”的建议保留，但现在有明确适配边界。
