# 非语言物体点运动 goal：对当前实现的影响

核对日期：2026-10-13。文献和作者代码事实见同目录
[一手调研](related-works-primary-audit-20261013.md)。本文件区分文献事实、
本地工程检查和下一步推断；未训练外部模型、未验证作者成功率。

## 最值得阅读的路线

**Dex4D** 最贴近灵巧手物体点 goal；**COIL** 最贴近可变点数、可变时间粒度的
轨迹条件 Actor；**SimToolReal** 提供对象位姿目标驱动灵巧手的独立执行方案。
这三者值得优先深入，而不是只检索名字含 flow 的策略。

### SimToolReal：位姿序列与抓取执行

[论文](https://arxiv.org/html/2602.16863v1) 用目标物体位姿驱动灵巧手，
部署额外需要当前物体位姿、本体状态和抓取区域包围盒。策略在仿真中通过
随机目标位姿训练，部署时顺序跟随示范位姿，达标才推进。其抓取阶段有专门
奖励，不能把接近和抓持解释成物体目标信息自动足够。

[官方仓库](https://github.com/tylerlum/simtoolreal) 公开仿真环境、RL训练、
部署和checkpoint下载入口，MIT；本次核对README与论文，未逐行审查执行源码。
其机器人为KUKA iiwa + Sharpa，与我们的执行合同不同，checkpoint不能直接替换R。

[附录A-D2](https://arxiv.org/html/2602.16863v1#A1.SS4.SSS2) 明确说明：
示范从30Hz降到3Hz；删掉开头物体静止的pregrasp段，首目标取物体高度超过
10cm的帧，让策略立即尝试抓起。这个阈值属于其工具/桌面协议，不能照搬为
我们三域的清洗阈值；更可借鉴的是“远期有效物体目标”和“达到后推进”。
其Task Progress是目标位姿跟踪进度，不等于所有功能任务都已完成。

### Dream2Flow：3D物体运动接口与优化控制

[项目](https://dream2flow.github.io/) 将视频中的物体3D运动转换为控制目标。
[官方代码](https://github.com/KDharmarajanDev/Dream2Flow) 已公开物体flow表示、
本地视频输入及PyRoki轨迹优化接口；README仍把RL reward/code列为TODO。
它的上游视频生成可以接受语言，下游flow控制接口无需把语言当作动作条件。
这可以给我们的几何goal提供数据处理参考，但不等价于一个已公开并复现的
灵巧手flow-conditioned神经Actor。

### 不应误判的候选

[FlowPolicy](https://github.com/zql-kk/FlowPolicy) 的flow matching是噪声到动作
的生成过程，并非用户指定物体点流目标。
[GeneralFlow](https://github.com/michaelyuancb/general_flow) 的完整预测接口仍
接收任务描述，再预测可执行flow；其表示/标签提取可借鉴，但不能把完整系统
直接称作无语言目标条件策略。
[FlowBot3D](https://github.com/r-pad/flowbot3d) 预测关节物体可动方向用于操控，
不是任意用户给定多时刻物体目标点的通用执行Actor。

## 本地工程发现：默认endpoint编码未显式保留对应关系

当前 [ObjectGoalEncoder](../../src/task/goal-point-wam/src/goal_point_wam/model.py)（168行）
默认 `endpoint` 路径把start和goal分别变成3D token，加两种类型embedding，
然后作为无point-ID的Transformer memory。它对start组、goal组的独立排列等变；
后续attention读取的是无序memory，因此无法凭索引得知某个当前点对应哪个目标点。
纯旋转/对称对象尤其可能需要这种对应信息。

Tiny CPU smoke使用hidden16/layers1、三点静态goal和同一目标点集合的循环重配：
memory逆排列后最大差 `2.3841858e-7`，null token差 `8.9406967e-8`。
记录在 `tmp/point-motion-goal-research-20261013/endpoint-correspondence-smoke.json`。
这是排列性质检查，不是训练/闭环证据，不能归因现有所有失败。
`start_delta` 路径先计算配对差值，有隐式对应信息，不适用完全相同的结论；
当前机器人control-context的相对物体位姿序列也不能被此检查否定。

**下一步推断：**借鉴Dex4D，把同一点的当前和目标位置或delta放进同一个token，
保留joint permutation invariance；再考虑COIL式时间/顺序条件。不要同时更换
goal来源、World结构和训练数据，否则无法归因。

## 对上一轮视频审计的解释修正

上一轮16点/80%有效率是我们“从原始轨迹构造可靠刚体标签”的保守门。
Dex4D公开了partial-point masking，并报告少于10个可见点时的操控，说明16点
不是所有策略的必需条件。[作者项目](https://dex4d.github.io/)

不能因此把2个点的未知刚体轨迹当成完整6D位姿真值，也不能认为丢失的点
仍有可靠未来标签。应分别讨论：当前观测部分可见、目标指定稀疏、目标真值
缺失、目标身份错标。适合下一Probe的是“可见点数变化+掩码训练”的执行鲁棒性，
不是追求所有视频帧都满足一个固定点数。

## 下一次最小决策实验建议（尚未执行）

1. 先用原生/仿真已知对象变换生成精确配对目标，检验平移和纯旋转；分别检查
   同步重排不应改变输出、打乱配对应能改变条件。暂避开视频标签不确定性。
2. 在相同底座/数据/预算下比较当前endpoint与显式paired目标编码。检查静态、
   null、正确和错配goal；目标敏感性必须与动作方向和实际任务收益共同评价。
3. “顺序达标后推进”与“严格按时间跟踪”是不同goal合同。我们需要先明确采用哪种，
   不能把COIL或Dex4D的waypoint方法当成已验证的时间deadline策略。
4. 若研究执行策略的抓持瓶颈，借鉴SimToolReal/Dex4D的仿真抓取训练和自身访问
   状态反馈。它们依靠RL/DAgger及接触相关奖励，不是只用视频手轨迹BC。

以上为方法建议，不改变MISSION。当前Cm还需在实际策略训练中用matched
Cm-on/off证明收益；Dex4D联合预测robot next-state的辅助World不自动满足
我们goal-blind、action-conditioned hand-object Cm的因果合同。
