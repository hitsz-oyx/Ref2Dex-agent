# Current Execution Campaign

本文件定义当前机器、资源与操作边界。

这些限制与科研 Mission 分开。

---

## Workspace

允许修改：

`/home2/wyy/oyx_ws/ai_ws`

当前 Ref2Dex-agent 应在该工作区内独立开发。

---

## External Project

外部项目：

`/home2/wyy/oyx_ws/Ref2Dex`

默认：

READ ONLY

允许：

* 读取代码；
* 读取数据；
* 读取 checkpoint；
* 创建指向其中内容的软链接。

禁止：

* 修改；
* 删除；
* 覆盖；
* 在其中生成 cache 或输出。

---

## System operations

禁止：

* sudo；
* apt/system package 修改；
* 修改系统服务；
* kill 未确认属于当前任务的进程；
* 影响其他用户 GPU 任务。

可以创建本项目自己的 conda environment。

---

## Network proxy

需要代理时：

export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897
export HTTP_PROXY=http://127.0.0.1:7897
export HTTPS_PROXY=http://127.0.0.1:7897

---

## GPU execution policy

默认优先使用 GPU，按当前任务选择设备：

* 神经网络训练、微调、重复模型推理和批量模型评估优先使用 GPU，包括离线
  V/Cm 拟合与评估；支持 GPU 加速的仿真采集也优先使用 GPU。
* 默认先用一张空闲 GPU，按实际吞吐和显存需求选择 batch；不要仅因任务属于
  Probe、数据量较小或离线分析而默认使用 CPU。
* 纯文件处理、标签审计和统计计算可使用 CPU。极小 smoke 若 GPU 启动成本
  明显超过收益，或当前存在明确的设备/实现限制，也可使用 CPU；模型计算
  选择 CPU 时，在实验卡或运行记录中说明具体原因。
* 每次任务进入训练或批量模型计算阶段时重新判断设备；历史实验的 CPU-only
  设置不自动延续为新任务限制。当前用户明确限定 CPU-only 的任务遵守该限定。

本策略不改变下面的 GPU、时间和存储上限；使用 GPU 前仍须检查显存和进程。

---

## GPU budget

硬上限：

最多同时使用 4 张 GPU。

使用 GPU 前检查显存和当前进程。

不得停止、抢占或干扰未知进程。


---

## Storage

当前工作产物总上限：

300 GB。

需要主动清理：

* 无用中间视频；
* 可重新生成的大型 debug artifact；
* 重复 cache；
* 无长期价值的临时 checkpoint。

不得删除仍可能作为研究证据的正式运行产物。

---

## Git

默认本地开发。

分支前缀：

`agent/`

不要自动 push 到远程。

每条主要研究路线应存在明确 Git checkpoint。



用户ref14_3在rollingGT-Y和noise gates之后授权actual hand flow→Y ranking
predictor阶段：保持Y/U，比较Direct、严格source-OOF E/I bottleneck与Hybrid，
按anchor隔离并检查shuffle；actualflow仍是事后oracle。离线gate未过不得进入
rolling learned-Y。此次固定Probe已完成但三臂未过70%screen；不改变Mission claim，
不重开execution、desiredflow、PPO/critic/reward，也不外推为E/I无效。


2026-10-06用户明确授权source_e260 frozen PPO critic candidate ranking诊断，
本项替代此前critic实验暂停：固定GT future observation与native PPO reward，
比较Q8=sum discountedreward+gamma^8V与原GT-Y，零模型/策略训练。先在已有
32anchor七候选的同前缀Z上作offline mosaic screen；不把它写成真实rolling收益。
正向才另行冻结真实rolling critic Probe；预算与范围见Task-local实验卡。

2026-10-06用户ref15授权新的training-only GT interaction auxiliary PPO Probe，
替代该路线的PPO暂停：h8真实rollout监督actor z，普通/条件/shuffle/stopgrad四臂，
source_e260恢复，保持原奖励与actor推理结构。单seed短训练、同z独立return诊断及
完整episode评价，资源预算与停止条件见Task卡P-20261006-gt-interaction-aux。

2026-10-10 用户在 ref7_3 开源路线审查后授权最小 reference-tracking control Probe。
本项仅为该 Probe 重开原生 Inspire/Gym 的小残差 PPO 与交互跟踪奖励；保留 owned
e260 作为独立对照，tracker 从零残差初始化。先使用明确声明的 measured robot/object
oracle reference 区分执行控制与 tau retargeting，不外推为 tau-only 或 Cm utility。
一张空闲 GPU，总上限27分钟/2GiB，不突破全局预算；协议与停止条件见
`src/task/consequence-evaluator/docs/experiments/probes/P-20261010-reference-tracking.md`。

2026-10-10 用户要求自主解决 tracker 不稳脱手问题。允许在同一控制路线做有界
wrist velocity feedforward 对照与必要的短 PPO 微调，保留原始任务放置和 measured
teacher holding 上界的不同语义。一张空闲 GPU，新增总上限32分钟/2GiB，协议见
`src/task/consequence-evaluator/docs/experiments/probes/P-20261010-reference-tracking-feedforward.md`。
不改变 Mission 或 Cm claim，不新建分支，不影响外部进程/数据。

2026-10-10 用户确认保留 τ、移除真实 q_ref 与未来物体参考，按 ref7_4 两层
几何retarget/闭环控制思路继续，不引入触觉。允许有界几何拟合、冻结对照与一次
短 PPO 微调；一张空闲 GPU，新上限34分钟/2GiB，协议见
`src/task/consequence-evaluator/docs/experiments/probes/P-20261010-tau-geometry-tracking.md`。
不改变 Mission/Cm claim；GT future hand 仍是明确的执行上界，不代表高层预测已完成。

2026-10-10 用户要求继续解决τ-only不稳。允许同一路线有界finger-command
canonicalization/feedback-coverage Probe，保护已有腕部映射，不引入真实未来q/
物体参考或触觉输入。一张空闲GPU，新增上限25分钟/2GiB；协议见
`src/task/consequence-evaluator/docs/experiments/probes/P-20261010-tau-finger-canonicalization.md`。
不改Mission/Cm claim或原保持/裁剪门槛，不新建分支、不扩展无界参数/seed sweep。

2026-10-10 用户ref8将近期优先级转向H→候选τ。允许一次旧H→τ预测器的冻结
数据/输入分布诊断及有证据支持的最小离线修复或候选覆盖Probe；PointWorld、GT E/I
必要性重复实验与在线selector继续冻结。单空闲GPU，上限16GPUmin/1GiB，协议见
`src/task/consequence-evaluator/docs/experiments/probes/P-20261010-history-tau-proposal-diagnosis.md`。
不改Mission/Cm claim，不新建分支、不扩大仿真/训练sweep。

2026-10-10 持续主线授权下新增生成τ评分迁移/URDF几何审计：冻结纯历史proposal
和T evaluator，不以旧Y冒充新候选的执行标签。单空闲GPU，15GPUmin/1GiB，协议见
`src/task/consequence-evaluator/docs/experiments/probes/P-20261010-generated-tau-score-feasibility.md`。
本轮不训练模型、不仿真、不启用PointWorld或改变Mission/Cm claim。

2026-10-10 持续主线下进入首个纯历史生成τ真实执行Probe：冻结proposal/T/保留的
fingerfit executor，原生几何修正后滚动执行，独立GT/persistence/位移/评分四角色。
单空闲GPU新增15GPUmin/1GiB，smoke120s+单次eval720s；无训练/额外seed/物理fork claim。
协议见`src/task/consequence-evaluator/docs/experiments/probes/P-20261010-generated-tau-native-execution.md`。

2026-10-10 持续主线下隔离生成τ接口与启动覆盖：冻结全部模型，GT在线投影与GT8步
前缀仅为显式privileged诊断，不加入纯H部署。单空闲GPU，新增5GPUmin/512MiB，
两次16env/128步接口诊断及一次修复后的542步纯H跟进，各<=240s（共用5GPUmin上限），
无训练/额外seed；协议见
`src/task/consequence-evaluator/docs/experiments/probes/P-20261010-generated-tau-interface-diagnosis.md`。

2026-10-10 新trajectory-policy主线的48D decoder覆盖Probe：旧Task实验继续暂停，
只读复用其几何/controller。单空闲GPU2，新增8GPUmin/512MiB，一次离线audit<=120s、
一次16env/542步原生执行<=300s；无训练/额外seed/WM，协议见
`src/task/trajectory-policy/docs/experiments/probes/P-20261010-trajectory-decoder-coverage.md`。

2026-10-10 trajectory-policy固定48D前缀拟合诊断：CPU4列线性投影与单空闲GPU2
FK复算，<=120s/2GPUmin/64MiB；无神经训练/仿真/额外seed/WM，协议见
`src/task/trajectory-policy/docs/experiments/probes/P-20261010-decoder-prefix-fitting.md`。

2026-10-10 trajectory-policy固定PCA D48覆盖Probe：单空闲GPU2，<=8GPUmin/512MiB，
手轨迹几何坐标basis拟合/audit<=120s，几何screen通过后一次16env原生执行<=300s；
无高层策略训练/WM/额外seed。协议见
`src/task/trajectory-policy/docs/experiments/probes/P-20261010-lowrank-trajectory-decoder.md`。

2026-10-10 trajectory-policy pose/FF metric D48 Probe：单空闲GPU4（GPU2/3启动前被其他
任务占用，原GPU2 preflight在模型/产物创建前退出），<=8GPUmin/512MiB，
fit+coverage<=120s，screen通过才一次16env/542步原生执行<=300s；无策略训练/WM/
额外seed。此次后停止48D重建搜索，转入可执行轨迹表示的H->c学习。协议见
`src/task/trajectory-policy/docs/experiments/probes/P-20261010-metric-trajectory-decoder.md`。

2026-10-10 持续主线下进入trajectory-policy独立H328->c288 actor初始化：单空闲GPU4，
新增12GPUmin/512MiB，一次2500update监督fit<=240s、一次16env完整执行<=300s、
审计<=120s。旧Task只读复用，保留全部失败行，无WM/额外seed sweep/新分支/push。
协议见`src/task/trajectory-policy/docs/experiments/probes/P-20261010-history-trajectory-actor.md`。

2026-10-10 首轮H actor启动误差后的一次startup-balanced初始化Decision Probe：单空闲GPU4，
新增10GPUmin/512MiB，fit<=240s，启动几何screen通过才一次16env完整执行<=300s及audit<=60s。
同数据/H/D/R/网络/optimizer，不加clock/phase策略输入；此次后停止BC权重搜索。
协议见`src/task/trajectory-policy/docs/experiments/probes/P-20261010-startup-balanced-actor.md`。

2026-10-10 trajectory-policy真实任务PPO Probe：单空闲GPU4，新增14GPUmin/512MiB，
debug smoke<=120s，24update/16env/<=49152实际环境交互train<=480s，单次完整冻结比较<=180s，
audit<=60s。固定D/R、H-only与真实当前几何奖励，无参考奖励/WM；协议见
`src/task/trajectory-policy/docs/experiments/probes/P-20261010-trajectory-ppo.md`。
