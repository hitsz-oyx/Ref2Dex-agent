# PTv3-Shared-WAM v2 当前状态

## 2026-10-15 10:21 CST 当前参考/表面几何信息 Probe

用户允许当前参考目标及误差，排除新增未来参考。GPU0启动四臂matched R-only
Probe：baseline/current_reference/current_geometry/both，各3000更新、batch128，
统一从主训练step500的R快照开始，GT τ、数据抽样和新增encoder容量相同。新增输入为
同tick参考q/物体相对误差24D，以及11手点到CAD最近点的当前位移33D；没有新增接触力
或16-body动力学测量。独立审查和3项已知值/未来不变性测试通过。此为oracle τ下的
离线预测诊断，不能用作抓取或部署性能结论；四卡主训练保持运行。
详见[实验卡](experiments/probes/P-20261015-R-current-information.md)。

首轮3000步已完成并独立复算。baseline/reference/geometry/both 的dev腕部XYZ
mean/p95分别59.43/219.86、60.85/199.81、58.65/209.09、59.67/187.94mm。
两输入合用尾部改善14.5%，平均不变，按预设门为UNCLEAR；四臂曲线仍在下降。
据此启动唯一一次预算延长：r3两卡并行、同初始R重新训练每臂12000步；不再自动延长。
GPU0 baseline/geometry，GPU1 reference/both，每worker900s/150MB，合计300MB。

最终12000步四臂已全部完成，停止延长。selected mean/p95分别为baseline49.06/168.37、
reference52.78/187.03、geometry49.99/162.11、both54.42/176.04mm；均无整体收益，
维持UNCLEAR和原生产输入。独立原始PD复算、前3000轨迹一致性、选点及输入敏感性
检查通过；早期both尾部优势未保持，不否定尚未采集的接触/刚体动力学信息。

## 2026-10-15 10:01 CST 远端80k权重的四卡q_PD微调已启动

用户明确要求无warmup、q_PD监督，并改用四卡/更大batch。远端科学step80000已取回，
严格加载parent与其原始数值统计；R/optimizer从头初始化，预测τ动作损失更新WAM+R，
GT-τ anchor仅更新R。科学run `ptv3-shared-v2-robot-finetune-20261015-r2` 已在本地
GPU4/5/6/7启动，128/rank/global512、12000更新，supervisor1436540每60秒监督，
代码dccd77ee。双卡r1按用户新要求正常停止，保留原始FAILED/signal15回执，不作为负证据。

大batch工程门发现原RL660物体跟踪准入仍包含手漂移超过6m的窗口，导致预训练固定
体素域越界。保留原corpus/voxel origin/坐标/标签，新增显式observed-fixed-grid.v1：
TRAIN110777→110469，308窗跨6序列排除；DEV11908不变。逐序列排除tick与旧新索引
写入run的encoder_workspace_admission.json，R归一化和sampler均绑定新TRAIN成员。
这只验证当前encoder输入域，不代表剩余窗口均为成功示范。17项测试、独立只读审查、
四卡global512真实四步/评估/保存通过；工程显存约6.5GiB/卡，不形成学习/抓取结论。
运行协议和首段资源观测见对应实验卡与run的startup_observation.json。

10:03:44 CST 已到step80/12000、loss有限。30秒GPU4/5/6/7平均利用率约87/90/88/91%，
物理显存峰值约6.6GiB/卡；稳定0.951秒/update、538样本/s，剩余纯训练约3.16h，
dev/保存开销另计。首段无资源异常，尚未到step500首次dev，不作收敛或抓取判断。

用户要求平台检查：step2220快照下仅4次固定dev；1000→2000总损失5.202→5.262，
pred-τ action0.4897→0.4361、GT anchor0.4838→0.4282，动作仍改善约11%。由总损失
减去两项动作得到的forecast4.229→4.398回升，抵消动作收益。rank0训练日志500步分组
均值4.241→3.568→3.000仍下降（不同抽样，不与dev绝对值直接比较）。因此是总dev
目标早期平台/回升，不是整体训练或动作平台；保持原训练，至更多dev点复核。best按
总损失仍为step1000，不能据best不更新断言动作无进展。图与JSON在主run的
plateau-step2220.{png,json}。尚无物理抓取平台结论。

## 2026-10-15 q 三维合同与远端 WAM 适配

RL660 reference-q raw replay 与 nearest-live branch follow-up 均通过独立
command/PD/clock 审计，但 q_ref 直接执行仍为 UNPROMISING 信号（全程手/物体
误差大、held=0；整体 audit 因 native 短回放门保持 UNCLEAR），因此没有把 q_ref
改成监督标签。新 v2 adapter 位于 `src/task/ptv3-shared-wam-v2`，每个窗口显式
保存 `q_ref_next`、`q_pd`、`q_real/q_real_next`，动作损失只使用 q_pd 的
continuous-lift 编码；本地 checkpoint 的 4-step engineering smoke 已通过，
13 项 v2 合同测试通过。远端 v2 80k checkpoint 尚未能读取：反向 SSH
`127.0.0.1:22023` connection refused，直连 `100.64.0.23:22` timeout；因此
两卡科学微调尚未启动，不将本地 smoke 冒充远端结果。GPU4/6 保留给远端父权重
到达后的两卡启动，GPU5/7 既有 geometry-lift 已自然完成 step19000/19000。
适配器实现 commit 为 `d4b011a4`，新增的
`tools/run/launch_robot_finetune.py` 只监督和终止自己创建的 torchrun 进程组，
每60秒记录 progress/GPU/磁盘，8GB本run产物预算与20GiB可用空间安全线均独立于
已由用户解除的旧400GB总上限。远端状态缓存仍停在 step33520，不能据此伪造
最新 parent；需等 SSH 恢复后按 registry、稳定 hash 和 checkpoint 元数据验收。
04:07 CST 复核显示 `tailscale whois 100.64.0.23` 为 `peer not found`，本机
Tailscale netcheck 亦无可用 UDP/DERP；因此当前阻塞是远端网络/Windows 转发状态，
不是本地微调代码或 GPU 资源问题。

2026-10-14：本地1000步Probe已完成，局部PROMISING。固定三域dev的hand_all24
从29.72/123.10/63.58降到23.83/87.16/55.14mm；scene在GRAB/ARCTIC改善，
OakInk2略差。best/latest为step1000，不能据此形成抓取或Cm增益结论。
模型98,591,142参数，geometry11,092,000、shared85,054,464。

远端GPU0/2已启动80k长训：`pretrain-main3-remote-80k-20261014-r1`，代码1650ffce。
每卡128/global256，从seed136重新初始化，不迁移工程权重或静默扩展旧1000步配置。
最多48h、run8GB，384窗固定dev（每域128）每1000步评估；每2000步恢复点，
保护best/latest与20k里程碑。监督PID1718152、torchrun1718160、worker1718169/1718170。
目录 `/home/wbcd/workspace/oyx_ws/wam-v2-20261014/repo`；graspenv为Torch2.0.1/cu118。
复用只读NAS输入与原horizon统计；已通过双卡四步及完整模型step4→5恢复更新。
启动固定step460快照约0.682秒/update、375.5样本/s，纯训练剩余约15.1h；dev/保存另计。
30秒GPU0/2平均利用率96.8%/96.3%，物理显存最大29107/23733MiB，尚无训练后dev。
本地预算watcher3838855在393GB主动停止本run，初始合计344.48GB，共享硬限400GB。

长训/恢复须使用修复代码d0288294，并在启动shell先执行 `ulimit -n 65536`。
这是进程软上限；hard1048576与系统配置不变。默认1024会在mmap缓存约1018FD时失败。
修复还处理Torch2.0 fused AdamW恢复的CPU step标量；远端最小GPU回归及完整模型
step4→5恢复更新均通过，模型/optimizer/RNG恢复、有限梯度和实际参数更新已核对。
本地Torch2.4 GPU回归与13项v2合同测试通过；原有61项新旧测试记录保持。
未改模型、数据、优化配方或旧Task的5/7微调，不操作远端1/3上的其他任务。

预检报告：`outputs/ptv3-shared-wam-v2/remote-preflight-20261014-r1/summary.json`。
短工程检查不保证长期吞吐/利用率；正式长训仍需固定预算、完整配置与持续资源观察。
详见[实验卡](experiments/probes/P-20261014-ptv3-shared-v2-pretrain.md)。
