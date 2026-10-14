---
schema: ref2dex.probe.v2
probe_id: P-20261014-ptv3-shared-v2-pretrain
experiment_id: P-20261014-ptv3-shared-v2-pretrain
date: 2026-10-14
task: ptv3-shared-wam-v2
branch: main
git_commit: 1650ffce
claim_id: C1
hypothesis_family: HF-ptv3-shared-v2-pretrain
probe_index_in_family: 1
seed_pool: probe
seeds: [136]
decision_changed_if_positive: continue bounded v2 pretraining if physical hand and scene errors improve on the fixed three-domain dev panel
decision_changed_if_negative: audit input routes and normalized target semantics before increasing compute or rejecting v2
status: RUNNING
run_id: pretrain-main3-remote-80k-20261014-r1
---

# 联合轻量PTv3与大shared的horizon标准化预训练

Result: RUNNING（远端80k阶段）。本地1000步局部PROMISING，远端双卡与恢复预检通过；按用户明确要求启动GPU0/2长训。
Decision: 用户已明确要求直接开始训练；采用预检通过的模型/配方，在独立远端run从头训练80000更新，保留本地Probe和工程checkpoint。

Class: Decision. 服务于三域交互表示与最终机器人适配；问题是减少几何参数、让hand进入
PTv3并使用统一horizon统计后，该v2组合是否能稳定学习物理手/物体预测。
1000步最便宜地区分可训练的候选与先修实现/优化的候选，不直接进行长训练或正式消融。

## Decision Note

- 当前旧660微调在GPU5/7继续，不修改其模型/训练源码。v2在当前main独立目录开发；
  用户没有授权新分支，故不新建分支。用户原会话批准的模型方向不改变MISSION claim。
- 输入以固定物理尺度处理；输出统计仅来自TRAIN共享4096窗。手编码前后双路；
  22手点独立回读。参数从头初始化，不强行迁移维度不同的旧PTv3。
- 先统计CPU数组（无神经计算），真实GPU四步接线/负载/恢复检查；通过后2卡4/6，
  每卡128/global256、1000次optimizer更新，最多2小时、单run8GB、共享总量400GB。
  预计10–40分钟，以实测吞吐更新。与旧训练合计4卡，不影响其他用户GPU0–3。
- 任一非有限量、梯度合同失败、外部GPU进程出现、源码变化、资源超限立即停止本run。
  不杀未知进程，不覆盖旧权重。只有实施结果具备改善信号再考虑更长预算。
- 成功条件为三域hand_all24与scene_all24物理EPE相对step0多数域改善，并且没有
  未解释的归一化/有效性错误。此门用于选择是否继续，不是正式显著性或因果归因。
  若某域恶化先核对幅度和mask，不把组合短训负结果升级为核心假设失败。

## 固定协议

见[架构](../../user/架构.md)与[配置](../../../configs/pretrain_probe.json)。
数据：既有pointworld-main3-sharedstats数据，无新增下载；原完整序列划分和采样比例。
TRAIN统计run `horizon-stats-main3-20261014-r1`，seed18（debug统计抽样，不是训练重复），
按mix sampler采4096窗，场景/手分别24×3 mean/std，std floor1mm，保存train_indices。
输出：`outputs/ptv3-shared-wam-v2/`，数据与checkpoint不复制到文档目录。
训练seed136、rank初始化136/137；dev seed138、96窗、每域32窗。rank不是科学重复。
AdamW，geometry/shared均1e-4、weight_decay=.01、clip1；不增加额外重建损失。
bf16、TF32、shared激活重计算、有界4workers/rank、prefetch2、每worker/source缓存128。
每250步验证，保存所有best与500步恢复点并保护best/latest，最多保留best+recent与最终。
metrics复用已审查的物理EPE与goal-only线性插值分组；normalized objective不与v1直接比较。

## 工程与运行记录

- 初始实现提交da694f1f；13项CPU测试覆盖mask/NaN、horizon24×3 roundtrip、同voxel
  密度不冲淡手分支、22语义槽位、goal到双头、梯度隔离与v1 checkpoint拒绝。
- 第一版真实三域四步 `engineering-main3-20261014-r1` 完成，双头均到达PTv3/shared/goal，
  hand编码经PTv3路径梯度非零；该候选几何支路33,988,832参数，超过设计预算。
  保留原始运行，随后把PTv3调整为11,092,000参数；shared保持85,054,464，
  最终98,591,142参数。这是根据实际参数统计修订工程配置，不是方法负结果。
- 最终配置 `engineering-main3-final-20261014-r2` 真实三域四步完成，hand/scene均到
  PTv3、shared、hand encoder、goal；hand encoder经PTv3梯度非零。61项新旧测试通过。
- `engineering-main3-partial-20261014-r1` step2→`engineering-main3-resumed-20261014-r1`
  step4，与连续r2相比RNG一致；max参数差3.50e-4，固定dev预测hand/scene最大差
  0.01515/0.00117mm。`engineering-resume-audit-20261014-r1/summary.json` pass=true，
  不声称稀疏GPU轨迹逐位相同。
- 单卡 `engineering-batch64-20261014-r1` 四步完成，约0.79秒/update，峰值分配7303MiB。
- 首次DDP工程启动在12:11:56因GPU4被另一会话的UniDexGrasp回放占用而正确退出；
  对方r7的gpu-started记录为12:11:49（launch PID3435226）。未杀对方进程，未开始训练。
  两个最小启动复现均无rank自误判；回放自然结束后重启，不修改占用保护。
- `engineering-ddp2-batch128-20261014-r2` 两卡每卡128/global256四步完成，
  稳定约1.31秒/update、196样本/s，rank0峰值分配约13.6GiB，物理显存约19–20GiB。
  比双卡64的单卡吞吐外推更高，故将Probe选为global256，保持学习率和1000更新。
  四步负载不保证所有未来batch峰值；OOM/外部进程出现仍停止保存失败记录。
- 以上均为工程证据，不代表预测质量或抓取收益。
- 正式Probe run `pretrain-main3-probe-20261014-r1`，运行代码95c1436a，监督PID3444957，
  GPU4/6，global256、1000更新。输入、worker PID、统计、配置与初始化由manifest.json记录。
  旧Task660大模型训练继续，不把它的权重或optimizer混入v2。

## 启动观察（固定step40快照）

`startup_gpu_observation.json`保存连续30秒采样：GPU4/6均值83.2%/84.6%，物理显存
峰值21423/20789MiB。rank0训练峰值分配14961MiB；最近20更新约1.067秒/update，
global256约240样本/s，剩余960更新的纯训练ETA约17分钟，含dev/保存预计18–22分钟。
数据workers完成初始化后loss/梯度有限，尚未到step250，不以训练loss报告效果。

固定dev step0的hand_all24为OakInk2/GRAB/ARCTIC 29.72/123.10/63.58mm，
scene_all24为3.59/69.99/51.35mm；后续只与相同固定面板的物理EPE比较。
三域初始量级不同，normalized objective不作为v1/v2跨架构收益结论。
监督器每30秒记录GPU，保护best/latest并在deadline/源码变化/外部进程等边界停止。

## 1000步Probe完成

运行代码95c1436a，1196.16秒，监督器exit_code=0；best/latest均为step1000。
固定96窗dev、每域32窗，与step0使用同一面板。下表为物理EPE，单位mm：

| 域 | hand all24：step0 → 1000 | scene all24：step0 → 1000 |
| --- | ---: | ---: |
| OakInk2 | 29.72 → 23.83 | 3.59 → 3.69 |
| GRAB | 123.10 → 87.16 | 69.99 → 41.04 |
| ARCTIC | 63.58 → 55.14 | 51.35 → 33.18 |

normalized dev objective为5.578→2.068。三域hand均改善、scene多数域改善，满足本卡
预先定义的继续投入门，标为局部PROMISING；OakInk2 scene略差，仍需观察静止点漂移。
单seed/小dev不支持显著性、v1/v2优劣、架构归因、真实物理推理或抓取有效结论。

## 2026-10-14 远端训练工程预检 Decision Note

用户要求远端大规模v2，指定GPU0/2；先确认能否正常训练。只做有界工程检查，
不在本阶段自动启动长训练或改变统计/数据划分。初次GPU2占用时等待，用户随后
明确告知卡已准备好，重新检查空闲后使用0/2，不干预1/3上的其他任务。

- 主机deepcybo-4-4090-48，沿用已有wbcd身份/严格SSH主机键；不新增身份。
- 独立目录 `/home/wbcd/workspace/oyx_ws/wam-v2-20261014/repo`，复用只读NAS
  三域data-remote；不修改旧远端项目、环境或输入数据。固定代码7c256e8a，
  PointWorld子模块05484826。horizon统计只更换data路径，数值完全保留。
- 远端graspenv为Torch2.0.1+cu118、flash-attn2.5.9.post1、spconv-cu118 2.3.6；
  与本地runtime不同，必须通过实际forward/backward/DDP/save/load检查，不能只依赖import。
- `engineering-remote-ddp2-batch128-20261014-r1`，GPU0/2、每卡128/global256、4步，
  debug seed17；最多900秒、整个新scope最多4GB。监督PID1669622；无科学训练结论。
- GPU冲突、源码变化、非有限量、OOM、deadline或存储越界即停止所属进程组。
  本地660微调继续在5/7；合计4张本项目GPU，在共享6张上限内。
- 详细部署/runtime/运行回执保存在本地 `tmp/wam-v2-remote-preflight-20261014/`，
  远端scope的deployment.json与repo/tmp/<run_id>/process.json保存实际路径、PID与边界。
  预检结果和恢复兼容修复见下节；工程结果不升级为科学结论。
- r1在rank1第三步取数据时退出，Errno24 Too many open files；前两次更新有限，
  没有得到checkpoint，不算模型/显存负证据。远端默认RLIMIT_NOFILE为soft1024、
  hard1048576。对同一rank1/worker0的step1/3抽样做CPU-only文件读取最小复现：
  soft1024在1018个FD附近失败，soft65536正常读完，峰值1571个FD。
  只把本次监督/训练/worker继承的进程软上限改到65536，hard与系统配置不变。
  相同模型/数据/batch的独立r2复检；旧失败日志与复现结果保留。

## 远端预检结果与恢复兼容修复

`engineering-remote-ddp2-batch128-20261014-r2` 在7c256e8a代码上完成4步，exit0，
model/gradient finite、真实PTv3/shared/双头接线与checkpoint保存通过。
每卡128/global256，稳定步骤2/3/4约0.796/0.790/0.791秒，约324样本/s。
rank0峰值分配14270MiB，末端物理显存GPU0/2约20483/21891MiB（48GiB卡），
scope含代码与完整checkpoint约1.53GB；SSD仍余约279GB，NAS约52.3TB。
四步吞吐只作工程容量估计，不能保证长训持续利用率、I/O与所有batch峰值。

随后完整checkpoint的继续更新检查捕获第二个runtime兼容问题：Torch2.0.1
`Optimizer.load_state_dict`对CPU映射的step标量不做设备迁移，而fused AdamW要求
该标量位于CUDA。最小两参数GPU回归在legacy路径稳定失败（step_device=cpu），
同一测试使用restore_optimizer后通过（step_device=cuda:0，step1→2）。
修复提交d0288294只修正fused/capturable恢复的step设备，不改变模型、统计或配方；
本地Torch2.4.1同一回归通过，13项v2合同测试通过。生产v2恢复入口已调用该函数。

独立完整模型load/optimizer/RNG恢复与下一步更新通过：step4→5，RNG复核一致，
loss3.1654、clip前梯度范数5.0623有限，参数实际更新；丢弃该工程模型，不另存科学checkpoint。
远端ready代码为d0288294。所有原始失败
与成功记录保留，完整数值报告位于本地 `outputs/ptv3-shared-wam-v2/remote-preflight-20261014-r1/`。
远端仅做工程预检，未启动大规模长训；预检完成后GPU0/2自然释放，不操作其他任务。

## 远端80k阶段：启动 Decision Note

- 用户在预检完成后明确要求“直接开始训练”。本地1000步已通过继续投入门，远端
  global256、真实梯度/保存/恢复与step4→5更新通过；核心问题/claim不变。
- root选择沿用原大规模预训练的80000更新预算，每卡128/global256，geometry/shared
  lr仍为1e-4，AdamW/clip/bf16/TF32/模型/数据/数值统计不变。从seed136重新初始化，
  不迁移工程权重；本地1000步Probe保留。不同Torch版本与执行不是独立seed证据。
- 选择从头完整运行，避免把只设1000步的旧checkpoint配置静默扩展或混用跨runtime
  optimizer；重新覆盖1000步成本约十余分钟，小于跨机器迁移大型checkpoint与新增恢复
  语义的成本。该80k阶段仍是同一研究问题的Probe，不另建experiment ID或分支。
- dev固定为384窗（每域128），seed138，step0与每1000步评估；较本地96窗扩大面板，
  因而不能把两阶段dev绝对值直接相减。每2000步恢复点，保留20k里程碑、best与最近
  恢复点；保护best/latest，不覆盖原权重。使用[remote80k配置](../../../configs/pretrain_remote_80k.json)。
- run_id=`pretrain-main3-remote-80k-20261014-r1`，远端GPU0/2（48GiB/卡），预估18–24h，
  硬上限48h；run产物8GB。每rank2个spawn worker、prefetch2、每source缓存128，
  进程soft nofile65536。旧本地5/7微调继续；远端1/3现有任务不触碰；按保守计数最多6卡。
- 启动前本地outputs+tmp约329.19GB；既有远端训练scope与新scope合计另计。
  加新run8GB预留后仍低于共享400GB。监督器独立于SSH会话；本地预算watcher每120秒
  合计本地与两个已声明远端scope，393GB主动停止本次已验证身份的远端监督器，
  留7GB余量；远端同时独立执行run8GB/deadline/冲突/源码漂移/无进展检查。
- 若改善则保留最佳与里程碑供后续robot适配；若dev恶化/平台先看固定物理EPE，
  不以训练loss升级结论。OOM、非有限量、外部GPU冲突或资源越界停止本run并保留已有
  完整恢复点，禁止杀未知进程。正式matched比较、多seed与抓取验证仍延后。

## 远端80k启动回执

运行代码1650ffce，监督PID1718152、torchrun1718160、rank worker1718169/1718170。
run为 `pretrain-main3-remote-80k-20261014-r1`，no-file soft65536/hard1048576。
独立监督器已在GPU0/2启动；初始工程梯度审查通过，hand/scene均进入PTv3/shared/goal，
hand encoder经PTv3路径非零。固定step460快照：loss/梯度有限，最近0.682秒/update，
global256、375.5样本/s，纯训练剩余15.06小时；后续dev、保存和I/O开销另计，
因此启动前18–24小时只作为保守预估，不是已经测得的耗时。
rank0峰值分配15447MiB；30秒逐秒采样中，GPU0/2平均利用率96.8%/96.3%，
物理显存最大29107/23733MiB（约28.43/23.18GiB），包含allocator缓存。
原始采样与进度保存在本地run目录的 `startup_gpu_observation.json`。
本快照尚未到step1000首次训练后dev，不从单个训练loss判断80k阶段质量；
本地1000步96窗与远端384窗面板不同，也不直接比较两阶段绝对分数。

本地预算watcher PID3838855，每120秒合计本地outputs/tmp与两个已声明远端scope。
初始合计344.48GB，低于393GB主动停止线与400GB硬上限；最多8GB新run预留也在界内。
本地回执与当前远端快照：`outputs/ptv3-shared-wam-v2/pretrain-main3-remote-80k-20261014-r1/`。
远端进度在对应scope的repo/outputs同名目录，checkpoint不回拷占用本地磁盘。

## Limitations / future evidence

单seed、1000更新只用于可训练性与继续投入决策，不能形成机器人抓取/Cm收益结论。
三项结构改动与归一化一起改变，没有匹配消融，不能把改善归因某个模块。
统计来自4096个TRAIN抽样窗，并非全数据精确统计；缺少独立采样误差评估。
后续正式比较需要固定物理指标、相同数据预算/初始化条件、多seed以及robot/native验证。
旧Task的微调继续使用旧统计与checkpoint；v2目前没有robot迁移执行入口。
