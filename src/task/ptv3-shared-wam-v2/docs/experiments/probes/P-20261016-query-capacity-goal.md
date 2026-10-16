---
schema: ref2dex.probe.v2
probe_id: P-20261016-query-capacity-goal
experiment_id: P-20261016-query-capacity-goal
date: 2026-10-16
task: ptv3-shared-wam-v2
branch: main
git_commit: 2879b585
claim_id: C1
hypothesis_family: HF-query-capacity-goal
probe_index_in_family: 1
seed_pool: probe
seeds: [221, 222]
decision_changed_if_positive: retain larger or explicitly goal-conditioned query actor for matched native qualification
decision_changed_if_negative: prioritize control-path and objective diagnostics instead of increasing shared control-head width
status: UNCLEAR
run_id: query-capacity-goal-20261016-r1
---

# Inspire query控制头容量、直接goal与当前观测瓶颈

Result: UNCLEAR。五臂固定6000步均COMPLETED。扩宽有中等正向信号，但未达预设
DEV mean>=10%门；直接goal与bypass未给一致收益。不关闭容量或goal假设，不作正式归因。
Decision: 不据此替换native控制器或扩大joint WAM更新；保留width512/no-direct-goal候选。
下一步优先在同头下区分单拍/8拍与腕部/全动作监督。旧raw811D腕部MLP与本轮8×15目标
不匹配，不将其数字差距归因shared、容量或goal。

Class: Decision。服务Objective A的可执行BC substrate，以及Objective B的shared readout定位。
用户明确授权并行控制头128/512×直接goal开关，并同步Shadow BC→RL。此前811D MLP
pooling对照未检验原query头的721→128融合，也不能排除直接计划goal的作用。

## 协议与最便宜的可判别读出

冻结原shared7k WAM，沿用strict TRAIN93680/DEV10376、原固定TRAIN/DEV256及对象面板。
缓存每窗口完整shared tokens和有效mask，不用mean-pool替代query memory。FP32/TF32编码，
FP16保存以限定IO/存储；每个最终actor用留存FP32 tokens重放，首拍XYZ偏差须<0.1mm。
TRAIN-only state87/current721/goal3及8×15 action归一化；goal=已有计划reference_obj[t+24]
位置减当前物体位置，和MLP直接goal完全同义，不读未来实测物体。

五臂：w128_g0/w128_g1/w512_g0/w512_g1，以及w512_g1_bypass。goal编码器均存在，
off仅在标准化后置零；同宽臂model221逐参数同初始化。原state/current融合与两层query
decoder不变，预测完整8×15 geometry-lift q_pd标签。bypass为721→120零初始化线性
直连标准化输出，同宽普通臂也分配但关闭该支路。因此这是增加直接读出能力的Probe，
不能把任何增益完全归因为信息损失，而忽略可训练容量变化。

全部fresh控制头、冻结WAM，half-startup/sequence-uniform同draw222，batch128，
AdamW LR5e-4/WD1e-4、clip1、BF16 actor，6000步固定终点，不挑选DEV checkpoint。
与旧joint7k、原MLP并非单变量比较，五臂内部才做因素归因。width改变也改变参数和
优化难度；报告参数数、TRAIN拟合及DEV分层，不能把wide更好直接称容量假设被正式证明。

主要读出原DEV256首拍XYZ mean/p95与startup，TRAIN256检验拟合，完整对象轨迹面板辅助。
同宽goal on相对off、同goal wide相对narrow分别比较；改善>=10%且startup不恶化才
进入少量native资格评估。无明显信号不自动扩大训练。Probe仅PROMISING/UNPROMISING/UNCLEAR。

## 资源、停止条件和追溯

输出`outputs/ptv3-shared-wam-v2/query-capacity-goal-20261016-r1/`。
完整token缓存<=52GiB；单臂<=2400s、最终权重/optimizer<=0.5GiB，磁盘始终保留20GiB。
缓存GPU2；训练最多并行GPU2/3/4/5（其余臂排队），Shadow可用GPU1，GPU0外部作业不动，
至少GPU6/7保持空闲。远端就绪后可以在专用目录复现相同代码/缓存协议，不改变实验因素。
非有限、缓存覆盖/身份/归一化/量化检查不通过、资源超限立即停止并保留错误记录。
命令、commit、PID、GPU、进度和ETA记录在各run manifest/process/progress。

## 6000步固定终点

| arm | 控制头参数 | TRAIN256 XYZ mm | DEV256 XYZ mm | DEV P95 mm | DEV起步 mm |
|---|---:|---:|---:|---:|---:|
| 128，无直接goal | 890759 | 44.554 | 47.151 | 152.933 | 84.727 |
| 128，直接goal | 890759 | 47.100 | 48.291 | 150.946 | 78.884 |
| 512，无直接goal | 10381319 | 38.242 | 43.304 | 127.660 | 66.301 |
| 512，直接goal | 10381319 | 45.118 | 48.808 | 160.606 | 66.778 |
| 512，直接goal+721直连 | 10381319 | 42.243 | 45.662 | 143.026 | 71.947 |

同宽所有初始tensor逐值相同，五臂first draw一致；draw由同独立rng生成。
无直接goal扩宽：mean/P95/startup分别-8.16%/-16.53%/-21.75%，TRAIN也改善。
直接goal：128 mean+2.42%，512 mean+12.71%；512下TRAIN也变差，不可简单归因DEV过拟合。
721直连相对同宽goal臂mean-6.44%、P95-10.95%，但startup+7.74%，未超过最佳无goal宽头。
宽度同时增加参数与融合宽度，bypass也增加有效读出容量，不能单独证实/否定128D信息损失。

编码458.3s，五臂各547–779s；完整310×768 tokens缓存约46GiB。16条DEV量化重放
最大XYZ偏差0.009–0.0294mm，全部<0.1mm，覆盖限制仍保留。code为2f4760ad（cache）、
2879b585（四臂）、97368b2f（bypass共卡）。final.pt保存模型/统计/optimizer，汇总位于
`outputs/ptv3-shared-wam-v2/query-capacity-goal-20261016-r1/summary/`。

512无goal头绑定原冻结WAM，dcce7a33下4-control/mug/seed0 native工程smoke完成，
server/native exit0、独立执行审计PASS，导出head与原终点、WAM与原7k逐值相同。
输出native-smoke和native-smoke-input/w512_g0.pt；只证明该臂可真实执行，不是完整抓取
资格，也未物理验证goal-on/bypass。因mean门未达，不追加完整三对象闭环。

远端部署原审计因NAS元数据权限和空PointWorld依赖目录失败，历史记录保留；在专用scope
补充依赖副本并改自己的symlink后，Shadow full/pilot与DExplore读者、normalizer/panel
stamps PASS，远端GPU0真实310-token/8×15 forward+控制头backward PASS。
证据`outputs/ptv3-shared-wam-v2/remote-finetune-data-20261016-r1/readiness-recovery.json`。
远端只做工程smoke，未把跨Torch版本数值混入本地五臂，也未验证远端native仿真。

## Limitations / future evidence

与腕部MLP的额外预算差异：两者虽均6000更新，MLP batch512、本轮batch128，抽样曝光
次数（含重复）为3072000与768000，差4倍；初始化/draw seed与数值精度也不同。五臂内部
保持匹配，但不能把跨架构差距归因attention/容量。后续须同时对齐标签、horizon与batch。
用户追问时逐窗复算首拍XYZ标签：`offset+scale*MLP_native_label`与本cache geometry-lift
前三维在全TRAIN/DEV最大差为2.39e-7/5.97e-8米。因此首拍物理XYZ评价含义相同；
训练任务仍不同（当前腕部6D vs8×15，Euler原生命令 vs旋转几何/手指/lift）。

1000步资源读出显示单卡GPU利用率约10%–15%，完整tokens的CPU装载/传输占主要时间。
为避免第五臂等待，允许其与已核验的本实验w128_g0 PID共用GPU2，preflight必须验证
GPU上只有该PID、同uid、同脚本、同实验输出父目录；不向未知占用卡投任务。
不改变batch、draw、模型、数值精度或优化设置；仍仅使用本地GPU1–5，6/7保持空闲。

单seed因素筛选，不是正式Validation；未检验joint WAM更新与其他goal姿态表示。
绕过支路是直接线性读出，未单独对齐有效参数数；若正向，再做匹配容量验证。
首轮FP16量化重放仅覆盖首16条DEV的XYZ，不代表全缓存/旋转已获验证；正式归因与native
资格前需在固定面板补查。所有臂共享同一量化cache，内部配对条件相同。
