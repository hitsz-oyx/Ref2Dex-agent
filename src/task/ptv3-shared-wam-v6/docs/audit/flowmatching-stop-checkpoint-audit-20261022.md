# FM 长训停止、checkpoint 与曲线 CPU 审计

结论：`flowmatching-overnight-20261022-r2` 已停止，GPU4/5/6在独立检查时各为6MiB、0%。
**主训练权重确实存在，最新保留的是 step11000，不是旧 trial128。**
owner 提到的11020有训练日志，没有在声明扫描范围发现该step权重；实际最后日志已到11380。
因此从最新保留权重到最后确认更新，至少有380 updates / 109440次样本曝光没有持久化。

本次没有信号、STOP文件、restart、GPU推理或训练，也没有修改原训练manifest。机主通过
Hermes在12:29发送SIGINT是owner提供的操作背景；训练日志独立证明了SIGINT与worker退出，
**不能从日志独立认证信号发送者身份**。本审计采用 long-running-tasks 监控约定，只读进程与CPU产物。

## 身份、停止与资源证据

- 原run：`outputs/ptv3-shared-wam-v6/flowmatching-overnight-20261022-r2/`。
- 冻结训练代码：`0917f47ebb74173a1f43b10f3e3a21c5689ae0b6`；实际入口：
  `tmp/flowmatching-overnight-20261022-r2/frozen/src/task/ptv3-shared-wam-v6/tools/run/train_pretrain.py`。
- main argv为三rank torchrun、local96/global288、resume自flow-trial/step000128.pt，
  原目标30000或累计43200s；policy训练运行Python3.8.20/Torch2.4.1+cu121/CUDA12.1、strictFP32。
- `training.log`：**2026-10-22 12:29:11.355672 +08** torchrun收到SIGINT；同秒将SIGINT转发
  1947527/1947528/1947529；三个rank均KeyboardInterrupt，torchrun1947491报告signal2。
- `process.json` mtime **12:29:16.235870 +08**，状态FAILED，training child exit_code1，result={}。
  这是中断路径的进程状态，不是已经证明数值发散；栈在backward/clip被中断，不是非有限异常。
- `training/result.json` 不存在；`training/progress.json` 仍为RUNNING/step11380，
  mtime **12:28:49.015708 +08**。它是最后进度快照，不是当前活跃状态。
- 独立stop receipt时间 **12:37:46.605578 +08**：11个从原stage记录/停止日志识别的PID均不存在，
  包括supervisor1926227、main torchrun1947491、三main workers；进程表无该run的trainer/supervisor。
  GPU4/5/6各 used6MiB/free24247MiB/util0%，对应GPU UUID无compute app。
  这是检查当时的释放证明，不是以后资源占有权。

完整命令输出、PID列表、GPU UUID与信号行保存在
`outputs/ptv3-shared-wam-v6/flowmatching-stop-audit-20261022-r2/stop-receipt.json`。
原始process的GPU字段仍是停止前快照，不能拿它否定当前释放检查；原文件保持原样。

## 所有实际保留权重：逐payload确认

扫描原run全部权重/软链/partial，并扫描本项目`outputs/`下2396个checkpoint路径，以
解析到该run的软链、flow命名或精确owner step文件名识别关联候选。合计15个目录项，
包含原run8个普通权重、6个软链，及额外保留下来的FM8500本地copy；对应9个唯一文件payload。
**全部实际CPU读取schema/step，不根据文件名或文件大小猜step。**

下面时间均为真实目标文件mtime、Asia/Shanghai、2026-10-22；软链自身mtime另记后表。
所有权重schema均为 `ref2dex.ptv3-shared-wam.v6`。

| 路径（除最后一行外相对原run） | payload step | 字节 | 目标文件mtime +08 | 用途 |
| --- | ---: | ---: | --- | --- |
| readiness/step000001.pt | 1 | 1217083169 | 01:20:05.940320 | 工程readiness，非主长训 |
| readiness/step000008.pt | 8 | 1217083169 | 01:20:28.588444 | 工程readiness，非主长训 |
| strict-recovery/step000009.pt | 9 | 1217108513 | 01:21:36.128812 | 严格恢复工程阶段 |
| strict-recovery/step000010.pt | 10 | 1217108513 | 01:21:40.504836 | 严格恢复工程阶段 |
| flow-trial/step000020.pt | 20 | 1217116001 | 01:23:19.449372 | trial中DEV objective best |
| flow-trial/step000128.pt | 128 | 1217116001 | **01:30:13.951571** | main初始化来源，不是停止时权重 |
| training/step010500.pt | **10500** | 1217116001 | **11:31:29.496499** | 主训练周期checkpoint |
| training/step011000.pt | **11000** | 1217116001 | **12:04:02.155160** | 最新保留主训练checkpoint |
| outputs/ptv3-shared-wam-v6/identifiability-audit-20261022-r1/flow-step008500.pt | **8500** | 1217116001 | **09:37:30.285244** | 独立诊断保存的本地FM copy；原training/8500已被正常retention删除 |

FM8500的关联还有已存在的`identifiability-audit-20261022-r1/checkpoint-audit.json`：
step8500、flow_matching=true、GT2048 geometry及精确normalizer/finite审计记录；本次另行CPU
读取其payload，初始化/恢复绑定与同一run链一致。它不替代最新11000，也不是11020。

所有软链已解析，**没有任一软链指向11020或11380**：

| 软链（相对原run） | 目标 | 软链mtime +08 |
| --- | --- | --- |
| strict-recovery/best.pt | strict-recovery/step000009.pt | 01:21:40.504836 |
| strict-recovery/trajectory-best.pt | strict-recovery/step000009.pt | 01:21:40.504836 |
| flow-trial/best.pt | flow-trial/step000020.pt | 01:30:13.955570 |
| flow-trial/trajectory-best.pt | strict-recovery/step000009.pt | 01:30:13.955570 |
| training/best.pt | **flow-trial/step000020.pt** | **12:04:02.155160** |
| training/trajectory-best.pt | **strict-recovery/step000009.pt** | **12:04:02.155160** |

最后两条软链虽然mtime较新，**目标权重仍旧**。它们保存的是继承的DEV最优selection，
不是latest别名；仅查看`best.pt`或软链mtime会误认长训没有保存/只剩01:30权重。
本次没有发现原run的`.pt.part`或停止后final checkpoint。

扫描边界：没有连接远端机器、没有访问凭据或遍历未知外部挂载。关联扫描依据上述路径/
软链/命名和已知诊断copy；**不能排除未知位置、任意改名且无provenance的外部export**。
原run及项目outputs的枚举、inode、所有真实mtime、target、payload与错误字段见
`checkpoint-inventory.json`，不将“未发现”写成全世界不存在。

## 11000不是旧trial权重，11020不是最新完成状态

CPU `mmap + FakeTensorMode` 读取11000：

- step11000、world_size3、global_batch288，累计elapsed38589.511590s；
- 626个model state项，其中49个flow参数项；617个optimizer state项、3rank RNG；
- model tensor storage合计405416396字节；完整checkpoint1217116001字节包含optimizer等。

另用真实CPU mmap只读取3个小sentinel tensor，不materialize巨型optimizer，均finite且
**不等于trial128**：

| tensor | shape | trial128 RMS | 11000 RMS | 两者RMS差 |
| --- | --- | ---: | ---: | ---: |
| flow_future.output.weight | 222×256 | .00231286 | .01886644 | .01745046 |
| flow_future.output.bias | 222 | .00048198 | .00317950 | .00303687 |
| hand_future.mlp.3.weight | 3×256 | .06207968 | .06464984 | .00320063 |

这验证最新payload含实际变化的模型权重，不是仅改文件名的trial128。只检查了这些sentinel
finite，**不是全optimizer数值/RNG严格恢复审计**；本轮不启动resume。

- owner观察点11020：有log row，elapsed38668.969409s（10.741380h），loss .640853669。
  声明扫描范围没有payload step11020。
- 最后明确完成并写日志的是**11380**：elapsed40077.662768s（11.132684h），loss .661000580。
  日志每20步写一次；SIGINT发生时可能还有未日志化完成更新，本次不编造精确最后optimizer step。
- 相对最后保留11000，**至少380updates ×288=109440次样本曝光**未保存；这是重复draw曝光量，
  不是独立数据量，更不是机器人环境交互步数。11000 checkpoint elapsed到最后log差
  **1488.151179s≈24.80min**；真实文件mtime间隔约24.78min。

## 曲线：8.7h→10.7h的原始数值与contact口径

本轮读取主训练**566条logged batch**、**22次固定DEV评估**。横轴采用原始累计训练elapsed，
包含前128步阶段，不等同纯GPU kernel时间或supervisor总wall。首先精确复现owner观察：

| 主训练step | 累计小时 | TRAIN总loss | TRAIN contact | TRAIN geometry_contact |
| ---: | ---: | ---: | ---: | ---: |
| **9000** | **8.711117** | **.689010152** | .243202806 | .289526790 |
| **11000** | **10.713360** | **.661275186** | .252237797 | .294094771 |
| 11020（owner观察点） | 10.741380 | .640853669 | .244334698 | .286622643 |
| **11380（最后日志）** | **11.132684** | **.661000580** | .239646703 | .282292217 |

因此“8.7h .689→10.7h .661”可在step9000→11000原日志精确找到，但不能用两个随机batch
证明收敛或严格contact不变。额外固定窗口（各中心±15min，未挑最好batch）为：

| 时间窗中心 | logged batches | 总loss均值 | contact均值 | geometry_contact均值 |
| ---: | ---: | ---: | ---: | ---: |
| 8.7h | 29 | .684049048 | .251234009 | .300649403 |
| 10.7h | 22 | .656991312 | .249398239 | .293192435 |

总loss约降3.96%，contact仅降约0.73%、geometry_contact约2.48%。**后期contact变化弱/缓慢**
是可保留的描述；“全程完全flat”“没有任何训练改善”不符合整个曲线。窗口重叠相关，
29/22不是独立训练重复，没有统计显著性或普遍饱和结论。

固定DEV进一步区分训练loss与物理几何：下表为配置的GRAB source1、固定32-window
DEV面板，prefix8以各horizon EPE均值再对8步取均值；不是当天另一条3356-window审计。

| step | GRAB几何contact MAE | GRAB contact loss component | hand prefix8 mm | object prefix8 mm | 全source macro objective |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | .147233333 | .309548881 | 17.0514 | 18.1642 | 1.161439 |
| 8500 | .149151291 | .309180103 | 17.4301 | 19.4686 | 1.272748 |
| 9000 | .148111066 | .309560079 | 17.6645 | 19.4414 | 1.283597 |
| 11000 | .148999034 | .310084213 | 17.4500 | 20.0132 | 1.301612 |

contact MAE在约.146–.150间波动、没有持续改善；32窗固定DEV与TRAIN更新/采样不是独立seed。
这些是同一模型续训曲线，**不是任务成功率、闭环可控性或FM架构无效的证据**。

图：`outputs/ptv3-shared-wam-v6/flowmatching-stop-audit-20261022-r2/training-curves.png`
及同名SVG；数值为`train-curve.csv`、`dev-curve.csv`、`summary.json`。
图中15条logged-batch trailing mean用于读趋势，不改变原始log。

## main保存路径是否有bug：证据与修复边界

**未发现“主阶段从来没保存”的证据。** 周期保存与retention实际工作：
config checkpoint_interval500、keep_recent2、milestone0、预算8GiB。保留10500/11000符合
最后两份策略；更早main8500文件消失符合正常清理，诊断copy另存后仍可读。

但**raw SIGINT跳过正常最终保存**有直接代码和产物证据：

1. 冻结trainer `train_pretrain.py:233–265` 的save先gather三rank RNG，再rank0写`.pt.part`
   原子rename、更新selection链接并retention；不是只有trial能保存。
2. 第365行在周期/新best时保存；第366行的last-completed最终保存位于正常for-loop之后。
   第275–279行只处理受控STOP/time/disk退出，没有把外部SIGINT转成该flag；main没有围住训练
   的KeyboardInterrupt最终保存handler/finally。此次rank在backward/clip中异常退出，未到366行。
3. 第380–392行原本负责result/终态progress写入，也被绕过；因此实际无training/result.json，
   progress残留RUNNING。supervisor `launch_flow_overnight.py:142–145`检测child exit1后记FAILED。
   原文件保持这个真实中断状态，不能回写成正常STOPPED/有final checkpoint。
4. 第194–196行继承旧selection，第252–255行更新链接；因此main的best20/trajectory9并非
   错误step保存，而是**selection与latest身份不同**。界面/查找若只用best别名，会把旧权重误报为最终权重。

供后续明确授权修复的建议，本轮不实施：把外部停止信号转换为所有rank一致的安全update边界
停止请求，完成一致model/optimizer/RNG保存后再退出；显式维护latest/last checkpoint身份与
STOPPED_BY_SIGNAL原因；监督器优先走已实现的受控STOP而非直接打断torchrun。
不能在任意异常处盲目保存一次部分optimizer更新，否则反而得到不一致checkpoint。

本轮没有改训练源码、选择指标、保存策略或原权重，也没有启动任何新方法实验。

## 可核对命令与产物

审计脚本：`src/task/ptv3-shared-wam-v6/tools/audit/audit_flow_stop_checkpoints.py`。
实际产物r1是首次Task内扫描，r2扩大到项目outputs；两次都为CPU只读，r1保留不覆盖。
复现需换一个不存在的输出目录，脚本拒绝覆盖已有审计：

```sh
/home2/wyy/miniconda3/envs/graspenv/bin/python \
  src/task/ptv3-shared-wam-v6/tools/audit/audit_flow_stop_checkpoints.py \
  --run outputs/ptv3-shared-wam-v6/flowmatching-overnight-20261022-r2 \
  --output outputs/ptv3-shared-wam-v6/flowmatching-stop-audit-20261022-r2 \
  --related-root outputs --owner-observed-step 11020
nvidia-smi -i 4,5,6 --query-gpu=index,uuid,memory.used,memory.free,utilization.gpu --format=csv
```

权重CPU读法为 `torch.load(map_location='cpu', mmap=True, weights_only=False)` 配合FakeTensorMode；
不materialize全部optimizer，不创建CUDA context。sentinel另做小tensor真实CPU读，证据在JSON。
目录、真实mtime及软链可另用：

```sh
find outputs/ptv3-shared-wam-v6/flowmatching-overnight-20261022-r2 \
  -type f -name '*.pt' -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n'
find outputs/ptv3-shared-wam-v6/flowmatching-overnight-20261022-r2 \
  -type l -name '*.pt' -printf '%p -> %l\n'
```

最终独立摘要：**目标进程已停/三卡释放；11000主权重存在且不同于trial128；11020只是观察日志点；
最后确认11380；至少380更新未final保存；后期contact改善弱，固定DEV未显示持续改善。**
