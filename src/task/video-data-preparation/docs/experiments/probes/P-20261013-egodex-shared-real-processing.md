---
schema: ref2dex.probe.v2
probe_id: P-20261013-egodex-shared-real-processing
experiment_id: P-20261013-egodex-shared-real-processing
date: 2026-10-13
task: video-data-preparation
branch: main
git_commit: 5dcda3e9d4962d7bd96814b9120d2be16f3a7774
claim_id: C1
hypothesis_family: HF-egodex-shared-track-admission
probe_index_in_family: 1
seed_pool: probe
seeds: []
decision_changed_if_positive: expand a bounded cohort of sparse goal-object windows using the same recipe
decision_changed_if_negative: fix the specific mask, coordinate, scale or tracking blocker before expanding preprocessing
status: UNCLEAR
run_id: egodex-shared-real-processing-20261013-r1
---

# 已下载 EgoDex 的 Shared-WAM 真实处理小探针

Result: UNCLEAR overall. 三个真实窗口完成，1/3 通过数值门；白色积木 32/32 endpoint 有效，位移中位数170mm；另两段在观察尺度标定处拒绝。只有 TRAIN=1、dev=0，不扩训。
Decision: 保留真实数据处理与 Shared-WAM 接入；先修/区分尺度标定及时间 mask 覆盖问题，不直接扩大下载集处理或开始训练。

Class: Blocker/Decision. 服务于 Shared-WAM 人类手物轨迹预训练子目标。问题是：已有
真实 EgoDex 上，当前观察 depth 与未来 label depth 能否落在相同米制坐标中，同时提供
正确目标表面的 32 点 correspondence？合格支持扩一个小 cohort；失败则定位实际阻塞，
不以合成 smoke 通过或单个失败关闭整个方法。

最便宜的判别：按下载 READY 中的原始 recording 顺序选择最多三个 basic_pick_place
episode，按固定时钟选 anchor，先查看当前 RGB 决定目标可见表面，再标记未来 mask。
完整原始 MP4/HDF5 小样本只读复制到本地，不重编码时间、不下载新数据或模型。
只用已观测 H4 构造 observation；25 个原生未来帧用于 tracker/label depth，point ID
在当前阶段固定。原生手和物体伪标签分别报告。

## 固定门与记录

- current query 至少 16 个，固定模型槽 32 个；候选至多 64 个，不按未来幸存重排。
- endpoint 有效/current query ≥80%，报告 32 槽绝对覆盖率及所有 horizon 的有效数。
- 沿用首版 observation/label current depth agreement：median≤30mm、p95≤60mm。
- 刚体诊断 median≤5mm、p95≤15mm，保留未拟合的原始 point labels。
- mask 是本轮 assistant 可视化审阅的保守表面标注，不是官方分割真值；有歧义或
  严重遮挡时拒绝，不扩大到背景/手上凑点。
- 准入只证明该配方可产出样本，不证明真实几何精度、抓取或 Cm 收益。

## Decision Note / 资源

远端已有 producer 与四个 reconstruction GPU 进程保持原状；本轮改用本机一张空闲
GPU，复用本地 Omega/CoTracker。远端只读核对和下载少量原始配对，本地模型推理
最多累计 30 GPU 分钟，单次受控进程最长 20 分钟；最多三个初始窗口。完整小样本
传输≤100MiB，本地新增产物与 scratch 合计≤2GiB，遵守共享 400GB 及本机 GPU
上限。没有新增长期身份或外部服务，也不删除/改写旧数据、checkpoint 或进程。

current target 错、标定不可辨识、OOM、非有限值、陌生 GPU 进程或预算触发时停止
该次执行，保留原因。工程接口错误在原卡记录修复，不把错误实现当作方法负证据。
若一个合理的修复仍在预算内，只补跑受影响窗口；改变处理假设先写短 Decision Note。

## 产物与身份

`outputs/video-data-preparation/egodex-shared-real-processing-20261013-r1/` 保存输入
清单、mask、处理 manifest、每窗状态和可视化；scratch 在项目 `tmp/`。实际固定代码
commit、vendor commit、资产路径与运行设备在执行前写入 manifest。无随机模型初始化
或训练，当前 sampling 为确定性 FPS/固定时钟，因此无需占用训练 seed。

## Limitations / future evidence

少量手工表面 mask 不代表自动 target grounding；局部准入率不代表全量下载集。
depth agreement/刚体残差是自洽检查，不是独立物体 metric ground truth。future mask
审阅不反向改变当前 query，端点 goal 仍属于 hindsight supervised task condition。
自动分割、多个 session 的覆盖、独立几何质量证据与训练收益留待有正向信号后验证。

## Completed result（2026-10-14）

运行开始于10月13日，本节在14日补齐记录。固定代码 `5dcda3e9`，work_version
同commit。Omega vendor `48b23c8`，CoTracker vendor `82e02e8`；模型资产复用
`outputs/cm-pointflow-effect-pretrain/pointwam-pipeline-assets-20261011/`，没有下载新权重。

远端四个 GPU 当时均有旧重建进程，故只读 SCP 三个完整原始配对至本地，共
7,087,471 bytes（约6.76MiB），在本机空闲GPU1执行。三个窗口分别为
`part2/basic_pick_place/{0,1,10}.hdf5`，tick=27/27/35，native length=88/93/107。
保持原始视频和标注时钟；没有重编码、没有修改远端 producer/consumer。

| Episode / 当前目标 | 结果 | 证据 |
| --- | --- | --- |
| 0 / 黄色芥末瓶 | REJECTED | 观察标定57样本，ratio relative IQR=0.337 >0.25；没有进入未来 tracking |
| 1 / 蓝色清洁块 | REJECTED | 观察标定40样本，relative IQR=0.455；软物体本来也只作诊断 |
| 10 / 左指下的白色积木 | 数值 QUALIFIED，局部正向信号 | 32 current/32 endpoint，目标位移中位数170.25mm；current depth agreement median4.93/p955.53mm |

总运行53.21秒，含模型初始化；积木窗口8.28秒，其中 observation1.78秒、tracking
1.13秒、label geometry5.06秒。后续稳定吞吐不能由这一窗直接外推。峰值Torch分配
8194.49MiB（约8.00GiB），GPU1已释放，owned watchdog也已停止。整个本地scope
约40MiB，不含已存在的模型权重。

实际后端核对发现首版强制 `mode=crop` 不适用于已有Omega的 balanced/max_size
接口；修复于 `8c73d9af`，后端沿用 vendor 合法默认 mode。真实运行在此修复之后；
保留了首次工程接口检查与修复记录，没有把接口错误作为方法负证据。

原始深度/track diagnostics、样本、HDF5、snapshot 和结果位于：
`outputs/video-data-preparation/egodex-shared-real-processing-20261013-r1/processing-r1/`。
`manifest.json` 明确代码、vendor、GPU、输入及预算；`events.jsonl`保存阶段耗时。

### 独立数组、读取与视觉核对

积木样本经过真实 HDF5→`EgoDexSharedWindows`→新 collate 读取：
scene `[1,2048,3]`、feature `[1,2048,18]`、目标监督 `[1,1,24,32,3]`、
手监督 `[1,24,2,11,3]`；浮点全部有限，source_id=3。有效物体点步628/768，
原生手点步368/528。没有产生机器人 action/effect。见 `shared-adapter-audit.json`。

人工检查当前与+8/+16/+24 overlay，点簇确实随左手下的积木抬起，而非下面另一块
静止白方块。全25帧已另存有效/无效点overlay，当前point ID未因未来有效性重排。
但是+18…+23的有效点下降到13/8/10/12/19/25，既有遮挡，也受稀疏手工polygon
插值误差影响；不能据endpoint32/32宣称整个中间过程标签完整。

目标表面很小，当前只有局部可见face。最近有效left semantic point距离约28–66mm，
其中多个tip没有confidence-qualified标注；这不是独立的接触距离测量，也不能据此
证实物体与native hand已毫米级对齐。原生手投影与RGB、单目depth均有误差边界。

快照仅TRAIN1/VAL0；freeze用显式 `require_both=False` 做数据合同核对。普通训练
入口仍要求非空dev，所以本probe不能作为完整训练语料。没有启动预训练或微调。

### 结果归因与下一决策

瓶子两手的median scale分别0.743/0.536，各手内部较稳定，但合并后的单尺度
不一致；清洁块则同一右手内部也不稳定。错误范围已定位在observation calibration，
不是因为future目标点数不足；两段都尚未执行tracker，不推断它们track会失败。

最小CPU补充筛查使用已保存的H4 depth，前三帧拟合、第四帧留出：瓶子改为正
scale+shift后，手部Z median误差46.6→18.3mm、p9597.7→40.2mm；清洁块的median
基本不变且拟合scale只有0.021，有压平geometry的风险；积木改善也有限。
这只是相同手点的近邻帧诊断，不是物体几何truth，**没有改变本轮pipeline标定模式或
放宽0.25门，也没有用仿射拟合重新发布标签**。见 `affine-calibration-screen.json`。

下一最小实验：固定当前mask/query与原始模型输出，比较有约束的标定方法并核对
独立geometry/静态背景、手-物局部对齐；先排除仅靠offset压平depth的假改善。
同一积木窗口的future mask可独立精修后在缓存上重算validity，不重选幸存query。
扩多个独立session与自动分割列作future evidence，不在本轮追加大量推理。

### 可视化与复现

当前选择：`current-sheet.jpg`、`current_polygons.json`。独立future mask与修订：
`annotations.json`、`mask-sheet-{0,1,10}.jpg`、`future_polygon_revision.json`。
轨迹预览：`visual-audit-r1/window-000000.jpg`、`all-track-steps-10.jpg`；标定审计：
`calibration-diagnostics.json`、`native-projection-diagnostics.json`。
这些均位于上述本地scope根目录，不在Task文档目录存储数据。

```bash
/home2/wyy/miniconda3/envs/fastwam/bin/python \
  src/task/video-data-preparation/tools/run/process_selected_windows.py \
  --inputs outputs/video-data-preparation/egodex-shared-real-processing-20261013-r1/inputs.json \
  --annotations outputs/video-data-preparation/egodex-shared-real-processing-20261013-r1/annotations.json \
  --assets outputs/cm-pointflow-effect-pretrain/pointwam-pipeline-assets-20261011 \
  --output outputs/video-data-preparation/egodex-shared-real-processing-20261013-r1/processing-r2 \
  --gpu 1 --seconds 1200
```

复跑使用新目录；本卡结论属于r1。后续修复应在新run中说明干预，不覆盖原始产物。
