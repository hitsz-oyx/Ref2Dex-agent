---
schema: ref2dex.probe.v2
probe_id: P-20261016-teacher-sequence-generalization
experiment_id: P-20261016-teacher-sequence-generalization
date: 2026-10-16
task: ptv3-shared-wam-v2
branch: main
git_commit: f4db2a4a
claim_id: C1
hypothesis_family: HF-full-teacher-input
probe_index_in_family: 3
seed_pool: probe
seeds: [180, 181]
decision_changed_if_positive: prioritize sequence coverage if isolated local interpolation succeeds but unseen sequences fail
decision_changed_if_negative: investigate control representation and optimization if isolated local interpolation also fails
status: PLANNED
run_id: teacher-sequence-generalization-20261016-r1
---

# 完整教师输入：局部插值还是序列记忆？

Class: Decision。服务 MISSION 自训控制器底座，先回答参考信息充分时的控制映射泛化，
不同时改变 WAM、手型、标签、模型、执行频率。用户已指定三条件设计并授权推进。
前轮全量缓存复算 TRAIN startup0.466181mm、DEV startup47.213093mm，差距可复现；
这不证明整个 TRAIN 动态映射已学会，也不直接证明学生记住了序列身份。

## 假设、决策和最便宜方法（运行前固定）

H1：若局部控制规律可插值而覆盖不足，同序列隔离片段会好、原DEV新序列会差；
优先测试少量新序列示范，不继续增加整套固定数据训练。
H2：若局部插值也差，先检查目标/输入表示/优化；不把错误归因于闭环。
H3：若三组离线都好，后续再检查实际执行动态及偏离后的恢复数据。
这些为候选解释，不以单 seed 给出控制规律是否存在的正式科学判决。

最便宜可判别实验：复用已全660序列通过teacher重放的1442D缓存、同次q_PD单步标签、
相同4,107,788参数MLP和teacher RMS/clamp。仅重划TRAIN中的连续片段，从scratch训练一组；
不新采集、不运行仿真、不训练WAM。没有精确的软件bug假设，因此不预设“修复成功”断言；
反馈环为固定三组的物理PD误差。原误差数组的起步差距已在本会话重新计算一致。

## 数据与无重叠划分

固定Inspire手型，cup/mug容器任务族：原TRAIN57序列、原DEV6序列，TRAIN覆盖全部10subjects。
选择只按object元数据，不按已有误差/抓取成败。保留这六条DEV所有原准入tick，
不读TEST、不把DEV标签用于训练/归一化；原全660基准也不修改。
cache：`outputs/ptv3-shared-wam-v2/full-teacher-cache-20261015-r1/`。
源准入沿用已审计缓存的历史规则，不混入新人工标注或更晚的清洗结果。

每条TRAIN序列按真实tick找最长连续准入run，留出其40%–60%的半开区间。
留出>=16帧且隔离后两侧各>=8个训练tick才进入插值评价；否则只参与训练，并显式记录。
按照原H4/future24窗口支持[t-3,t+24]保守隔离，训练query到任意留出query距离必须>=28帧；
隔离带完全不参与训练。当前实际输入只用同拍状态及未来1/16参考，标签单步，
这个隔离也覆盖实际所需支持。不存在历史动态状态的独立轨迹重采样，本来就属于同一rollout插值。
将逐序列interval、排除数、最小距离和三组原cache行号存入membership.json，方便独立复核。

三组为seen（真正采样训练的所有状态）、interpolation（隔离片段）、unseen（原DEV）。
主统计为sequence-macro XYZ PD mean；同时保存tick mean/median/p95、逐序列、startup<16、
later>=16、原序列40%–60%时间段和executable/clipping/rotation/finger指标。
插值段没有startup，不用startup拟合优异掩盖later失败；另外列出有留出片段序列的seen-later。
各组phase组成可能不同，分层结果与每条序列必须一并解释。

## 固定训练与判据

模型/decoder与前轮full_relative完全相同：1442→1024→1024→1024→512→12，ReLU，
first6预测q_PD-liveq、active fingers绝对值，decoder加同拍liveq；q不额外进入网络。
初始化180、采样181，沿用旧seed以控制实现差异，不是独立重复。
teacher提供固定RMS epsilon1e-5/clamp[-5,5]，不初始化学生网络；无tau。
目标均值/标准差只由实际训练行拟合，floor1e-3，heldout与隔离带均不进入目标统计。
sequence-uniform、半数startup采样，batch512、AdamW3e-4/decay1e-4、L1/clip1。
24k更新，12k后cosine到3e-6；每1k报告全部三组，固定24k末步决定，不作heldout早停选点。
保存0/12k/24k权重及末步完整预测；smoke独立输出、不得进入科学结果。

基本seen拟合参考门为sequence-macro mean<=5mm；局部插值好参考门为mean<=10mm且tick p95<=30mm；
unseen mean>20mm为仍有明显新序列缺口，<=20mm是离线值得继续的信号。
先判断seen是否拟合，再判断插值和新序列；出现边界/分层冲突则UNCLEAR，不机械升级结论。
seen好、插值好而unseen差→覆盖诊断PROMISING；seen好但插值差→该控制映射的局部泛化UNPROMISING。
seen尚未拟合→UNCLEAR；全部离线好只支持开始闭环检查，不能声称闭环已解决。

## 资源、版本、停止

当前分支main，保留其他会话未提交修改，不新建分支。运行前对本轮代码建立独立Git checkpoint；
manifest记录实际commit，本文头部记录设计时base；此后补录运行commit。
本地GPU5（启动前0–4有既有工作、5/6/7空闲），最多新增一张，保持GPU6/7两卡空闲。
单run<=1200s、<=512MB、磁盘保留20GiB；工程smoke另<=512MB。
训练开始后观察显存/利用率/ETA，遇非有限、foreign GPU consumer、磁盘/产物/超时预算失败即停自己。
只写本仓库输出，外部teacher checkpoint只读；TMPDIR固定为项目tmp。

## Limitations / future evidence

单seed、反复使用的原DEV，不作正式结论；两物体族不是全660覆盖。
完整teacher输入及其RMS是特权参考信息；源rollout不保证成功抓取。
隔离支持防窗口重叠，不保证同序列邻近时刻统计独立；这正是局部插值而非新序列验证。
未来的少量示范补齐、增量样本数、表示变更及闭环恢复均需本轮结果后再设计，当前不扩展预算。
