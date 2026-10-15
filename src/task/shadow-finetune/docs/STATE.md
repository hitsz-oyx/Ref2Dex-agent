# ShadowHand 原生微调当前状态

2026-10-15：固定计算预算的小数据/全量对照已在GPU6启动，实验
[P-20261015-shadow-data-scale](experiments/probes/P-20261015-shadow-data-scale.md)。
固定代码57737b7b、detached工作树`tmp/shadow-scale-code-20261015-r2`，supervisor2083589。
small1000步及原生终点评估已完成，full已更新到100/1000，之后自动进行相同起点评估与汇总。
父权重同80k WAM-v2、新Shadow动作头，seed214、batch64、共同full DEV64窗和small TRAIN统计。
small7物体/41轨迹/6232窗，full1567物体/10118轨迹/1536897窗；数据合同及nested逐值核验通过。

用户已允许临时一张空卡做小规模测试；本轮仅GPU6、总≤1小时/14GB，GPU7空闲，原Inspire
GPU0–5及远端训练继续。GPU6首轮采样平均77.8%利用率/最大3.6GiB，当前约0.55秒/步；
full实际约0.80秒/步、峰值allocated3017MiB，剩余更新暂估12分钟。
非有限/超时/磁盘<20GiB/预算/外来进程触发仅停止own组。
首次r1因工作树第三方依赖缺失在训练前退出，失败记录保留，r2已补齐并通过真实更新。

先前工程/小pilot已经完成，学生16env官方success1/16（TRAIN1/8、DEV0/8），teacher5/16；
这不是当前匹配数据量对照的一臂，也没有证明跨物体泛化或Inspire失败的数据量归因。
当前对照尚未完成，不能作数据量收益判断。实际进程/进度以
`outputs/shadow-finetune/shadow-scale-controls-20261015-r2/process.json`与各臂progress为准。

本次matched small终点学生官方success7/16（TRAIN6/8、DEV1/8），末帧2/16、同步抬升保持60帧3/16；
teacher官方8/16（7/8、1/8），末帧/保持均7/16。独立raw计数与命令时钟复核一致，初态误差0、
无invalid，不将sticky7/16说成稳定抓持。small与full实际初始actor51个张量完全相同。

CPU阶段覆盖诊断：完整forecast要求只保留安全action8候选中351321/706676个近目标query；
7868条含近目标query，action-only可达9791条（TRAIN10118条）。原始数据不是没有保持阶段，
但完整未来支持显著减少晚期BC监督；当前对照仍冻结原规则，结果后再决定是否做尾段最小对照。
完整数字与解释边界见实验卡，不提前下数据量结论。
