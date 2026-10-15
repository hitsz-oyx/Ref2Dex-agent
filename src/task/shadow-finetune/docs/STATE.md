# ShadowHand 原生微调当前状态

2026-10-15：固定计算预算的小数据/全量对照已在GPU6启动，实验
[P-20261015-shadow-data-scale](experiments/probes/P-20261015-shadow-data-scale.md)。
固定代码57737b7b、detached工作树`tmp/shadow-scale-code-20261015-r2`，supervisor2083589。
当前small臂180/1000，之后自动进行原生终点评估、full1000步和相同起点评估，再核验匹配与汇总。
父权重同80k WAM-v2、新Shadow动作头，seed214、batch64、共同full DEV64窗和small TRAIN统计。
small7物体/41轨迹/6232窗，full1567物体/10118轨迹/1536897窗；数据合同及nested逐值核验通过。

用户已允许临时一张空卡做小规模测试；本轮仅GPU6、总≤1小时/14GB，GPU7空闲，原Inspire
GPU0–5及远端训练继续。GPU6首轮采样平均77.8%利用率/最大3.6GiB，当前约0.55秒/步；
整轮暂估25–35分钟，full实际吞吐待测。非有限/超时/磁盘<20GiB/预算/外来进程触发仅停止own组。
首次r1因工作树第三方依赖缺失在训练前退出，失败记录保留，r2已补齐并通过真实更新。

先前工程/小pilot已经完成，学生16env官方success1/16（TRAIN1/8、DEV0/8），teacher5/16；
这不是当前匹配数据量对照的一臂，也没有证明跨物体泛化或Inspire失败的数据量归因。
当前对照尚未完成，不能作数据量收益判断。实际进程/进度以
`outputs/shadow-finetune/shadow-scale-controls-20261015-r2/process.json`与各臂progress为准。
