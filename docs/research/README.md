# 研究文档组织说明

根级 `docs/research/` 不作为默认的实验记录目录，也不替代实验卡、运行 manifest 或 Git。

当前规则：

- 实验协议、状态、结论摘要和产物链接写入对应 Task 的
  `docs/experiments/probes/` 或 `docs/experiments/validations/`；
- 详细方法说明、文献边界和跨多个实验的综合分析可以按需写入
  `src/task/<TaskName>/docs/research/`；
- 数据、标签和复现检查代码写入对应 Task 的 `tools/audit/`；
- 只有真正跨 Task 的综合材料才保留在这里；没有这类材料时，不需要新建文件；
- 2026-10-04 之前的根级研究材料保存在
  [`root research archive`](../archive/2026-10-04-root-research/README.md)。

这里的目录说明不是当前研究事实入口。当前目标、状态、资源边界和实验索引分别见
[`MISSION.md`](../MISSION.md)、[`STATE.md`](../STATE.md)、[`CAMPAIGN.md`](../CAMPAIGN.md)
和 [`experiments/INDEX.md`](../experiments/INDEX.md)。
