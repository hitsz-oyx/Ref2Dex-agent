# Research 文档

## 当前独立工作树研究

`agent/contact-response-cm` 的现状见 [STATE](../STATE.md)，首轮实际结果见
[contact-response results](20261001-contact-response-results.md)，文献和 novelty
边界见 [literature](20261001-contact-response-literature.md)。论文工作稿位于
[paper](../../paper/README.md)。此分支的实验、写作与产物独立于原 paired-evaluator
工作树；旧冻结指令和状态记录用于解释历史证据。

这里按用途索引研究事实；具体的当前文件暂时保留在 `docs/` 根目录，因为
`tools/verify.py`、实验卡和已有交接把它们作为稳定入口。

| 类别 | 入口 |
| --- | --- |
| 研究问题 | [`MISSION.md`](../MISSION.md) |
| 当前事实与 North-star | [`STATE.md`](../STATE.md) |
| 机器和资源边界 | [`CAMPAIGN.md`](../CAMPAIGN.md) |
| 当前可消费路线 | [`RESEARCH_QUEUE.yaml`](../RESEARCH_QUEUE.yaml) |
| seed 归属 | [`SEED_LEDGER.yaml`](../SEED_LEDGER.yaml) |
| 论文后补实验 | [`RESEARCH_DEBT.md`](../RESEARCH_DEBT.md) |
| 项目全局概览 | [`项目总览.md`](../项目总览.md) |

研究记录已经按目录分开：

* `decisions/`：Decision Checkpoint 和路线选择；
* `experiments/probes/`、`experiments/validations/`：实验卡和正式验证；
* `handoffs/`：研究路线交接和证据审计；
* `activities/`：跨 Task 研究活动；
* `archive/`、`logs/`：只读历史资料，不是默认上下文。

旧的根级 `plan/`、`指导/` 和多代理 workflow 规范已删除。Task 内部仍可能保留与历史
实验绑定的计划或指导文件；它们只用于解释对应证据，不是仓库级运行规则。
