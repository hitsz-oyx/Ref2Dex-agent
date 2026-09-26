# Baseline 工作树的冻结后 Probe 复核（2026-09-26）

本记录是监督审计，不更改 [Option A 冻结决定](../decisions/D-20260926-after-hf04-route-review.md) 或 Probe 的科学标签。

* `main` 的 Option A 冻结提交 `e1f75e0` 时间为 14:40:44（Asia/Shanghai）。
* 注册为 CPU-only 的 `agent_baseline` 工作树在 `agent/grab-full-baseline` 分支产生了 local-residual Cm-on/off GPU 评估。两份 manifest 均记录 `gpu_count=1`、`run_status=COMPLETED`，完成时间分别为 14:45:09 和 14:45:14。
* 注册 thread `01a0d6bf-ba61-7f92-8af7-d68c91f4007d` 的 rollout 显示：14:36:07 创建 r2 输出目录并做预检，14:38:35 发起一次评估调用，14:42:51 再次发起包含 r2 Cm-off/on 的评估调用。最后一次调用在冻结提交之后。manifest 自身没有 `started_at`，所以具体模拟器进程启动秒数仍以 rollout 调用记录为准。
* 该分支的代码提交 `4c638e2` 时间为 14:12:59，结果卡提交 `bcf2637` 时间为 14:54:48。结果卡的两份 manifest SHA256 与现有文件一致。检查时没有该工作树归属的 GPU 计算进程。
* 注册 thread 的 Goal 当前为 `paused`，但 rollout 证明该 thread 发起了上述调用。Goal 状态与手动活动 turn 可以不同；工作树中的其他 Codex CLI 进程不据此归属同一实验。

处置：保留该分支和原始产物供审计；`4c638e2`、`bcf2637` 暂不合入 `main`，不把新结果纳入当前 frozen campaign 的正式证据链，不启动复测。若需要接受这次评估为例外或改变 `agent_baseline` 的 GPU 权限，先由用户作出明确决策。下一次分配任务前，主代理须核对该工作树是否已接收主线冻结决定及资源边界。

用户于 2026-09-26 决定：**维持隔离并继续监督**。上述两笔提交保持在 baseline 分支，不作为本轮 mainline 集成候选；主代理继续轮询注册 thread 和资源状态。
