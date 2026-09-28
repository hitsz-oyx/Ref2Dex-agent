# Workflow 文档审计（2026-09-28）

## 结论

当前固定角色、Broker、四类消息和 root 决策循环已经在 `AGENTS.md`、Broker 文档和
固定角色文档中形成主链路。`docs/ref.md` 当前是一份“六专家蒸馏为单一 student”的
研究方向建议，不是已经批准的实验任务。历史研究记录中的旧 agent 名称、旧 queue
消息和旧 supervisor 不应被当作当前实现。

## 发现的冲突

1. `docs/ref.md` 建议先做离线 BC，再做 closed-loop aggregation 和小预算 PPO
   finetune；但 `docs/STATE.md`、`docs/RESEARCH_QUEUE.yaml` 和
   `D-20260927-cm-freeze-option-a.md` 明确规定当前不登记 HF06、不启动 GPU、收集、
   online/PPO 或新的 Cm 实验。两者不是科学结论冲突，而是“候选机制”和“当前授权”
   的边界；在新 high-level goal、资源预算和 Decision Checkpoint 之前，不能按
   `ref.md` 直接执行。
2. `docs/ref.md` 说该六专家→single student 路线尚未被测试；现有记录支持“没有找到
   对当前六个 self-trained experts 的完整蒸馏 Probe/Validation”，但历史资料确实
   提到过 teacher/distillation 和其他 shared actor。正式登记该空白前，应把它写成
   新实验卡并固定数据、teacher、student、门槛和资源，而不是把建议段落当成证据。
3. workflow 文档内部还有一个实现边界差异：`AGENT_BROKER.md` 把 runtime 操作交给
   provider adapter，而 `scripts/root_watchdog.py` 在 Broker 模式仍直接调用
   app-server resume，再写 `CONTROL/RESUME`。这不影响本次研究路线，但属于后续
   workflow 实现迁移项；本次不借文档改名掩盖它。
4. 原 `create-ref2dex-agent` skill 中的直接 `codex queue --thread`/`GOAL_DISPATCH`
   示例已改成 Broker `TASK_DISPATCH`；直接 queue 仅保留为 adapter 内部兼容说明。

## 不属于当前冲突的旧内容

* `POLL_EVENT`、`ROOT_LIVENESS_WAKE` 和旧 `agent_result_poller` 出现在兼容脚本及其
  测试中；它们已经被文档标为 legacy，不是 Broker 四类消息。
* `agent_cm_temporal`、`agent_poller` 等出现在 handoff、decision、activity 和 log
  中时，是历史身份记录；当前稳定组织身份是 `agent_cm`，不能回写历史文件。
* `docs/plan/` 和 `docs/指导/` 讨论旧治理阶段；它们被保留为历史资料，不作为新对话
  默认上下文。

## 分类决定

* 当前 workflow 由 `docs/workflow/README.md` 索引；稳定入口暂不移动，以免破坏脚本、
  测试和已有相对链接。
* 当前研究事实由 `docs/research/README.md` 索引；`decisions/`、`experiments/`、
  `handoffs/`、`activities/`、`archive/` 和 `logs/` 继续承担各自唯一职责。
* 顶层的旧 Cm closeout 和旧 handoff 移入 `docs/handoffs/`，根路径留下兼容链接。
