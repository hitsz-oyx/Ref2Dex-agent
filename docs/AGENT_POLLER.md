# 结果轮询代理运行合同

**角色：** `agent_poller`（对话名 `agent/poller`）  ·  **所有者：** `/root`

轮询代理使用 `/home2/wyy/oyx_ws/.codex_oyx_NewAPI`，只观察 `docs/AGENT_REGISTRY.json` 中登记的子代理。它发现状态变化后向 root 的精确 `(codex_home, conversation_id)` 发送一次 `POLL_EVENT`；root 负责审查证据、决定下一步和合并。轮询代理不能把提交、manifest 或 Goal 终态解释为科学结论。

## 观察范围

由 [agent_result_poller.py](../scripts/agent_result_poller.py) 在等待/CPU 阶段每 5 分钟、发现归属 GPU 计算进程后每 2 分钟读取：

* 每个子代理工作树的 HEAD；
* 对应 thread 的 Goal ID/状态；
* 该工作树 `outputs` 下最新 `run_manifest.json` 的路径和修改时间；
* 工作目录归属该工作树的 GPU 计算进程 PID。

首次运行只建立快照。之后只有这些字段变化才排队 `POLL_EVENT`，成功排队后才更新去重状态。队列失败会保留旧快照并在下一轮重试。脚本不能观察未写入 manifest、未提交且未改变 Goal/GPU 状态的内部想法；子代理仍须按 [协作合同](AGENT_COORDINATION.md) 提交 handoff。

## 启动和运行

先按 [新建代理 skill](../.agents/skills/create-ref2dex-agent/SKILL.md) 启动 NewAPI 对话并登记。复用 `proxy.py` 的 18080 前，必须按该 skill 做有界 HTTP health check；仅有 LISTEN 端口不代表请求会返回。若检查超时，按 skill 的临时独立副本和 invocation-local `model_providers.rlg.base_url` override 运行，不修改或停止共享 proxy/config，也不启动第二个 18080 服务。在 poller 工作树中运行：

```bash
python3 scripts/agent_result_poller.py \
  --registry /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/docs/AGENT_REGISTRY.json \
  --state outputs/agent_poller/state.json \
  --codex-node /home2/wyy/.nvm/versions/node/v24.19.0/bin/node \
  --codex-js /home2/wyy/.nvm/versions/node/v24.19.0/lib/node_modules/@openai/codex/bin/codex.js \
  --interval 300
```

进程应归属于 poller 对话的长任务；按 `long-running-tasks` skill 稀疏检查其输出。脚本的普通无变化轮询不调用模型，也不发送队列消息；poller 对话检查进程时仍会消耗 token。`NOTIFIED` 和 `POLL_ERROR` 需要代理立即检查。运行前确认 `STATE.md` 的 Cm campaign 仍冻结。轮询代理没有 GPU、训练、评估、进程终止、分支合并或修改其他工作树的权限；只可维护自己的去重状态和运行记录。

## 通知与恢复

`POLL_EVENT` 只表示“有变化需要 root 审查”，包含 agent key、变化字段、旧/新 HEAD/Goal/GPU/manifest 快照。不得对 root 发送固定心跳。通知使用注册表中的 root 身份及 root 的 `CODEX_HOME`，而非 poller 的 NewAPI home。若 root 对话暂时停下，新结果由队列唤起审查；若队列不可用，保留事件并报告 `POLL_ERROR`。

重启前检查旧进程和 `outputs/agent_poller/state.json`；同一时间只允许一个 poller 进程。停止时只停止已确认属于该代理的轮询进程，不触碰其他 Codex 或实验进程。
