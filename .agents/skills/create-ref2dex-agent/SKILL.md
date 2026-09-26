---
name: create-ref2dex-agent
description: 在 Ref2Dex 中新建 Codex 代理对话、选择对应 CODEX_HOME 与网络入口、处理重名并登记独立身份和工作树时使用；向现有代理派发任务时不用。
---

# 新建 Ref2Dex 代理

先读 `AGENTS.md`、`docs/AGENT_COORDINATION.md`、`docs/AGENT_REGISTRY.json`、`docs/STATE.md` 和 `docs/CAMPAIGN.md`。明确新代理的角色、可修改路径、资源权限和要解决的目标；新对话不继承其他代理的实验授权。

## 命名与工作树

选一个描述角色的对话名。若已有同名对话，依次在名称末尾加 `-2`、`-3` 等，取最小未用数字；例如 `agent/workflow`、`agent/workflow-2`。新 `agent_key` 也须唯一，可对应写成 `agent_workflow_2`。保留旧对话名称和身份，不能靠重命名覆盖旧 thread。

需要独立实现路线时，从当前 `main` 创建 `agent/<short-description>` 分支和同级工作树。先检查同名分支与目录，避免覆盖现有修改。工作树目录、分支、对话名和 `agent_key` 分别登记；它们可以相互对应，但不能拿其中一个代替 thread ID。

## 按 `CODEX_HOME` 启动

`CODEX_HOME` 只在将要启动该对话的终端中设置。选择该代理自己的 home，不依赖 root 的环境变量或注册表默认值。

### `/home2/wyy/oyx_ws/.codex_oyx_NewAPI`

先确认本地 `127.0.0.1:18080` 服务是否已运行；若未运行，在单独终端启动并保持该进程运行：

```bash
cd /home2/wyy/oyx_ws
python proxy.py
```

该 home 的 `config.toml` 指向 `http://127.0.0.1:18080/v1`。然后在新代理终端设置：

```bash
export CODEX_HOME=/home2/wyy/oyx_ws/.codex_oyx_NewAPI
cd /home2/wyy/oyx_ws/ai_ws/<该代理的工作树目录>
<当前环境使用的 Codex CLI>
```

已有服务时复用，不重复占用 18080 端口；不停止归属不明的代理进程。

### `/home2/wyy/oyx_ws/.codex_oyx` 或 `.codex_oyx_frj`

在新代理终端设置所选 home 和四个代理变量，然后启动 Codex：

```bash
export CODEX_HOME=/home2/wyy/oyx_ws/.codex_oyx
export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897
export HTTP_PROXY=http://127.0.0.1:7897
export HTTPS_PROXY=http://127.0.0.1:7897
cd /home2/wyy/oyx_ws/ai_ws/<该代理的工作树目录>
<当前环境使用的 Codex CLI>
```

若选择 `.codex_oyx_frj`，只把上面 `CODEX_HOME` 的值改为 `/home2/wyy/oyx_ws/.codex_oyx_frj`；四个代理变量相同。

## 登记与交接

新对话产生 thread ID 后，在 `docs/AGENT_REGISTRY.json` 新增一条记录：唯一的 `agent_key`、`display_name`、`conversation_id`、该代理实际使用的 `codex_home`，由它派生的 `session_root`/三个数据库路径/`rollout_locator`，以及角色、工作树、分支、资源边界和 handoff。身份键是 `(codex_home, conversation_id)`。若是在替换旧对话，把旧身份移入 `retired_conversations`，不要复用旧 thread ID。

运行 `docs/AGENT_COORDINATION.md` 第 1 节的注册表校验和 `python3 tools/verify.py --changed`。主代理再核对新工作树已知晓 `main` 的最新 `STATE`/`CAMPAIGN` 决定，然后按第 4 节派发一个明确 Goal；当前冻结的 Cm campaign 不因创建新代理而解冻。
