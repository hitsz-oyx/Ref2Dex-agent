---
name: create-ref2dex-agent
description: 在 Ref2Dex 中新建 Codex 代理对话、选择对应 CODEX_HOME 与网络入口、处理重名并登记独立身份和工作树时使用；向现有代理派发任务时不用。
---

## Fixed-role policy (active workflow)

Ref2Dex now uses a fixed pool: `root`, `agent_cm`, `agent_rl`, `agent_eval`, and
`agent_infra`. Before creating anything, check `docs/AGENT_REGISTRY.json`:

Normal task dispatch does not create or resume an agent conversation from the
root model. The root submits `TASK_DISPATCH` to `scripts/agent_broker.py`; the
broker routes it to the local binding and provider adapter. Use this skill only
when a fixed role needs a runtime rebind or when the user has explicitly
approved a new long-lived role.

1. If the requested capability belongs to an existing pool role, **rebind that
   role's runtime** in the machine-local `.runtime/AGENT_BINDINGS.json`. A new
   conversation, `CODEX_HOME`, or worktree may replace a broken runtime, but the
   stable `agent_key` must stay unchanged.
2. Creating a new `agent_key`, long-lived branch, or role is
   `user_approval_only`. Do not infer approval from an ordinary task dispatch or
   from a worker's `NEEDS_HELP` handoff.
3. Workers never create workers. Send cross-capability needs back to `/root`.

The registry's legacy `agents` records remain auditable runtime history. They do
not expand the fixed pool. After a rebind, preserve the old `(codex_home,
conversation_id)` in `retired_conversations` and never reuse its thread ID.

# 新建 Ref2Dex 代理

先读 `AGENTS.md`、`docs/AGENT_ROLES.yaml`、`docs/AGENT_BROKER.md`、
`docs/AGENT_COORDINATION.md`、`docs/ROOT_AGENT.md`、`docs/AGENT_REGISTRY.json`、
`docs/STATE.md` 和 `docs/CAMPAIGN.md`。如果要观察或恢复运行时，再读
`docs/AGENT_POLLER.md` 和本机 `.runtime/AGENT_BINDINGS.json`。明确新代理的角色、
可修改路径、资源权限和要解决的目标；新对话不继承其他代理的实验授权。

## 命名与工作树

选一个描述角色的对话名。若已有同名对话，依次在名称末尾加 `-2`、`-3` 等，取最小未用数字；例如 `agent/workflow`、`agent/workflow-2`。新 `agent_key` 也须唯一，可对应写成 `agent_workflow_2`。保留旧对话名称和身份，不能靠重命名覆盖旧 thread。

需要独立实现路线时，从当前 `main` 创建 `agent/<short-description>` 分支和同级工作树。先检查同名分支与目录，避免覆盖现有修改。工作树目录、分支、对话名和 `agent_key` 分别登记；它们可以相互对应，但不能拿其中一个代替 thread ID。

## 按 `CODEX_HOME` 启动

`CODEX_HOME` 只在将要启动该对话的终端中设置。选择该代理自己的 home，不依赖 root 的环境变量或注册表默认值。启动是一个有界的 bootstrap：先验证运行时和 HTTP 入口，再创建或恢复对话；不要用无限重试掩盖入口、thread 或模型容量问题。

### 先固定 Node 与 Codex CLI

不要调用系统的 `node`、`npx` 或未解析路径的 `codex`。这台机器的系统 Node 可能仍是 v10，无法可靠运行当前 CLI。使用绝对的 Node v24 和 CLI 文件，并在启动前做一次只读版本检查：

```bash
CODEX_NODE=/home2/wyy/.nvm/versions/node/v24.19.0/bin/node
CODEX_JS=/home2/wyy/.nvm/versions/node/v24.19.0/lib/node_modules/@openai/codex/bin/codex.js
test -x "$CODEX_NODE" && test -f "$CODEX_JS"
"$CODEX_NODE" "$CODEX_JS" --version
```

后续 `resume`、`queue` 或普通启动都用同一对绝对路径。若版本检查失败，先修复 CLI 路径，不要创建新的代理 thread。

### `/home2/wyy/oyx_ws/.codex_oyx_NewAPI`

先对 `127.0.0.1:18080` 做**有界 HTTP 健康检查**。端口处于 LISTEN 只说明有进程占用，不能说明 Codex 请求会返回；不要只用 `ss`/`nc` 判断可用。可以接受 401/404 等 HTTP 响应（它们证明请求已返回），但 `000`、连接拒绝或超时都算失败：

```bash
check_proxy() {
  local base_url="$1" code
  code="$(curl --noproxy '*' --silent --show-error --http1.1 \
    --connect-timeout 1 --max-time 4 \
    --output /dev/null --write-out '%{http_code}' \
    "${base_url}/v1/models" 2>/dev/null || true)"
  test -n "$code" && test "$code" != "000"
}

check_proxy http://127.0.0.1:18080
```

这个 home 的 provider 是 loopback URL。**每一次** NewAPI bootstrap、TUI 启动、`resume` 或 `queue` invocation 都必须按 invocation 清除外部代理变量；不要只依赖 `NO_PROXY`，也不要把当前 shell 的 proxy 环境传给 Codex。用同一 shell 中的 helper 可以避免遗漏：

```bash
NEWAPI_HOME=/home2/wyy/oyx_ws/.codex_oyx_NewAPI
codex_newapi() {
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
    CODEX_HOME="$NEWAPI_HOME" \
    "$CODEX_NODE" "$CODEX_JS" "$@"
}
```

`codex_newapi` 的 `env -u` 是 process-local 的；不要修改共享 `config.toml` 或全局 shell 环境。健康的共享 18080 和下面的临时 loopback 副本都使用这个 helper。只有 `.codex_oyx`/`.codex_oyx_frj` 段落明确保留四个 proxy 变量，不能把那些 `export` 复制到 NewAPI 命令。

若 18080 没有响应，在单独终端启动共享代理并记录其归属：

```bash
cd /home2/wyy/oyx_ws
python3 proxy.py
```

启动后再次运行上面的有界检查。已有服务时复用，不重复占用 18080；不停止、重启或修改归属不明的共享进程。

如果 18080 **能建立连接但 HTTP 检查超时/卡住**，不要修改共享 `proxy.py`、共享 `config.toml`，也不要杀掉该进程。为这一次 invocation 运行独立副本：把脚本复制到临时目录，只修改副本的监听端口，使用空闲的 loopback 端口，并再次做有界检查。例如：

```bash
PROXY_TMP="$(mktemp -d /tmp/ref2dex-proxy.XXXXXX)"
PROXY_PID=""
cleanup_proxy() {
  if test -n "${PROXY_PID:-}" && kill -0 "$PROXY_PID" 2>/dev/null; then
    kill "$PROXY_PID" 2>/dev/null || true
    wait "$PROXY_PID" 2>/dev/null || true
  fi
  if test -n "${PROXY_TMP:-}" && test -d "$PROXY_TMP"; then
    rm -rf -- "$PROXY_TMP"
  fi
}
trap cleanup_proxy EXIT INT TERM

cp /home2/wyy/oyx_ws/proxy.py "$PROXY_TMP/proxy.py"
PROXY_PORT="$(python3 - <<'PY'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
)"
python3 - "$PROXY_TMP/proxy.py" "$PROXY_PORT" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
port = sys.argv[2]
text = path.read_text(encoding="utf-8")
text = text.replace("http://127.0.0.1:18080", f"http://127.0.0.1:{port}")
text = text.replace('("127.0.0.1", 18080)', f'("127.0.0.1", {port})')
path.write_text(text, encoding="utf-8")
PY

python3 "$PROXY_TMP/proxy.py" >"$PROXY_TMP/proxy.log" 2>&1 &
PROXY_PID="$!"
BASE_URL="http://127.0.0.1:${PROXY_PORT}"
proxy_ready=0
for attempt in 1 2 3 4 5; do
  if check_proxy "$BASE_URL"; then
    proxy_ready=1
    break
  fi
  sleep 1
done
test "$proxy_ready" -eq 1
kill -0 "$PROXY_PID"
ps -o pid=,args= -p "$PROXY_PID"
```

The copied proxy is disposable and must not be installed over the shared file. If its bounded health check also fails, stop and report the endpoint failure; do not turn it into an unbounded retry loop. Override only the provider URL for that invocation; do not edit the shared config:

```bash
WORKTREE=/home2/wyy/oyx_ws/ai_ws/agent-workflow
cd "$WORKTREE"
codex_newapi \
  -c "model_providers.rlg.base_url=\"${BASE_URL}/v1\"" \
  exec "<initial prompt>"
```

The `-c` override is process-local. Use the same override for `resume`/`queue` while the copied proxy is alive. On normal exit, the trap stops only the recorded copied-proxy PID and removes its validated temporary directory; it never touches 18080 or another agent's process.

该 home 的默认 `config.toml` 指向 `http://127.0.0.1:18080/v1`。健康的 18080 服务可直接复用；不健康时才使用上述副本和 invocation-local override。然后在新代理终端设置：

```bash
WORKTREE=/home2/wyy/oyx_ws/ai_ws/agent-workflow
cd "$WORKTREE"
codex_newapi
```

已有服务时复用，不重复占用 18080 端口；不停止归属不明的代理进程。

### `/home2/wyy/oyx_ws/.codex_oyx` 或 `.codex_oyx_frj`

这些 home 走外部代理；四个 proxy 变量只在本段落保留。不要把它们继承或复制到上面的 NewAPI invocation。然后在新代理终端设置所选 home 和四个代理变量，再启动 Codex：

```bash
export CODEX_HOME=/home2/wyy/oyx_ws/.codex_oyx
export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897
export HTTP_PROXY=http://127.0.0.1:7897
export HTTPS_PROXY=http://127.0.0.1:7897
WORKTREE=/home2/wyy/oyx_ws/ai_ws/agent-workflow
cd "$WORKTREE"
"$CODEX_NODE" "$CODEX_JS"
```

若选择 `.codex_oyx_frj`，只把上面 `CODEX_HOME` 的值改为 `/home2/wyy/oyx_ws/.codex_oyx_frj`；四个代理变量相同。

### 首轮失败后的 thread 恢复

首轮请求可能在模型返回前已经创建并持久化 thread。看到 timeout、proxy error 或首轮 turn 失败时，先保留并确认 CLI 输出/会话记录中的**精确 thread UUID**，再修复入口；不要凭显示名另建一个对话，也不要用 `--last` 猜测。

```bash
THREAD_ID="paste-the-existing-thread-uuid-here"
WORKTREE=/home2/wyy/oyx_ws/ai_ws/agent-workflow
PROVIDER_ARGS=()
if test -n "${BASE_URL:-}"; then
  PROVIDER_ARGS=(-c "model_providers.rlg.base_url=\"${BASE_URL}/v1\"")
fi
codex_newapi \
  "${PROVIDER_ARGS[@]}" \
  resume "$THREAD_ID" -C "$WORKTREE"
```

runtime 恢复只确认同一 `(CODEX_HOME, THREAD_ID)` 身份。普通任务派发不直接向
thread queue 写消息，而是由 root 把 `TASK_DISPATCH` 写入 Agent Broker；provider
adapter 再决定如何唤醒对应 runtime：

```bash
python3 scripts/agent_broker.py dispatch \
  --task-id '<task-id>' \
  --target '<stable-agent-key>' \
  --objective '<明确任务>'
```

直接 `codex queue --thread` 只能作为 provider adapter 内部的兼容实现，不能作为
root 的任务协议，也不能再使用 `GOAL_DISPATCH` 名称。恢复后仍须登记同一 runtime
身份并由 Broker 记录任务状态。

恢复前核对 `(CODEX_HOME, THREAD_ID)` 与注册表身份键、工作树和 branch 一致；恢复后再登记/更新同一条记录。只有确认没有可恢复的持久化 ID 时才创建新 thread，并说明原因。

### C1 故障交接（2026-09-27）

thread `01a0de76-3a3f-7293-88f8-18c140024f9f` 的 session 已持久化，但 astra、sol 和 TUI 首轮都在 `task_started` 后没有 assistant `response_item`。该进程继承了四个 proxy 变量，活动连接落到 `127.0.0.1:7897`，而 NewAPI provider 配置是 `127.0.0.1:18080`; 对 18080 的有界直连检查返回 401，session 没有 capacity/429/502/504 错误。因此这次实际故障归类为本地 provider 被无关外部 proxy 环境干扰，而不是模型容量。

安全恢复只向已核对归属的原 thread 进程发送一次 bounded interrupt，然后用上面的 `codex_newapi`、同一 UUID、可用的 `gpt-6-sol` 和 invocation-local loopback URL override 做 bounded `resume`/`queue`。恢复得到 assistant response 和 `task_complete`，C1 10/10 原始结果未重跑或修改；不要由此创建新 thread、重启共享 proxy 或更改共享配置。

### 区分模型容量错误

不要把模型容量问题误判为 Node 或 proxy 问题：连接拒绝、`000`/超时、502/504 通常是入口问题；`server_overloaded`、capacity/429 或明确的模型暂不可用响应则是容量问题。容量问题只允许有限次重试（例如两次、带短退避），然后：

* 保留原 thread ID，不重复创建代理；
* 使用当前确实可用的模型，通过 `-m <available-model>` 或 `-c 'model="<available-model>"'` 重试/恢复；
* 在交接中记录实际选择的模型和错误类别。

若错误类别不清楚，停止并交接诊断，不要同时换 thread、CODEX_HOME、工作树和模型，使故障不可追溯。

### 启动后的安全核对与清理

至少核对以下项目后才把 bootstrap 视为成功：

1. `git -C <worktree> branch --show-current` 是目标 branch，且没有误写 `main` 或其他 worktree；
2. 绝对 Node/CLI 的 `--version` 成功；
3. 共享或副本 proxy 的 HTTP 检查在规定时限内返回；副本还要核对 `PROXY_PID` 的命令行和监听端口确实属于本次启动；
4. 注册表记录的 `(codex_home, conversation_id)`、worktree、branch 与实际值一致；
5. 首轮失败时，原 thread/session 日志仍保留，供 `resume` 和审计使用。

退出时只清理自己创建的副本进程、临时目录和日志；不删除 session/thread、注册表记录、共享 `config.toml` 或无法确认归属的进程。不要使用 `pkill`、`killall` 或宽泛的递归删除。

### Linked worktree 的 Git 提交边界

managed workspace 可能只把代理工作树设为可写，而把 linked worktree 的真实 Git
metadata 设为只读。C1 的实证是：

```text
<worktree>/.git
  -> /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent-baseline/.git/worktrees/<name>
```

`git commit` 因无法创建该目录下的 `index.lock` 而返回 `Read-only file system`；
宿主机的 Unix mode 和 `/home2` 挂载并不是原因。不要反复重试，也不要把整个
baseline `.git`、其 common `objects`/`refs` 或 `--dangerously-bypass-approvals-and-sandbox`
加入代理权限。

这不是假设性边界：C1 thread
`01a0de76-3a3f-7293-88f8-18c140024f9f` 在 2026-09-26T17:09Z 的普通提交已命中
上述 `index.lock` 错误；17:11Z 的临时 `commit-tree` 调用随后停在 stale
code-mode cell，未留下 Git 子进程或可用 commit object。后续 handoff 因而保留源
文件并由 root 做 byte-exact 集成，而不是让 owner 放宽 sandbox。

CLI `0.156.1` 的 `--add-dir <DIR>` 只是通用的额外可写目录选项，不是 Git metadata
授权；`codex exec resume --help` 也不列出该选项。只有在一次独立、可丢弃的 CPU
预检中证明**精确**目录和所需 object/ref 路径均可写、且不会扩大到其他 worktree
或共享配置后，才可以在初始启动命令前放置 `--add-dir <exact-path>`。现有 thread
的 turn 权限档案未被证明会因事后加该参数而改变；没有上述证明时，选择下面的
SHA handoff，而不是尝试覆盖 sandbox。

默认的 owner handoff 必须包含：

```text
COMMIT=none
COMMIT_BLOCKER=linked Git metadata is read-only (index.lock)
BRANCH=<exact branch>  BASE_COMMIT=<sha>
INTEGRATION_MODE=BYTE_EXACT_ROOT_IMPORT
FILES=<allowed path>:<sha256>:<mode>:<size>, ...
```

owner 对每个允许路径运行 `sha256sum`，确认进程已结束并保留原文件；不要为了造
commit 修改研究内容。root 在验收 owner 的 terminal handoff、路径白名单和 hash
后，才可在自己的可写集成工作树中机械复制同一字节序列，逐个重算 hash、运行
治理验证并提交。hash 不匹配、路径超出白名单或需要解释/修复内容时，退回 owner；
root 不借此接管分析、重写卡片或放宽资源权限。

## 登记与交接

新对话产生 thread ID 后，在 `docs/AGENT_REGISTRY.json` 新增一条记录：唯一的 `agent_key`、`display_name`、`conversation_id`、该代理实际使用的 `codex_home`，由它派生的 `session_root`/三个数据库路径/`rollout_locator`，以及角色、工作树、分支、资源边界和 handoff。身份键是 `(codex_home, conversation_id)`。若是在替换旧对话，把旧身份移入 `retired_conversations`，不要复用旧 thread ID。

运行 `docs/AGENT_COORDINATION.md` 第 1 节的注册表校验和 `python3 tools/verify.py --changed`。主代理再核对新工作树已知晓 `main` 的最新 `STATE`/`CAMPAIGN` 决定，然后按第 4 节派发一个明确 Goal；当前冻结的 Cm campaign 不因创建新代理而解冻。
