# D-20261001 — 废弃多代理工作流并恢复容量看门狗

## 当前需要决定的问题

旧 root/worker、Broker、固定角色和多套 workflow 文档已经与当前会话工作方式冲突；容量错误
仍需要一个低权限的自动续接入口。

## 关键证据

* 当前仓库存在多套互相覆盖的 Broker、root watchdog、poller 和历史 plan 入口。
* 历史 `codex_research_supervisor.py` 已具备 SQLite thread 读取和 `codex queue` 能力。
* 历史 `workflow_recovery.py` 的容量策略是等待 60 秒后续接，但它依赖将被废弃的 Broker。

## 选择与理由

删除旧规范和 Broker 运行时，保留研究实验、决策、交接和日志证据；以历史 supervisor 的
读取/排队机制为基础，使用一个单进程看门狗扫描本机 `.codex*`，仅在容量错误后发送 `继续`。
当前会话不创建或派发子代理。

## 成本、停止条件与边界

看门狗只读 Codex SQLite/rollout，并把状态写入 `.runtime/`；它不修改仓库外数据、不解除
用户暂停、不重置预算。发现 CLI 协议变化、无法确认 thread 归属或 queue 状态不可读时跳过
该会话并记录，不盲目重试。

## 收尾审计记录

2026-10-01 收尾检查发现两个仍在运行的历史守护进程：本仓库的
`scripts/root_watchdog.py`（PID 3426912）和 `scripts/codex_overload_watchdog.py`
（PID 3459148）。它们属于已退役的 Broker/root workflow，不承担当前科研实验；按本决策
停止这两个明确归属本仓库的进程。停止后只保留
`scripts/codex_research_supervisor.py` 作为容量故障看门狗，并重新做一次 `--once` 预检。
