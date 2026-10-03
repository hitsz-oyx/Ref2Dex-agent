# Worktree cleanup record — 2026-10-03

本记录对应外部归档目录：

`/home2/wyy/oyx_ws/ai_ws/.ref2dex-branch-archive/20261003/`

完整的工作树状态、diff、未跟踪文件副本和 ignored 文件清单保存在该目录；紧凑机器清单见
[`worktree_cleanup_manifest.json`](./worktree_cleanup_manifest.json)。

清理前的 Git 状态由提交 `15f24aa` 和标签 `snapshot/cm-20261003-audit` 作为代码基准。
本次只移除干净且没有检测到运行进程使用的 worktree，所有分支和提交仍保留。

已移除的 worktree：

- `Ref2Dex-agent-main` → `main`
- `Ref2Dex-agent-workflow-accounts` → `agent/workflow-accounts`
- `Ref2Dex-agent-workflow-adoption` → `agent/workflow-runtime-adoption`
- `Ref2Dex-agent-workflow-v22-audit` → `agent/root-goal-lifecycle-test-20260927`

保留原因：

- `Ref2Dex-agent-baseline` 是 Git common directory，同时含大量实验数据；
- `Ref2Dex-agent` 有当前未提交修改和运行时 supervisor；
- `Ref2Dex-agent-cm`、`Ref2Dex-agent-infra`、`Ref2Dex-agent-rl` 有未提交或未跟踪内容；
- `Ref2Dex-agent-contact-response` 在清理时仍有 native replay 运行。

另外，已将旧的 workflow handoff 和一个历史 Cm probe 的两个结果文件复制到外部归档后，删除了
对应的孤立目录 `Ref2Dex-agent-workflow-v22`、`ai_ws/ai_ws`，以及空的
`Ref2Dex-agent-runtime`。

ignored cache/runtime/output 文件只记录路径清单，没有重复复制；保留的脏 worktree 没有删除其数据。
