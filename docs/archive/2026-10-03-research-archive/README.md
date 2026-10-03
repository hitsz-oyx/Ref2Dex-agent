# 2026-10-03 research archive

这是首个研究归档批次，目标是让 2026-10-03 前后的 Cm 实验可以从当前索引回溯到原始卡、结果文件、Git 提交和工作树状态。

## 入口

- 当前研究入口：[research index](../../research/INDEX.md)
- 机器可读清单：[manifest.json](manifest.json)
- 根目录 `docs/STATE.md` 的工作树快照：[STATE_20261003_prearchive.md](STATE_20261003_prearchive.md)

根目录 `docs/STATE.md` 保留不动，只作为旧 handoff、decision 和 hash 记录的 legacy 兼容路径，不再承担当前研究入口功能。当前文件在归档时含有未提交修改，因此归档副本明确标为 working-tree snapshot，不把它当作干净科学快照。

## 回溯规则

每条实验记录以 experiment card 为准；每条科学判断还需要结合 manifest 中的 commit、branch、artifact hash、dirty paths 和 worktree 状态。归档摘要是导航，不替代原始证据。

## 本批次验证

归档工具、索引链接和 repository verification 的命令与结果记录在 manifest 的 `validation` 字段中。现有未提交的研究修改、未跟踪 skill 和其他工作树内容不自动纳入 archive commit。

## Worktree 处置

本批次保留 baseline primary worktree、当前 `agent/cm-residual-policy`、`agent/cm`、
`agent/contact-response-cm`、`agent/infra`、`main` 和 `agent/rl`，因为它们存在未提交
内容或承担 canonical/reference 作用。保留含有 ignored 运行产物的 workflow worktree，
不直接删除其目录。已移除以下 clean 且无 ignored 文件、无进程使用的 checkout；对应
branch/ref 仍保留：

- `agent/current-policy-value-supervision`
- `agent/runtime-agent_cm`
- `agent/runtime-agent_eval`
- `agent/runtime-agent_infra`
- `agent/runtime-agent_rl`
- `agent/runtime-root`

baseline 目录是 Git common-dir，不属于可删除的普通 worktree。所有路径、HEAD、dirty
状态和剩余 worktree 均以 `manifest.json` 为准。
- [2026-10-03 worktree cleanup](./WORKTREE_CLEANUP_20261003.md)：旧 worktree 的状态快照、清理范围和分支保留记录。

本次文档清理还删除了重复的根级 handoff/收尾文件，并将退役的工作流设计、验证记录、
旧治理活动和旧 ADR 移入 `workflow-v1/`、`activities/governance-v1/` 与 `adr/`。
对应的 Git 清理前快照为 `snapshot/oracle-before-doc-cleanup-20261003`。
