# 工作流操作指南

## 安装与配置

需要 Python 3.8+、PyYAML 和 Node 24+。执行层依赖固定为 Orchestrator CLI 0.1.0，
版本与完整性锁在 `tools/workflow_runtime/package-lock.json`。工作树内安装：

```bash
npm ci --prefix tools/workflow_runtime
```

确保上述 npm 来自 Node 24 的 PATH，不修改系统 Node 或全局包。配置样例见
[workflow.example.json](../../configs/workflow.example.json)。复制为本机
`.runtime/workflow.json`，填写绝对路径、每个角色的独立账号和 worktree。
command 为 argv 数组，建议显式写 Node 24 和本地 CLI 的绝对路径；env.PATH 也要
使后台子进程找到正确的 Node/Codex。模型凭据仅放本机配置或各自账号目录。

首次启用前运行[工程 smoke](../../tools/workflow_smoke.py)，再用有界真实账号任务
验证每个账号的身份、root JSON 输出和后台续接。不要仅凭 thread ID 不同认定账号
隔离。对运行中的实际 campaign，此门槛完成前保持旧派发入口。

## 命令入口

新命令默认读取 `.runtime/workflow.json`，可用 `--config` 选择独立配置。

```bash
python3 scripts/researchctl.py supervisor status
python3 scripts/researchctl.py supervisor pause
python3 scripts/researchctl.py supervisor resume --legacy-dispatch-disabled
python3 scripts/researchctl.py supervisor run --detach
python3 scripts/researchctl.py --config /absolute/workflow.json supervisor tick
python3 scripts/researchctl.py dispatch --task /absolute/task.json
python3 scripts/researchctl.py reconcile
python3 scripts/researchctl.py accept T-example --decision accepted --reason '检查的证据及结论边界'
```

`--legacy-dispatch-disabled` 是操作者对旧 owner 已停止派发的确认，命令不会替你停止
未知进程。暂停只影响新派发，现有实验不会被 kill。后台运行期间，其他前台
dispatch/accept/tick 会因 owner 锁拒绝；前台可随时 status/pause/resume。

后台日志为 state_dir 下的 `supervisor.log`。`attention` 表示需要检查身份、未知
投递或连续故障；`budget_limited` 表示预算耗尽。先查看账号隔离的 Orchestrator
read/logs/events，再 reconcile，不能对未知投递盲目改 task_id 重发。
resume 重置恢复计数，不重置花费预算；新的授权 campaign 使用新的 state_dir。
暂停后若 guardian 因意外退出而停止，应在恢复时重新运行 `run --detach`。

任务格式见 [task.example.json](../../configs/task.example.json)。root 将自行生成
同一格式；手动 dispatch 主要用于有界 smoke 或部署验证。

## 旧任务收尾

旧命令需显式选择，例如：

```bash
python3 scripts/researchctl.py --legacy supervisor pause --lease .runtime/SUPERVISOR_LEASE.json
```

旧 lease/registry/broker 参数也会选择兼容路径，保留已有启动脚本；新路径与旧
数据库无依赖。先停止旧的未来派发和恢复 owner，保留已启动实验及产物，再启用新
监督。旧兼容代码的保留是收尾需要，不代表可以并行开启两个派发 owner。
