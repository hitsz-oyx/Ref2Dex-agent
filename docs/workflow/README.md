# 单会话容量看门狗

仓库只保留一个运行时辅助进程：
[`scripts/codex_research_supervisor.py`](../../scripts/codex_research_supervisor.py)。它源自旧
的 Codex supervisor，但已经去掉 root/worker、Broker、任务租约和研究决策。

看门狗会：

1. 扫描 `/home2/wyy/oyx_ws` 下所有 `.codex*` CODEX_HOME；
2. 只考虑最近 24 小时更新且未归档的 thread；
3. 在 rollout 中发现 `Selected model is at capacity`、`server_overloaded` 等容量错误后
   记录恢复状态；
4. 线程空闲且没有排队输入时，每 60 秒最多发送一次 `继续`；
5. 看到新的 turn 开始后清除该线程的恢复状态，超过活跃窗口后自动忘记。

它不会读取实验指标、创建子代理、改变 Goal、绕过用户暂停或重置预算。状态和锁只写入
`.runtime/`。

## 预检

先只读检查，不发送消息：

```bash
python3 scripts/codex_research_supervisor.py \
  --dry-run --once \
  --state .runtime/session_capacity_watchdog/state.json
```

确认输出和状态文件后再启动实际看门狗。需要发送消息时显式提供 Codex CLI 或 Node
入口：

```bash
python3 scripts/codex_research_supervisor.py \
  --codex-node /home2/wyy/.nvm/versions/node/v24.19.0/bin/node \
  --codex-js /home2/wyy/.nvm/versions/node/v24.19.0/lib/node_modules/@openai/codex/bin/codex.js \
  --state .runtime/session_capacity_watchdog/state.json
```

停止进程即可停止看门狗；它不会停止或修改任何 Codex 会话和研究进程。
