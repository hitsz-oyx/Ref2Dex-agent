# 上下文窗口

默认只读本文件，然后读下面标成必读的文件。归档只有标题在这里。不要打开归档，除非当前问题点名了那一条的日期或卡片。

窗口可以继续变长。一条决定关闭之后，把它从 `recent.md` 移到 `archive/`，这里只留一行。不要把归档正文贴回必读。

## 必读

- [recent.md](recent.md)
- [dispatch.json](dispatch.json)

## 归档

| 日期 | 内容 | 文件 |
|---|---|---|
| 2026-10-23 | 九源诊断 C1–C5 停住。C2 在旧原点退回，C3/C5 没有质量结论 | [archive/2026-10-23-c1-c5.md](archive/2026-10-23-c1-c5.md) |

## 固定指针

- 仓库：https://github.com/hitsz-oyx/Ref2Dex-agent
- 当前战役文档以本窗口所在提交为准。不要靠聊天记录补提交号。
- GPT 只延续这一条对话：https://chatgpt.com/c/6adc1912-f734-83ea-bbf2-dc3b6e341bab 。其他对话不在它的上下文里。
- Codex 用本机这条线程：`01a16c55-5a34-77f0-b0a2-c5205fa3e161`。家目录是 `D:/project/stock_ws/.codex_oyx_frj`，模型保持 `gpt-6.1-sol`。续写这条，不另开。
- 调度入口是 `tools/dispatch_step.py`。不加 `--execute` 时只报告动作，不联系 GPT，也不启动 Codex。
