# Issue tracker: Local Markdown

本次工作流重构使用仓库内 Markdown issue，避免在未配置远程 tracker 时发布远程内容。
一项工作对应 `.scratch/<feature>/spec.md`，顶部 `Status:` 表示 triage 状态。
发布即创建该文件；默认 ready-for-agent。用户可运行 /setup-matt-pocock-skills 更换 tracker。
