# PW 工作树完整整合

用户授权将 `cm-pointflow-effect-pretrain` 完整合入当前 `main`，随后删除旧工作树。
固定分支提交为 `2324146395816dc5f4400fae5a091855b8a1b141`，包括合并后新增的
Goal-Point-WAM Probe 与 baseline 审计记录。旧 `epic-scene-flow` 和 `ref8`
路径是该工作树的软链接，并非另外两份 checkout。Git 分支和历史提交保留。

全部本地 outputs 已同步到主工作区，普通文件使用硬链接逐文件核对；旧 tmp
保留到 `tmp/retired-cm-worktree-20261011/`。独立 PointWorld clone、其他 ignored
文件、迁移清单和生成脚本保存在
[归档目录清单](../../outputs/archive/worktree-retirement-20261011/manifest.json)。
5 份用户草稿复制到主工作区原相对路径，保持未提交状态。
远端 NAS 数据仍按原实验卡声明保留，本次不下载、不删除远端数据。

移除旧路径前修复了本地符号链接。历史 JSON/checkpoint 记录中的旧绝对路径
保留原样以保护 hash；复现时按归档的 prefix_mapping/tmp_mapping 映射到新位置。
归档 tmp 中的历史虚拟环境不承诺迁移后直接可执行。

共享三源 corpus 另建 `data-main-worktree` 路径副本，改写 index 路径，保留原始
`data/processed/manifest.json` 的字节。所有 source/index hash 与 train-only
normalization source identity 保持一致。Goal-Point-WAM README 使用新入口。
此操作不修改模型、标签、数据 split 或科研结论，也不启动训练。

删除前检查 source HEAD、未提交内容、文件覆盖清单、当前进程 cwd/参数/打开文件，
并验证主工作区 corpus 实际读取和文档合同。源内容漂移或发现使用者则停止删除；
主工作区用户 Mission/Campaign 保持原样。只有已确认完整保留的 worktree 及两个
软链接可移除；未授权删除 Git 分支或历史产物。
