# PointWorld 合并前归档与复现

2026-10-11 用户选择以 `cm-pointflow-effect-pretrain` 的实现为主，将旧 PW
实现及实验保留为历史记录。主分支合并固定提交
`5f12ae31cc725a9f5d660ba476456a77c47d94b9`，包括修复后的固定 voxel grid、
sharedstats 基线、PointWAM 工具以及 Goal-Point-WAM 三域工程 smoke。
最终研究目标仍是验证世界模型对 RL 的帮助；本次合并不形成新的科研结论。

## 旧数据与代码

归档标签 `archive/pw-premerge-20261011` 指向合并前主分支
`102410de1cc5acf8733465ca2ee466465ce52643`。旧数据和 checkpoint 原位保留，
不改名、不删除、不覆盖，以免破坏实验 manifest 和路径引用。
[归档清单](../../outputs/archive/pointworld-legacy-20261011/manifest.json)
记录 54 个历史产物目录、checkpoint 身份和用户文件摘要。
其中共享原始数据仍可被新路线读取；归档不撤销原始数据的使用许可。

旧 `pointworld-main3-20261007/train-action/best.pt` 的 SHA256 为
`f70573c4d06d368a7d8ab921ff03c936bb0e1b8565df9363b0f1faeb9ab6ba13`，
训练身份提交为 `9019fd467c2803888e031b43eda5c505b14e4021`。
旧 checkpoint 的 batch-relative voxel 原点与当前固定 grid 不同。
参数能加载并不保证预测语义相同，不能直接用当前实现复算旧 PW 结果。
现有来源 hash 校验保持严格，旧 PW cache/audit 入口仅用于历史复现。

需要复现时，从上述归档标签读取当时的 policy/cache/audit 工具，再依据
checkpoint 的 `identity.implementation_sources`、`vendor_sources` 和训练提交
核对完整模型实现。使用独立复现目录和新 run_id，保持原始产物只读；
不要在当前主分支覆盖新实现，也不要关闭来源校验。创建复现分支仍须遵守用户授权。

## 产物同步

源 worktree `outputs/` 中 44,467 个普通文件及 8 个软链接已同步到主 worktree。
普通文件使用同盘硬链接，逐文件核对 device/inode/size；没有覆盖目标文件，
也没有新增约 46.7 GB 的数据副本。完整同步 inventory 在
`tmp/integrate-cm-20261011/artifact-sync.json`。
远端 NAS 的 EgoDex 数据仍保留在实验卡声明的远端路径，本次没有下载它们。
两处硬链接共享文件内容，实验产物应保持只读；新运行必须使用新输出目录。

## Decision Note

问题是接受新 PW 实现并保留历史复现能力。关键证据是旧 checkpoint 的六个
来源文件与新实现不同，而 sharedstats 修复及新路线已有工程证据。
按用户选择合并新实现，保留主分支已有 DExplore/DAgger 状态与历史结果，
原位归档旧 PW，按 inode 同步产物。这是文件与代码整合，不启动训练或仿真。

合并前两个 worktree 的 outputs/tmp 去重后已占用约 313.6 GB，超过 Campaign
300 GB 上限；本次不增加大文件占用，也不放宽上限。后续新增大规模产物前须
先处理既有资源超额，不能为清理空间删除归属不明或未确认可删的证据。
停止条件是用户文件发生变化、产物碰撞或校验失败；先解决这些问题再提交。
本次没有外部不可逆操作，也没有新的长期身份或研究 claim。

## 整合验证

合并后的 cm-pointflow-effect-pretrain、trajectory-policy 和 goal-point-wam
合同测试共 176 passed / 6 skipped；未启动新的 GPU 训练或物理回放。
这些小型接口测试和文件检查使用 CPU，以免为整合检查占用 GPU；它们不检验科学收益。
`tools/verify.py --changed` 通过文档链接、实验卡 schema 和 seed 合同检查。
Goal-Point-WAM smoke 卡的缺失必填身份字段已补齐，扩展卡中 clips/records 的
文字混淆已修正。用户的 AGENTS、Mission、Campaign 均按 SHA256 核对原样保留。
