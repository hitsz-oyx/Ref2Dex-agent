---
name: create-ref2dex-agent
description: 显式科研角色的账号/provider绑定建立或迁移时使用；普通独立会话与已有角色任务派发不用。
---

# 建立科研角色绑定

普通用户会话默认独立，直接执行用户指令，不加入科研角色池、不写 registry。
仅当用户明确要求加入科研工作流或迁移已有科研角色绑定时使用本技能。
日常向已配置角色派任务使用 researchctl，不创建新长期身份或恢复旧 Goal。

## 最小上下文

读取 AGENTS、docs/workflow/README、docs/AGENT_ROLES.yaml 和 docs/CAMPAIGN.md。
仅当核对实际研究授权需要时读取 STATE/实验卡。新路径不读取旧 registry、Broker 数据库
或历史 thread；旧任务收尾规范在 docs/archive/workflow-v1，按需追溯。

固定角色为 root、agent_cm、agent_rl、agent_eval、agent_infra；职责来自角色 YAML。
现有角色的 runtime 可以替换，新增长期角色须有明确用户授权；worker 不创建 worker。

## 绑定与验证

1. 在授权工作区创建独立工作树；检查同名目录/分支和未提交修改，不覆盖用户工作。
2. 为绑定指定绝对的 CODEX_HOME、workspace、store 和 Node/Codex PATH。逻辑角色之间
   使用不同工作树/store；逐角色配置允许当前 CODEX_HOME 相同，此时凭据/额度相同。
   同角色备选保留工作树但使用不同账号目录/store。root 首次通过 bind-root 传入前台
   主代理 CODEX_HOME 与运行环境，后台使用保存配置；有历史的 campaign 不直接改绑 root。
3. 模型路径使用 Codex harness。选择真实 provider 依赖该账号的 Codex 配置，不能只改
   provider 显示名。凭据、本机 workflow.json、执行数据库不提交，工具输出不打印密钥。
4. 使用 Node 24 的绝对路径做有界版本与 HTTP 入口检查，不修改系统 Node 或全局依赖。
   loopback provider 每次调用应通过显式 binding.env 清除外部代理变量；仅设置 NO_PROXY
   不替代实际 HTTP 检查。连接失败或超时不证明服务可用，不用无限重试掩盖问题。
5. 在独立临时 campaign 做小规模 CPU/工程账号验证：实际 launch、read、身份归属、退出，
   两个账号不串用。工程 smoke 与真实模型回合分别验证，thread ID 不同不等于账号隔离。
6. 验收后写本机绑定。备选必须显式 verified:true；同一角色 provider 名唯一。指定 provider
   的任务不跨 provider；未知投递先 reconcile，禁止切账号盲目重投。

账号身份与有效配置被 store 封印。更换身份/provider/工作树使用新 store，不自动接管
非空未知 store。普通 OAuth token 刷新不改变账号身份。旧系统任务继续在旧系统收尾，
切换新派发前保证唯一 owner，不停止未知进程或改动独立会话。

执行前再次核对用户授权与资源边界。超授权动作不执行；持续自主期间记阻碍，不主动
请求用户选择。验证结果、可用性和配置位置应简短报告，不复制运行日志或凭据。
