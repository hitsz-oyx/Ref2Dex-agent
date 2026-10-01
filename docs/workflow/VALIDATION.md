# 工作流重构验证记录

日期：2026-09-30。分支：`agent/workflow-runtime-adoption`。此记录只说明工程验证，
不形成科研结论，也不代表现有 campaign 已切换。

## 环境与结果

最终 Python 验证使用 `/home2/wyy/miniconda3/envs/graspenv/bin/python`，关闭 CUDA，
OMP/MKL 线程数为 2。执行后端使用 Node 24.19.0 和锁定的 Orchestrator CLI 0.1.0。
此前系统 Python 的验证由本次 graspenv 验证补充。

| 验证 | 结果 |
| --- | --- |
| 新工作流 CLI 集成与旧 broker/poller/supervisor 回归 | 94 passed |
| 治理测试（包含搬迁测试路径回归） | 16 passed |
| tools/verify.py --changed | PASS |
| 四个新入口/运行文件 mypy 类型检查 | PASS |
| 真实 Orchestrator 自定义 process smoke | PASS，输出 ENGINEERING_SMOKE_OK，模型调用 0 |
| 全仓 pytest | 收集阶段 10 errors，均为现有 Cm 代码引用缺失的 `src.task.Cm.dataset` 路径 |

新 CLI 测试覆盖暂停与恢复、独立 store/账号封印、身份变更、OAuth 刷新、预算、
幂等投递、丢失响应后的 reconcile、后台 owner/guardian 有限恢复、暂停竞争、
工作区/实验权限和拒绝未验收的模型 harness。全仓收集错误位于未修改的 Cm 科研
模块与其原有测试；本次没有修改这些科研实现。

回归命令：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q \
 tests/integration/test_research_workflow.py tests/test_workflow_orchestration.py \
 tests/test_agent_broker.py tests/test_agent_result_poller.py tests/test_codex_research_supervisor.py
```

## Standards Review

并行审查和修复后复核通过。关闭的问题包括：慢外部读取占用暂停写锁、暂停时
消费尚未应用的 root 决策、暂停被恢复失败计数覆盖，以及新增测试目录归属。

## Spec Review

并行审查和修复后复核通过。关闭的问题包括：身份封印遗漏隐式账号/provider
配置、角色实验权限与授权工作区未验证、已验收历史记录读取失败阻塞新工作。
模型执行限定为已检查的 Codex harness；其他原生模型 harness 不能借工程标记绕过。

## 第二阶段：持续推进与文档精简

在同一 graspenv 环境完成 CLI 行为切片和回归：106 passed；8 个入口、运行与索引
模块 mypy 通过。审查补充 7 项入口回归，CLI 现有 34 项；provider 专项 6 项通过。新增覆盖无默认总截止、持久用户指令、旧 root 决策失效、受阻
等待、正式 Validation 完成门槛、provider 备选/指定、独立工作树保护、决策记录、
停止后台 owner 时保留现有任务，以及实验索引不改原卡。

真实 CPU-only 工程联调使用临时独立工作树、store 和现有账号配置副本：root 的
session_meta 为 openai，infra 为 rlg；worker 输出 WORKFLOW_PROVIDER_OK，关闭
前台调用后由 detached supervisor 的 root 验收。联调发现并修复 worker 在 root
回合中完成时被 idle 状态漏掉的竞态；现在用 root 实际观察的指纹判定新事件，
并有公开 CLI 回归。联调完成后停止自己的 owner，等所有模型回合自然结束，
删除临时凭据副本；原始账号文件、科研绑定和训练进程未修改。

全仓 pytest 仍在收集阶段报上述 10 个既有 Cm 路径错误。本记录不把结构化
Validation 验收等同于科学事实成立；root/eval 仍须核对预注册、原始结果与方法。

## 第二阶段审查

Standards 轴：3 项原发现已关闭并复核通过。派发拒绝无法确认的 Git 工作树；恢复通知
不会被旧 provider 观察覆盖；root 全部备选保持 Codex。

Spec 轴：4 项原发现已修复。最终执行绑定也检查工程/研究能力，CODEX_HOME 使用解析
路径全局隔离；确认终态连接故障有限重试后可切换，root 同样覆盖；重要决定以稳定 ID
更新结果，并显示研究范围与启停状态。未知投递和不可读执行仍须 reconcile，不推断
失败或切换来重发。Standards 与 Spec 最终只读复核均通过；7 项新增回归分别观察过失败再修复。

## 上线边界

现有 campaign 尚未切换。真实两 provider 工程链路已验证，其他账号/provider、
长期科研验收和旧新 owner 切换仍须按 [README.md](README.md) 操作。
依赖的 6 项 npm audit 公告尚未修复，限制见同一入口的依赖审计说明。

## 容量与网关恢复：2026-09-30 后续实施

用户确认的局部恢复设计已实施，规则以 [README.md](README.md) 为准。
Selected model is at capacity 在失败结束后等待 60 秒，仍沿用原任务/provider；
网关初始失败后最多 3 次恢复尝试，不切 provider。root 耗尽后 attention，worker
耗尽只停止该任务恢复；现有其他实验可收尾。明确预算不清零。

graspenv、CUDA 关闭：公开入口 43 项与治理 16 项由 tools/verify.py --changed 通过；
旧 broker/poller/supervisor 与索引回归 79 passed。4 个受影响运行模块 mypy 通过。
测试包含实际等待一分钟、暂停、原任务执行链、响应丢失核对、无会话时的契约续接、
原生 Goal 查询期间暂停、GPU 等待后重新准入、首次故障禁止提前驳回，以及新目标
释放旧失败角色。全仓 pytest 仍为上述 10 个既有 Cm 收集错误。

真实 Orchestrator 0.1.0 加模拟 Codex 进程验证通过，模型调用 0：通过公开 events
读取 thread.started 元数据，resume 使用同一 provider thread 并产生新执行 ID。
普通 read 不包含 provider ID；实现不读取后端私有状态文件，也不发送 slash 命令。
本轮没有调用真实账号或切换现有科研绑定。

### Standards

最终只读复核通过。关闭 root 耗尽后误报 active 和原生 Goal 使用旧快照的问题。
未提交修改无法可靠确认归属，保留原内容并停止自动续接，现已明确列为保护限制，
不宣称可以自动恢复任意脏工作树。

### Spec

最终只读复核通过。关闭恢复绕过 GPU 准入、首次故障提前驳回和旧目标失败永久
占用角色的问题；新派发与恢复共用并发资源检查。没有未关闭审查发现。

## 合并主分支：2026-09-30

由用户明确授权，将工作流分支 4f42f65 合并进 main。集成在独立 main 工作树完成，
没有 Git 冲突；运行代码与已审查工作流分支一致。STATE 去除分支迁移说明并保留
主分支已有结论边界，原主分支完整 STATE 与归档逐字核对一致（仅调整相对链接）。
没有纳入独立研究分支尚未交付的科研结果。

新 main 工作树以 graspenv、CUDA 关闭运行 tools/verify.py --changed，结果 PASS；
实验索引一致性与 git diff --check 通过。全仓测试既有的 Cm 导入错误及依赖审计
限制沿用前述记录。此合并未部署新 campaign、启动生产 owner、改写 runtime binding
或停止已有实验；没有向远程 push。

## 逐角色账号配置与前台 root 传入：2026-10-01

用户确认各 worker 分别配置主、备用绑定，当前四角色均使用 .codex_oyx，usage limit
后切换 .codex_oyx_frj。允许跨角色 CODEX_HOME 相同，但工作树/store 独立；相同目录
仍代表相同登录凭据与额度。角色内主备目录必须不同。样例逐项写出，真实备用验收前
保持 verified:false；本轮不启动生产 campaign 或修改旧科研绑定。

bind-root 仅在 owner 已停止、paused 且无执行历史时捕获前台明确传入的 CODEX_HOME
与运行环境，原子保存本机配置。后台使用保存的配置；环境之后改变不会改绑账号。
不自动读取聊天记录或临时 model/profile。历史 campaign 保留原绑定；前后台沿用唯一
owner 锁和持久指令入口。验证过程复制备选配置，避免归一化误触配置一致性检查。

usage limit 的失败执行通过同一逻辑任务进行账号交接。旧、新执行记录分别保存 store，
不跨账号 resume thread；恢复意图先落盘，丢失响应在目标 store 核对，不重复启动。
无可用备选时等待，其他角色可继续；Pause 不启动交接，容量/网关策略不变。

公开 CLI 新增 6 项工程回归：逐角色相同目录、usage limit 交接、目标 store 丢失响应
核对、Pause 与无备选等待、root 捕获/拒绝运行中改绑、guardian/历史 campaign 保护。
其中相关 11 项回归通过，5 个修改的运行模块 mypy 通过；模型调用 0，未做真实账号
可用性验收。以 3b775eb 为固定比较点，最终 tools/verify.py --changed PASS：
49 项公开 CLI 与 16 项当前治理测试全部通过；Markdown 链接与 diff 检查通过。

全仓 pytest 仍有既有 10 个 Cm 导入收集错误；额外运行全部治理测试发现 5 项旧 Goal
文档检查仍读当前跳转入口，未切换至历史规范。相关测试/文档与 3b775eb 完全相同，
main 上单独重跑同样失败；本次不将旧 Goal 合同重新塞回当前入口。

Standards 与 Spec 分别只读复核，均无未关闭发现；真实备用账号验收与生产 owner
切换仍是部署条件，不以工程控制测试代替。
