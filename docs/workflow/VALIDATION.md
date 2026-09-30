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
