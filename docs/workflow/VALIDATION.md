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

## 尚未执行的上线验证

未调用真实模型，未修改现有绑定、训练进程或 campaign 的派发入口。真实多账号
身份、provider 输出、后台 root 回合和旧新 owner 切换仍按操作指南验证。
依赖的 6 项 npm audit 公告尚未修复，具体限制见 [OPERATIONS.md](OPERATIONS.md)。
