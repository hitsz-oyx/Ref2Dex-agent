# Ref2Dex 工作流

当前工作流只有三个规范入口：

- [架构与行为合同](ARCHITECTURE.md)：研究主管、执行层、暂停、恢复和验收。
- [操作指南](OPERATIONS.md)：配置、启动、查询、暂停、收尾与迁移。
- [固定角色](../AGENT_ROLES.yaml)：角色职责和资源权限的唯一 tracked 定义。

[重构规格](../../.scratch/workflow-runtime-adoption/spec.md)与[设计讨论](REDESIGN.md)记录来源，
不增加另一套运行规则。历史 v1 合同位于 [archive/workflow-v1](../archive/workflow-v1/)。
本机绑定与控制状态留在 `.runtime/`；研究证据仍属于实验卡、manifest 与 Git。
