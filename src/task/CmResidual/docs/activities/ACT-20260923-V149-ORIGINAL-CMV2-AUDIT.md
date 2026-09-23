# V1.49 原版 Cmv2 真实转移审计活动

- date: `2026-09-23`
- branch: `agent/v149-cmv2-audit`
- run_status: `COMPLETED`
- official actor checkpoint: `null`

固定 checkpoint、真实转移、抽样、smoke、
GPU/产物预算、停止条件和科研门槛见 experiment card。
外部 Cmv2 checkpoint 只读软链接到工作区；
本审计不修改/训练策略或原版模型。

第一次8样本 smoke 在导入旧 CmDecoder 链时因
无关模块缺失提前退出，没有创建输出目录或模型
推理结果。修复为脚本内独立姿态差运算、增加
2项测试后，以 commit `259496e` 重跑成功。
正式单卡 GPU5、固定256样本审计 `COMPLETED`，
run manifest 记录原版/CmLite/转移 SHA、代码
commit、microbatch8、peak GPU allocation；
`audit.json` 记录每 seed/分层 EPE、动作打乱、
token 激活、延迟和参数量。产物<1GB，外部
项目只读，GPU6与其他进程未受影响。

原版在 seed95/96 接触样本 EPE 39.048/52.739mm，
零预测5.360/10.349mm，CmLite3.477/6.748mm；
原版名义手流接法的离线门槛失败。此负结果
不能泛化为 Cmv2 架构整体，需离线 oracle
真实手流实验分离手动力学与表征问题。
