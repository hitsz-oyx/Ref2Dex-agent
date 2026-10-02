# 冻结Cm生成动作：原生工程合同

工程目的：接入[GPU生成器](P-20261002-optimized-contact-actions-engineering.md)，
实际随机执行生成程序并独立复算。不是科学Probe，不形成候选机会或utility结论。
旧HF19失败门及slot2关闭保持；完整goal、Mission和最终claim不变。

单GPU1空闲时运行seed590/assignment15590、96环境、首episode、650 tick、
每early/clear最多一个窗口、H10。触发仍为当前连续3步联合净力presence、
rest高度至少5mm、warmup10、足够剩余步；不按未来或当前速度筛选。
执行前生成8个程序：base、rotation_cup、冻结Cm梯度生成、直接评分生成、
shuffled生成、3个随机组合。10槽均匀分配，base/cup各2槽，其余各1槽。
各生成器固定32步/lr .15/clip1、同专家simplex、同起点旋转锚；模型不训练。
全部候选形成后才随机分配；H10保持系数、每步六专家真实反馈更新。

[采集器](../../../scripts/collect_optimized_contact_source.py)明确标记Cm参与和
输入系数优化；保存每批完整pre-only输入、全部随机和最终权重、三个模式
33步规划轨迹、成员预测、候选初始指令，以及真实10步全bank/PD/pre/post/
世界关键点/原始力/meshCLR。保存批次顺序以精确重放批量优化及分配RNG。
[独立审计](../../../scripts/audit_optimized_contact_source.py)复算实际动作、
PD耦合、范围、旋转、物理标签、全部pre/post几何；重新组装在线输入、
重放随机proposal/allocation/32步规划，并用完整冻结NN核对最终后果预测。
检查checkpoint/专家参数冻结及所有输入hash不漂移。

[启动器](../../../scripts/run_optimized_contact_engineering.py)只运行一工程phase；
原生240秒、父进程290秒/300秒硬限、256MiB输出；审计另设300秒硬限。
目标合计不超过5分钟；若实际超出目标，记录真实成本并停止扩大规模。
只终止自己启动且有句柄的子进程，不影响他人GPU。缓存/输出限仓库内独立目录。
工程通过后才登记科学候选机会卡；不从此卡启动PPO、完整成功率矩阵或长期V。
