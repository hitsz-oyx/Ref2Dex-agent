# 随机组合源采集：当前工程状态

离线动作合同已通过，原生工程尚未启动，新候选机会/Cm信息/闭环utility未证。
[采集合同](P-20261002-contact-geometry-source.md)保留原目标的真实执行、排序
和重新决策要求，不把本工程结果当成科学Probe结果。

从已审计的HF18 seed551实际记录读取193个pre-state，生成1158个新组合命令。
XYZ与每指动作都在六专家坐标范围内，固定旋转PD误差0；40个历史实际base
和20个实际rotation_cup的首步PD目标均完整复现，误差0。随机生成可重复，
不修改专家bank；非法非凸权重会拒绝；重复base/cup合并概率.2，其余.1。
权重和最大误差2.38e-7。仅CPU小型命令代数，未生成新物理后果。
见[离线记录](P-20261002-contact-geometry-action-engineering-r1.json)。

[采集器](../../../scripts/collect_contact_geometry_source.py)保存固定系数、完整
候选集、10步全专家bank/实际PD/原始力、pre与post真实link位置和速度。
窗口起点旋转锚定，每步新专家反馈，early/clear各限额，旧模型和输入不改。
源码语法检查通过；尚未在原生环境跑过，不能声称整采集器runtime已通过。

[启动器](../../../scripts/run_contact_geometry_engineering.py)默认只生成可审阅
计划；显式native入口仅运行一个用于工程检查的phase，96env、每层1窗口、
650tick/240秒，整体600秒/256MiB。34个输入文件hash及已有专家/motion来源
已核对，见[计划](P-20261002-contact-geometry-native-engineering-plan-r2.json)。
GPU admission在创建原生运行目录之前完成，设备不可用时不写假运行状态。
本次未调用native入口，计划产物不是live job，无进程在后台等待。
R1计划已被未执行的R2计划替代，旧产物保留；补齐全10步post-native观测，
独立审计覆盖每个pre/post相对位置与速度、分配/PD/force/mesh与联合标签。
[审计工程检查](P-20261002-contact-geometry-record-audit-engineering-r1.json)通过
30步内存fixture，10类损坏均拒绝，包括最后一帧位置/速度、PD、概率、
reset及group split；明确是synthetic软件检查，不是新物理转移。

当前连续三轮nvidia-smi exit9、CUDA0设备，执行环境缺少/dev/nvidia0和
/dev/nvidiactl；[设备边界证据](P-20261002-contact-geometry-gpu-blocker.json)已保存。已请求用户提供现成执行入口或恢复
访问的方法；不升级权限、不改系统、不启动CPU替代物理仿真。不再用旧动作
数据冒充新组合监督。设备恢复后先做此原生工程与独立记录审计，再登记下一
科学源采集/拟合合同；工程准备完成，下一步必须执行原生phase；没有可继续的真实新组合数据。
完整goal仍未完成，满足三轮同一外部阻塞条件，执行状态将设为blocked；
恢复GPU访问后才能继续原生工程、新源采集与科学Probe。
