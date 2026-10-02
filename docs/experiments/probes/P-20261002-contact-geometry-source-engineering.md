# 随机组合源采集：工程完成

GPU访问已恢复，实际CUDA矩阵运算通过。物理GPU0被其他用户占用，本次
选择空闲GPU1；单卡隔离后原生进程中的cuda:0对应物理GPU1。
原生仿真与独立记录审计均已完成。这里只确认可执行性，不构成新候选优势、
Cm信息或闭环utility结论；[采集合同](P-20261002-contact-geometry-source.md)
和原目标的真实执行、排序、重新决策要求保留。

离线动作合同：HF18 seed551实际193个pre-state、1158个新组合命令；XYZ与
每指动作在六专家范围内，40个历史base/20个cup原生PD复现误差0。权重和
最大误差2.38e-7。见[离线检查](P-20261002-contact-geometry-action-engineering-r1.json)。
独立审计软件检查拒绝10类synthetic损坏，包括最后post位置/速度、PD、
概率、reset和group split；[该记录](P-20261002-contact-geometry-record-audit-engineering-r1.json)
明确是内存fixture，不能代替真实物理数据。

恢复后的第一次native尝试r1在仿真启动前失败：采集器只接受输出根目录
的直接子目录，而启动器使用run/seed570嵌套目录。修复为允许任务根目录
内的新独立目录，继续拒绝越界、符号链接及覆盖；修复提交1ebad5d。
r1失败目录与旧提交输入保留。先前GPU0 admission被占用而拒绝，也计入成本。

r2原生工程采用[新计划](P-20261002-contact-geometry-native-engineering-plan-r5.json)：
seed570、96env、early/clear每层最多1窗口、650tick/240秒；整体600秒/256MiB。
完成108个H10窗口、82episodes，初始clear26；8类分配计数
16/20/13/16/10/8/17/8。全部六专家在保存观测上回放最大误差0。
原生父/子进程完成，工具返回exit0，全部34输入hash未变。
未执行候选仍没有逐状态真实后果，不能从此声称oracle或真实regret。

[独立GPU审计](P-20261002-contact-geometry-native-engineering-audit-r2.json)完成
1080步实际命令、PD、合并propensity、固定旋转、rawforce存在代理、完整mesh
间隙及H10联合标签复算。固定旋转/mesh/支持标签误差0；PD最大1.19e-7；
pre/post相对位置最大3.28e-7m，速度1.91e-6m/s，均通过原限制。最后post
帧亦验证，全部窗口完整且不跨reset。审计工具返回exit0。
力只是存在代理，掌部/指端是link原点，不声称已经识别真实接触点或接触对。

[工程完成记录](P-20261002-contact-geometry-native-engineering-completion-r2.json)
保存终态/hash与成本：成功原生118.38秒、失败4.60秒、独立审计3.75秒；含离线、
计划、GPU admission失败和额外检查的保守累计194.82秒，产物16.91MiB。
旧[GPU阻塞证据](P-20261002-contact-geometry-gpu-blocker.json)与
[恢复记录](P-20261002-contact-geometry-gpu-restored.json)保留。
目前没有后台采集或训练进程，本工程不消耗科学Probe slot。下一步登记随机
组合真实监督与物理作用模型的新科学合同/预算，再采集；完整goal及C3仍OPEN。
