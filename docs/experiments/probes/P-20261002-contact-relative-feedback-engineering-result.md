# 当前手物相对运动反馈：原生执行工程通过

run_status: COMPLETED
scope: engineering only; no scientific Probe classification

固定0.5增益，保留Cup逐步XYZ/手指名义指令和起始腕旋转。静态排除623/590
共129观察窗口，独立command误差0、未来字段污染和第一步位置候选等Cup通过。
读取实际URDF三个世界轴平移关节并核对native DOF顺序与恒等reset根变换。

排除seed669原生重跑完成49H10/37episode/490步：base24/Cup12/速度反馈6/
位置反馈7。实际速度反馈6窗口/60步、位置反馈7窗口/63步PD不同于同观测Cup；
位置第一步与Cup相同、随后才响应，不据初始指令相同排除整段程序。
手指保持相应专家、旋转锚定，逐步完整六专家重放差0。独立随机分配、current
输入、native PD、原始力接触、全mesh clearance、前后观测相对几何全部通过。
同观测Cup countercommand只证明动作改变，不是未执行Cup轨迹真值或regret。

首次运行因遗漏player批量初始化而在首次推理失败，parent42.010秒保留。
真实rl_games接口RED/GREEN确认根因，只补get_batch_size；gain/seed/RNG不改。
失败与修正合并后累计162.771秒，原生与审计exit0，全部自己PIDterminal/GPU释放。
工程不评价增益、没有Cm拟合/推理；固定独立候选机会Probe后才能判断是否值得学。
核心目标与HF23失败标签不变，C3仍OPEN。

证据：[设计](../../decisions/D-20261002-contact-relative-feedback.md)、
[批量修正](../../decisions/D-20261002-relative-feedback-batch-initialization.md)、
[静态GPU检查](P-20261002-contact-relative-feedback-static-r1.json)、
[完整原生manifest](P-20261002-contact-relative-feedback-native-engineering-r2.json)、
[独立执行审计](P-20261002-contact-relative-feedback-native-audit-r2.json)。
