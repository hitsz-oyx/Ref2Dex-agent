# 接触约束原生执行工程：完成

run_status: COMPLETED
engineering_passed: true

固定649/160tick首次原生采集超时，parent343.856秒，失败完整保留，没有科学
结果。计算等价修复后同条件r2通过370检测/7窗口，native176.259秒、完整audit
130.283秒，parent317.067秒；随机分配没有Cm，因此尚不算Cm分支通过。

独立且明确的同649/60tick首窗口Cm smoke r3完成95检测/3窗口，实际1Cm/
1direct/1shuffled；强制首窗口真实概率1，其余原随机流。Cm实际10步的PD均
不同于同观测下unguarded，全部指令/PD/rawforce/mesh/pre-post几何及全检测
32步优化/完整NN/nuisance/RNG/参数冻结审计通过，native和audit均exit0。
原native/规划/PD关键差0；risk完整NN浮点误差在固定2e-5门内。

r3 native82.171秒、audit44.594秒，parent137.388秒。包含首轮失败、r2、r3、
性能检查保守10秒、准备60秒及旧固定状态工程16.456秒，累计预算884.767秒，
工程部分在900秒内。共享源和模型训练另报。全部forced数据排除科学分析。

下一步固定科学651–658，5臂完全随机，旧模型/门保持；工程通过不能说明实际
风险改善、抓取收益或策略学习。C3仍OPEN。

证据：[固定科学卡](P-20261002-contact-risk-interventions.md)、
[计算等价](P-20261002-contact-guard-cost-equivalence-r1.json)、
[r2覆盖](P-20261002-contact-risk-interventions-native-engineering-r2.json)、
[r3执行分支](P-20261002-contact-risk-interventions-native-engineering-r3.json)。
