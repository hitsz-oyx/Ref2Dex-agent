# 结构化 Cm 独立局部收益运行

固定代码14e836b，科学卡[P-20261002-structured-contact-opportunity](../experiments/probes/P-20261002-structured-contact-opportunity.md)。
由当前会话直接启动，无子代理。原生工程排除623完成33H10窗口，父/native/audit
exit0，154.2618秒；全候选前置预测独立全NN误差0，16slot概率/分配重放通过。

首个源run_id `P-20261002-structured-contact-opportunity-source-r1` 以
`KeyError('command')` 在所有native子进程启动前FAILED，父进程已exit1；
全部phase无native_pid。原manifest与耗时保留，不是有效科学Probe，不占新slot。
启动器修为读取原生工程实际的native_command，恢复时核实此前无native进程，
严格保留同8seed/模型/候选/指标/支持门并计入失败耗时与固定预算。

当前run_id `P-20261002-structured-contact-opportunity-source-r2`，输出位于
`src/task/CmResidual/research/contact_consequence/output/` 下同名目录。
启动后进程现场核实parent2717138与三个native2717324/2717321/2717323存活，
GPU4/5/6各约18GiB，PhysX/CUDA计算。这里只记录启动事实；最终状态以实际
process handle与该目录run_manifest.json为准，不从此静态记录假定仍存活。

固定8seed611–618、96env、300tick、每stratum2窗口、16均匀slots。
每phase原生600秒/parent640秒，完整审计420秒/parent450秒；三lane并行，
新工程/准备预留400秒、pipeline<=3200秒、slot<=3600秒/8GiB。
真实收益只在完整panel与全审计终态后按既定AIPW门分类，同时报告原HT。
尚未获得本轮科学结论，不表示策略学习或最终稳定抓取完成。
