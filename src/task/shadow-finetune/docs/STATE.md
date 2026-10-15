# ShadowHand 原生微调当前状态

2026-10-15：固定计算预算的小数据/全量对照已完成，两臂1000步和相同16env闭环，
五个stage均exit0，耗时25.8分钟。运行固定57737b7b，输出
`outputs/shadow-finetune/shadow-scale-controls-20261015-r2`，当前没有Shadow训练进程。
GPU6已释放，GPU7空闲；原Inspire GPU0–5与远端训练独立继续。

共同DEV64首拍canonical MAE small0.11705/full0.10978，仅改善6.21%；
hand/scene forecast normalized L1改善36.8%/41.4%，不能直接换算成毫米。
原生学生官方sticky成功small7/16、full2/16；末帧2/16、1/16；同步抬升保持60帧3/16、0/16。
DEV学生官方均1/8，teacher官方8/16、保持7/16（DEV仅1/8）。这些不是接触稳定认证。
结果UNCLEAR，未达到首拍>=20%改善和DEV成功+2/8；不关闭数据量假设或归因Inspire。

matching与独立raw复算均通过：51个实际初始actor张量完全相同、共同DEV物理身份/统计一致；
两臂teacher完整200状态轨迹逐值相同，初态与命令时钟误差0，无invalid。
两臂250/500/750/1000八个完整checkpoint实际加载审计通过，optimizer/模型有限、统计一致。
采样重放最终RNG精确匹配：small覆盖6231/6232独立query，full62636/1536897（4.08%），
虽覆盖所有1567物体，每条已见轨迹平均只抽6.34次，尚不能评价full收敛后的能力。

全量TRAIN1536897首拍label已用官方冻结300D actor穷举重放，PASS：
max1.1206e-5、MAE6.0609e-8、1e-4阈值下异常0，20.09秒/peak77.1MiB。
这排除准入首拍的粗大标签/teacher身份问题，不覆盖全部未来命令或保证闭环。
报告`outputs/shadow-finetune/full-teacher-label-audit-20261015-r1/report.json`。

完整forecast支持只保留安全action8候选中351321/706676个近目标query，晚期BC覆盖明显减少。
但当前仍有7868/10118条TRAIN含近目标query，不能说完全没有保持阶段。
分层推理已完成：small自己41个TRAIN起点首拍MAE0.22263、torque0.67439；
full模型在配对64起点MAE0.20798，近目标0.03542、安全尾段0.03131。
原DEV64首拍复算差<=4.03e-7，17.05秒/peak668.5MiB，无参数更新，GPU6已释放。
41个small起点均在原训练中抽到4–16次（共427/64000），有曝光但仍拟合不足。
下一步优先检验起步分层采样/动作拟合，尾段BC后排；不把teacher条件状态的低尾段误差
当作排除闭环分布移位的证据，不直接归因Inspire。详情及局限见实验卡。
报告`outputs/shadow-finetune/phase-fitting-20261015-r4/{report.json,metrics.csv,small-startup-exposure.json}`。

详见[P-20261015-shadow-data-scale](experiments/probes/P-20261015-shadow-data-scale.md)，
比较图/CSV及独立审计在r2的`summary/`目录。
先前pilot（seed213、selected500、学生1/16）不是当前matched小数据臂，不跨种子混报。
