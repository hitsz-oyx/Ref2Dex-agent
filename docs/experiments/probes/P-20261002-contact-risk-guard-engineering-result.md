# 接触风险约束：工程完成，控制作用稀疏

run_status: COMPLETED
scope: engineering only; no scientific Probe classification

固定接触后果模型与独立高度评分模型，先检查排除科学训练的623/33早期状态，
再按预先固定的覆盖修正加入同样排除的590/96状态。模型、32步动作优化、
2pp joint/contact/geometry与5pp support门均不改，没有新增仿真或训练。

第一轮33状态：无最终接触风险违规，Cm与无约束高度控制器有1个PD差异，但
没有接触约束介入。第二轮129状态：无约束控制器有4个接触风险违规，Cm在
全部4个状态改变PD；共5个PD差异。接触约束介入4/129（3.10%），31个动作
不同于Cup，47个当前状态OOD按原规则返回Cup。尚不能说明实际降低风险。

两轮完整网络高度/风险/Cup复算与独立PD映射差均0；state-only风险控制器
等于无约束高度控制器，参数冻结，未来字段污染不影响输入。第二轮GPU5，
进程exit0；第二轮8.412秒，累计工程16.456秒，未突破300秒/64MiB边界。
已保存模型、输入、决策与规划文件hash。旧数据、训练成本另报。

下一步固定独立真实执行Probe，针对接触风险约束介入检验实际接触丢失与
抬升代价。低覆盖需要反映在采集成本和停止条件里；不能把工程通过或
预测风险下降当成真实收益。HF22原门UNPROMISING保留，C3仍OPEN，不启动PPO。

证据：[固定职责](../../decisions/D-20261002-contact-risk-constrained-scoring.md)、
[固定覆盖修正](../../decisions/D-20261002-contact-risk-state-coverage.md)、
[33状态结果](P-20261002-contact-risk-guard-engineering-r1.json)、
[129状态结果](P-20261002-contact-risk-guard-engineering-r2.json)。
