# HD04：随机化中心动作有小幅速度信息，线性资格门未过

source_experiment_id: P-20261002-randomized-motor-response
run_id: P-20261002-randomized-motor-response-fit-r1
run_status: COMPLETED
Probe classification: UNPROMISING

复用已审计HF19全部12源的第一真实控制周期；fit1899、cal989/248episode/
26组，dynamic cal960，held857完全不构造特征/目标。相对实际候选集概率均值
中心化native12独立PD，状态基线加当前几何条件线性响应，固定ridge.01一次求解。
velocity/position/CLR残差来自实际post、重力和当前CV先验；不积分端点净力。

| 全cal误差 | 动作响应 | state-only | shuffled |
|---|---:|---:|---:|
| object dv向量RMSE(m/s) | .32550 | .33672 | .36088 |
| object dp向量RMSE(mm) | 6.993 | 7.063 | 7.593 |
| meshCLR RMSE(mm) | 5.018 | 4.903 | 5.230 |

速度比state改善3.33%，动态帧3.45%，未达10%；位移改善0.99%，间隙反而
更差，未过原5%门。比CV速度.37640更好，该非劣门通过。分组描述性90%
速度RMSE差[-.01992,-.000968]保留小幅动作信号，不升级完整资格，也不推出
非线性Cm没有作用。该源、该状态表示上的线性响应不足以承担预定职责。

完整候选预测的概率加权均值等于共享state基线（差9.24e-14），零中心动作
精确回到state。native起始PD差2.37e-7；独立标量rawforce/掌部几何/真实第一post/
fit-only归一、GPU正规方程、state与shuffled系数、全部候选预测和原门复算通过，
最大差1.65e-12。聚合标准化动作协方差谱[.486,1.759]非退化，但不能证明每个
具体状态都有完整动作秩或全候选真值；这里没有逐状态oracle、控制收益或抓取结论。

GPU5拟合exit0，独立GPU审计exit0，自己fit PIDterminal；累计预算45.608秒/
6424651bytes（6.13MiB），含准备30秒/收尾5秒，旧源1415.783秒成本单列。
关闭HD04线性配方1/1，不调ridge、seed或增加拟合次数。下一步判断非线性
接触状态响应及观察信息是否足以支持物理控制，避免再回到固定动作目录或
绝对收益头；仍须真实候选机会、动作后果/排序与全部控制、直接再决策及最终策略验收。

证据：[固定卡](P-20261002-randomized-motor-response.md)、
[完整结果](P-20261002-randomized-motor-response-results-r1.json)、
[独立审计](P-20261002-randomized-motor-response-audit-r1.json)、
[成本与关闭](P-20261002-randomized-motor-response-completion-r1.json)。
