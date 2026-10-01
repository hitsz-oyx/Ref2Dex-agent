# 冻结目录 Cm 的实际闭环结果

冻结协议：[实验卡](P-20261002-catalog-closed-loop-utility.md)。
运行 COMPLETED，原收益门 **UNPROMISING**，HF15预算2/2关闭；局部正向信号保留。

GPU完成12科学phase、821个完整H10窗口/161episode/25initialgroup；全部为
原NN训练/校准未使用的初始组，方法选择仍属于探索，不是未见正式Validation。
五控制器p=.2真实随机分配，所有支持门通过。Cm实际170窗口，其中91相对
base改变独立PD目标，160不同于固定7；662/1700实际Cm控制帧非base。
同一实际状态上的Cm/state-policy推荐一致78.319%，不是实际反事实收益。

| Cm减对照 | 保留支持高度增益mm | 初始组90%区间mm | 几何间隙丢失差pp |
| --- | ---: | --- | ---: |
| 状态局部结果评分器 | −4.445 | [−13.153,+2.003] | 0.000 |
| 打乱程序标签物理模型 | +23.538 | [+9.641,+35.409] | −7.917 |
| base | +24.125 | [+13.704,+31.583] | −3.654 |
| fit选定固定rotation_cup7 | −27.651 | [−34.701,−17.101] | +1.827 |

base/shuffled两对照均通过原收益/episode区间/风险门；state-policy和固定7未通过。
固定7每2步重取rotation锚点，是冻结协议中的固定控制器，不能等同原整10步锚点。
其真实observed局部score41.814mm，Cm13.385mm；不能据此称最终稳定抓取率。
净力代理不能识别手物contact pairs，几何CLR是source mesh对tableplane近似。

后验**决策诊断**只比较每次换程序前真实执行的2步prefix，避免把重新规划后
的H10实际结果当成单一H10程序预测真值。4105prefix：Cm高度RMSE8.733mm，
head-shuffled9.440、常速度10.748；fit-domain3660prefix中为6.716/7.614/9.366。
因此不是完全没有物理信息。短期joint存在Brier .02040却差于直接延续当前
净力存在代理 .01157（域内.01457 vs .01011）；仍有可改的物理预测问题。
这一后验诊断不改变原门、不证明ranking或修复长预测与短执行合同。

所有12,315次model决策、原生PD/专家/参考命令/rawforce/geometry/outcome/propensity
回放通过，自己的native进程全部结束，GPU0/1释放。模型、校准、六专家均冻结。
原slot2全部拟合、失败工程、正式工程、采集、回放/统计/诊断合计1511.60秒/
97.54MB；没有追加seed/epochs或降低原calmargin，C3仍OPEN。

[数值结果](P-20261002-catalog-closed-loop-utility-results.json)、
[完整回放](P-20261002-catalog-closed-loop-utility-record-audit.json)、
[短期物理诊断](P-20261002-catalog-prefix-physics-diagnosis.json)、
[终态审计](P-20261002-catalog-closed-loop-completion-audit.json)。
