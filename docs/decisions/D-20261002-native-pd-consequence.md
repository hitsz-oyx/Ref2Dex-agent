# 从程序收益头转向真实执行器条件的短物理后果

HF15完整slot2关闭UNPROMISING。实际Cm vsbase+24.125mm，但vsfit固定7−27.651、
vs局部结果评分器−4.445；91/170窗口真实改变base，已排除“候选不执行”的旧问题。
4105真实H2prefix后验诊断：高度RMSE8.73优于常速度10.75，但jointBrier.0204
差于当前存在代理延续.0116。原H10预测与2step重新规划也有合同差异；这些不是
充分证明的单一失败原因。原模型、阈值和所有失败门保留，禁止降低门限追门。

需要决定：可直接观测的执行器目标与原始力，能否改善短物理转移的动作信息？
最便宜动作：利用HF15原始完整程序记录内每个真实step的pre-state、真实PD目标
和当前原始力，学习action-conditioned一步后果；不猜未来专家命令、不把新
闭环H10标签当原整10步预测真值，不用当前held utility数据拟合新模型。

HF16预算2 Probe；slot1<=60min/8GiB、单GPU：9nets（cm/state-only/action-shuffled）
各1000updates、batch256，相同seed/optimizer/minibatches。所有条件获得相同pre
物理状态/力，只有Cm获得实际当前12独立PD目标减pre-q。监督一步height/CLR/
净力存在、接触+间隙支持高度联合结果；采用当前速度的height/CLR线性skip及
当前存在代理logit skip，不把不可靠长期V作为label。中间状态和力严格取前一
真实step，初始组保持旧9851的fit/cal/held划分，未来状态只作目标。

Probe门在首次fit前固定：held已离桌prestate足够>=500transition/24episode/8组；
Cm heightRMSE和CLRMAE均比state-only与shuffled好至少5%，jointBrier不差于
存在代理延续；联合支持高度MAE不差于两学习控制。达标PROMISING，充分支持
下失败UNPROMISING，支持不足UNCLEAR。不增加epochs/seeds或改loss追门。
正向才开展执行合同明确的多步/局部控制设计；失败回更高层物理状态/候选表示。
这不是用一步替代最终10–20step后果或稳定抓取证明，仅是实际动作物理信息
的最小判别。C3OPEN，Mission/claim/资源权限不变，无新增外部授权边界。
