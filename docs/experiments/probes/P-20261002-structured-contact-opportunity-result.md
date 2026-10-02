# HF21 独立真实机会结果：控制链打通，当前配方收益未过门

source_experiment_id: P-20261002-structured-contact-opportunity
run_id: P-20261002-structured-contact-opportunity-source-r2
run_status: COMPLETED
Probe classification: UNPROMISING

固定8seed611–618、96env、300tick、每stratum2窗口、16slot随机分配全部完成；
1245H10窗口/563first episodes，fit624/cal323/held298，均满足预定支持门。
8个native及8个完整物理/规划/前置预测审计均exit0。纯统计独立rawforce/标签/
概率/均值/分组与episode bootstrap/全部固定门复算通过，标签差0、统计差1.78e-14。

| 主要 AIPW 比较 | cal 高度差(mm) | held 高度差(mm) | held group90(mm) |
|---|---:|---:|---|
| Cm − Cup/state-only/fit-best-fixed | +4.867 | +3.420 | [-3.325, 9.280] |
| Cm − direct-score | -3.164 | +3.487 | [-2.434, 8.706] |
| Cm − shuffled | +22.377 | +7.126 | [-0.861, 14.326] |
| Cm − always-base | +14.160 | +13.455 | [3.759, 23.003] |

主要估计器与门均按卡预先固定，原HT并列保留，不挑更有利者。重复分配噪声
门cal41.609mm、held9.616mm。高度相对base有探索正向信号，但不能通过完整
门：held joint存在相对base -27.92pp、相对shuffled -16.94pp；cal相对direct
高度/风险/接触也未过门。相对强Cup的joint差cal+1.70pp/held+0.98pp、几何loss
差cal-8.61pp/held-0.58pp，但高度区间跨0、增益未超过噪声门。支持充分，
因此当前配方UNPROMISING，而不是以缺样本解释或追加seed。

作用链确实修复：cal69个实际Cm窗口中55个改变Cup PD，550/690步；held73个
中45个改变，450/730步。提议PD改变比例cal76.47%、held68.79%。cal101个、
held99个episode发生重复规划，后续窗口189/152，最多4次/episode。此处是
随机行为分布中的重复观察局部响应，不能推导纯Cm-on完整闭环策略的因果值。

后验cal诊断用于选择下一路线，不改变本次分类：实际Cm生成臂的高度MAE
17.628mm（69事实匹配），Cup14.271mm、direct20.895mm；旧reused-held整体
9.980mm不代表新生成状态分布。Cm生成臂平均高度预测偏差+1.932mm，joint
预测.8004、实际.8116；不是所有失败都能解释为统一高度过估或接触校准偏差。
这些实际匹配的条件均值不是随机化因果臂比较，也未做新的NN拟合。

原r1启动器KeyError在任何native启动前退出，保留并计成本；同协议修复读取
native_command后r2执行，无新seed/模型/门。含新工程、准备预留和失败、源、
全部审计与统计约2261.93秒；原生49.14GPU分钟、完整审计39.37GPU分钟另报，
旧slot1模型与源成本分开，详见[完成账本](P-20261002-structured-contact-opportunity-completion-r2.json)。

结论：已经有可用预测信息和实质控制覆盖，但强对照增益未建立；不启动当前
配方PPO或最终成功率矩阵，也不否定Cm核心假设。下一步优先利用新生成分布
的实际fit后果适配模型，并明确保护hand/object共同存在，先过信息门再新真实执行。

证据：[固定卡](P-20261002-structured-contact-opportunity.md)、
[完整统计](P-20261002-structured-contact-opportunity-results-r2.json)、
[独立复算](P-20261002-structured-contact-opportunity-statistics-audit-r2.json)。
