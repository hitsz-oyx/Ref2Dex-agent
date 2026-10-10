# Retargeter已有研究与本轮去重

用户询问仓库已有研究后，只读核对旧Task代码、实验卡、fit manifest/result和标定统计。
旧Task实验仍暂停；本文不重新启动它，也不把历史Probe当正式结论。

## 已有，不是本Task首次提出

1. `consequence-evaluator/hand_action_retargeter.py::HandActionRetargeter`已是
   width128/2block/4head Transformer，actual未来24步11点位移+current q/dq→actual
   24步18D native actions。结构化train/val/test为96/64/64episodes，2400updates，
   原fit r1 test raw L1 .016367、horizon0 .012562；不是只做几何IK或没有learned R。
   v2已加当前object frame手几何与previous native A；context-fit-r2 result实值
   test .015712/horizon0 .009652。不同数据/归一化不能与新标准化L1 .304456直接比。
2. object-relative GT-servo研究已做rigid-root腕姿恢复、next-motion/current dq的
   train-only PD inverse。标定仅三个旧launch的first40 approach帧，非全contact/hold。
   r14/r15 wrist+source fingers均held483，fullgeometry冷启动r15却238；几何准不等于
   稳定抓持。新step-inverse使用两份实际PPO全episode数据，只检验不同范围的局部标定。
3. fixed-wrist+learned fingers只有held4/6/8；fixed-wrist+live teacher fingers也仅
   8/10/9，均零clip。没有把失败隔离为纯finger/preload问题，不能用此结论换掉整个R。
4. 旧command-coverage audit有动作变化，但contact/hold近似state匹配数不足，
   不能判定几何与q/dq是否充分识别preload。没有证据证明触觉必要；当前不加触觉。

来源卡：
- [full-action数据/learned Transformer与执行](../../../consequence-evaluator/docs/experiments/probes/P-20261010-hand-action-retarget-data.md)
- [GT-servo/PD标定和冷启动界限](../../../consequence-evaluator/docs/experiments/probes/P-20261009-object-relative-gt-servo.md)
- [fixed-wrist归因对照](../../../consequence-evaluator/docs/experiments/probes/P-20261010-hand-action-fixed-wrist-teacher-finger.md)
- [command覆盖/匹配证据不足](../../../consequence-evaluator/docs/experiments/probes/P-20261010-finger-command-coverage.md)

## 本轮哪些已重复，哪些尚未检验

“真实τ+state监督learned native R”整体已经研究过；此前ref1_1设计表的“当前实现”
只描述最近使用的TauTracker，没有覆盖历史full-action Transformer，不能当全仓库事实。
新generic R的主要新信息是纯current s87/active12/horizon8合同、固定state-only与
同tick τshuffle、独立PD精度审计；不是首次提出/首次训练learned retargeter。

旧两个Transformer的action queries均独立learned query/time，hand输入作为memory。
未在这两段源码看到每action slot直接读τ[k]及τ[k]-τ[k-1]的query分支。
因此local-motion是一个窄的待检验结构变化；它还增加参数，不能单归因于alignment。
已完成generic fit与全局PD标定不作废，但应把它们定位为重复基线/新条件诊断。

## 更新下一决策

先把旧v1 frozen checkpoint在新Task的相同actual PPO validation窗口上重算，核对
完整输入/单位/normalization/首8 active labels。它不需previous command，也不读触觉，
符合当前观测边界。旧v2包含previous command，当前不能无说明加入公平对照。
旧模型在不同数据上训练，因此只作reuse/sanity baseline，不能据此做matched架构claim。
结果决定是否优先沿用/诊断旧R，而非在更少失败数据上重新证明同一架构。
local-motion训练尚未启动，先完成该去重基线；不得直接重启旧Task的失败架构sweep。
