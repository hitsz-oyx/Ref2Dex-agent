# HF18 slot1：反馈控制律条件的 H10 接触至抬升物理信息

Decision：在尚未满足中心rest+3cm及meshCLR2mm的状态上，真实执行器目标和
已知短反馈控制律能否提供可用的10步任务物理信息？正向才进入对应直接控制。
先用现有随机实际执行数据查资格，避免先训练再发现缺乏抬升监督。

唯一训练源HF15原3917完整随机程序窗口，不加入HF15/HF16/HF17闭环utility。
整初始组SHA9851 fit<50/cal<70/held>=70不变。early定义为pre未同时满足
中心rest+3cm与CLR2mm；原source所有触发均为连续3步归一化净力存在代理。
资格：early held>=300窗口/24ep/8组；新形成抬升标签的正和负各>=24窗口/
8ep/4组，early cal正负各>=12窗口。新形成抬升标签：末3帧均joint净力存在、
CLR>=2mm且中心rest+3cm；净力是presence代理，不声称已识别手物接触对。
支持不足不拟合NN，slot1保持UNCLEAR，下一步补定向实际监督而非扫描模型。

所有条件共享pre-history10x69、pre-nativeobs、pre原始手/objectforce及实际mg
归一化、preCLR/height；仅Cm获得first-step真实独立PD目标减当前q和反馈law。
law为六专家one-hot+固定rotation anchor bit；6/7分别base/cup+anchor。
state-only二者均置零；action-shuffled将motor+law一起在fit的preclear/nonclear
层内置换，cal/held使用真实law。训练from-scratch，相同seed11601–11603、
1000Adamupdates/.0003/wd.0001/batch256/gradclip1；批次seed+30000/shuffle+40000。
所有normalizer仅fit，std下限.001/clip8，校准和留出均不更新模型。

监督输出：10帧height delta和CLR delta（10mm单位）、10帧joint presence；
末3帧joint+CLR支持、clear之后失去CLR事件、新形成3cm支持抬升，以及末3帧
支持正高度最小值减pre正高度（10mm单位）。事件loss从pre或过去帧已clear
后再次CLR<2mm定义，初始仍在桌面不直接视为掉落。连续轨迹用当前vz线性
skip，binary用当前观测状态logit skip；joint支持高度单独监督，不相乘边际
均值。future只作目标，不输入模型。三ensemble平均评估，所有指标原样保存。

信息门：early held height轨迹RMSE和CLR轨迹MAE各比state-only与shuffled
好>=5%；new-lift Brier不差于两个控制与early永不形成抬升的persist基线；
联合支持高度MAE不差于两个控制。支持充分且全门通过PROMISING；充分支持
失败UNPROMISING；资格不足UNCLEAR。不改门限/epoch/seed/标签追门。
这不是排序regret、闭环收益、最终稳定率或完整物理状态预测证明。

GPU0/1新鲜admission，默认一张，slot1总60min/8GiB含资格/所有拟合失败/
时序标签归一化与冻结来源审计；独立审计通过才接纳结果。旧失败和一阶物理
信息保留，C3OPEN。最终稳定成功率仍在真实局部控制/策略学习之后。
