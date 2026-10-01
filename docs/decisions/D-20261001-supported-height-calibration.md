# 概率可靠性与已取得支持高度的保留

2026-10-01，HF10 slot2/2，Decision。原slot1接触MAE门失败，保持失败记录，
不额外训练网络追.20。为后续真实控制试验固定更符合物理作用的新setup。

证据：高norm窗口仅cal9.25%，低norm接触MAE仍.232，不足以优先归因于尺度。
joint-contact Brier.157优于constant.223，但ECE.096且预测.99的窗口实际仅
.908保持接触，需要概率可靠性适配。原指标MAE不是概率校准：若真实率为q，
E|p−Y|=q+p(1−2q)，偏好硬0/1而非真实概率q；Brier的额外误差是(p−q)^2。
此处不把改指标称为原门通过，也不凭它形成utility结论。

原正增益score把已有抬升随后失去支持计为0收益，忽略损失。新得分使用
“末3步接触支持的离桌高度−决策时已经取得的离桌高度”，允许负值。
它由模型已预测的高度/联合接触和当前观察高度计算，无任意drop惩罚权重；
release head仍作为相对base不增风险的guard，联系物理保留而不是瞬态均值。

行动：冻结现有9个物理网络，仅为joint-contact/release拟合正比例logit缩放+
偏置，GPU100步LBFGS/2参数/head、弱正则1e−4；按cal初始帧分3fold（seed9841）
交叉拟合可靠性审查，再用全cal固定最终参数/margin。保留fit/cal边界和全部
原artifact。proper setup门：crossfit joint Brier<=.8×constant，ECE<=.05，
release Brier<=constant，class支持和非零提议。不过则不运行原定控制采集，
HF10停止本地调整；通过才进入固定新seed351–356的五推荐器随机Probe。

新真实数据预固定：各96env/首episode最多8窗口，2+8/cooldown6/650ticks；
五推荐器Cm/state-only/shuffled/base/fit-onlybestfixed等概率分配，重复动作
合并action propensity，所有已开始窗口完整保留。各NN与概率参数在采集前
冻结。新增数据只作前瞻评价，不做在线拟合或阈值搜索。

控制screen：Cm signed-supported-height比每个控制>=.5mm，vsbase的frame90%
下界>0；全状态新release差的frame90%上界<=.02，处理前已抬升分层<=.05；
末3步联合接触差>=−.02；介入5–80%，每项不同动作对比至少32实际匹配窗口、
15episode/侧。报告全部状态、实际动作变化和所有区间。共同动作的差为零，
不能用它们冒充不同动作的支持量。此为混合历史下单决策物理机制Probe，
不替代完整pure-policy或训练所得策略。任一缺支持标UNCLEAR，失败不调参追门。

slot2含setup和采集总≤60分钟/8GiB，单GPU4；预计10分钟。原slot1不重开，
HF09预算3/3不重置，MISSION/权限无变化；无新的外部授权需求。
