# HD03：先改后果表示，再改控制器

诊断运行COMPLETED，原始HF20仍UNCLEAR；本页不形成新的策略收益结论。
原[合同](P-20261002-generated-contact-consequence-diagnostic.md)与冻结模型未改。
GPU1在8.425秒完成cal306窗口、8程序、四模式各3成员的完整网络预测。
与实际规划日志复现的高度最大差9.16e-5mm、几何损失概率差5.67e-7，属于
批次形状不同的浮点误差。参数指纹前后一致、所有输入hash未变，进程exit0。

Cm相对cup预测平均支持高度增加2.216mm、手存在概率增加3.562pp；190个初始
PD不同cup的窗口为+3.569mm/+5.736pp。306窗口中，没有生成程序的预测joint
概率上界低于cup下界；不支持“模型已明确预警接触损失，只被控制器忽略”的
简单解释。这是模型内部预测，不能代替未执行cup/生成程序的真实后果。

后果输出存在明确事件矛盾：生成程序与cup均有55/306窗口的支持抬升概率
超过手/物存在概率上限，22窗口超过5个百分点，最大超出30.55个百分点。
实际支持抬升事件必然要求手/物末3步存在，原独立输出头没有编码这一关系。
这证明预测内部不一致，不直接证明实际接触识别或所有预测无效。

实际随机分配标签上，Cm支持高度MAE23.940mm，state-only24.863、shuffled
25.214、direct-score25.245mm；手存在Brier分别.13191/.13674/.13555/.18516。
这些cal统计说明仍有部分动作信息，但预测候选增益2.216mm远小于绝对后果
误差；绝对MAE不是动作差估计的误差界，不能由此计算逐状态regret。
仅25个实际Cm窗口上高度MAE18.609mm，对state-only16.336mm；样本有限且
动作由生成器选择，不能以此对整个分布下正式反证Cm。
direct-score只有高度头训练，其他事件是当前状态先验，不能称物理学习对照。

[独立审计](P-20261002-generated-contact-diagnostic-audit-r1.json)从原始hand/object
force与mg重建事实标签，用NumPy独立复算Brier、MAE、事件包含矛盾和Fréchet
计数：标签最大差1.48e-5mm、统计最大差4.08e-6，全部通过。CPU只用于这次
标签和算术审计，没有模型计算。GPU诊断与审计共12.04秒，保存5.93MiB；
借用HF20源与HF19预训练成本另列，未生成新rollout或训练。

原始预测和完整结果在任务输出
`src/task/CmResidual/research/contact_consequence/output/P-20261002-generated-contact-consequence-diagnostic-r1/`，
摘要[JSON](P-20261002-generated-contact-diagnostic-results-r1.json)与[审计](P-20261002-generated-contact-diagnostic-audit-r1.json)已归档。

下一步采用[结构化后果决定](../../decisions/D-20261002-structured-contact-consequences.md)：
先让任务事件预测满足物理包含关系，并学习强cup参考附近的实际动作效果；
预测改进和真实收益仍须分别通过门槛。当前冻结生成配方不继续加门限扫描。
