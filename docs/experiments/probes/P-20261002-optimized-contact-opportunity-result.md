# HF20：冻结后果模型生成动作的真实机会结果

source_experiment_id: P-20261002-optimized-contact-opportunity
family: HF20
probe_index_in_family: 1
run_status: COMPLETED
label: UNCLEAR

原[实验合同](P-20261002-optimized-contact-opportunity.md)不改。12个科学seed
591–602均完成原生采集和完整独立审计，共1273个H10窗口/12730实际执行步。
全部候选在随机分配前由当前观测生成，模型和专家冻结；每状态只有实际分配
候选的后果，没有逐状态全候选反事实真值。

fit633、cal306、held334。留出Cm实际匹配23窗口，低于预设24支持门，不能
事后降低门槛或补seed。对fit选定强cup的保留支持高度HT差为−6.372mm，
motion/start组90%区间[−14.214,+.916]，episode90%区间[−21.207,+10.544]。
噪声门为23.406mm；全部比较均未通过原收益/接触/噪声联合门。结果没有证明
生成程序超过强cup，也不能升级为Cm核心思想的负因果结论。

控制路径确实生效：留出23个Cm匹配窗口中16窗口/160步的实际PD指令不同于
同一实际轨迹上重算的cup；初始候选PD改变比例66.77%。这不代表不同cup轨迹
的后果优势。原生pre/post物理、32步规划、完整NN、随机分配与冻结输入均通过。
独立统计审计物理score误差0、统计最大差8.89e-16。

原始joint HT差−19.46pp包含分配权重质量差异：Cm23窗口的HT质量.6886，
cup62窗口.9281；实际条件观测joint率分别15/23与43/62。这些条件率不是随机
总体因果差，不能把原始HT值直接解释成接触率下降19.46个百分点，也不替换
原估计器/科学门。

累计源wall2074.29秒，加原生工程900秒及分析/独立统计审计共2981.63秒。
源GPU占用3652.18秒、完整审计2442.55秒，合计101.58GPU分钟；三GPU并行
缩短wall而不减少计算。共享HF19预训练28.165分钟/498.84MiB另计。源、分析
及独立审计进程均exit0，资源已释放。

证据：[固定结果](P-20261002-optimized-contact-opportunity-results-r1.json)、
[独立审计](P-20261002-optimized-contact-opportunity-audit-r1.json)、
[分配诊断](P-20261002-optimized-contact-allocation-diagnostic-r1.json)。

关闭这个冻结生成配方，不启动只允许正向后开展的slot2适配/重规划或PPO。
当前生成目标只惩罚几何间隙损失，没有单独使用未来手/物保持预测；下一步
[HD03](P-20261002-generated-contact-consequence-diagnostic.md)只重放cal，区分
模型已经提示保持风险但解码器忽略、与优化后动作分布上预测不可靠。核心Cm
作用链条和策略训练目标仍开放。
