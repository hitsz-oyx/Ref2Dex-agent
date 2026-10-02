# HF22：接触损失可学，但适配高度的原门未通过

source_experiment_id: P-20261002-support-preserving-contact-effects
run_id: P-20261002-support-preserving-contact-fit-r2
run_status: COMPLETED
Probe classification: UNPROMISING

固定6263事实窗口，旧fit2532+新fit624各占batch一半，旧fit归一化继承，四模式
三成员各500 GPU更新完成。新cal323/known-held298没有参与拟合；数据已用于
路线决定，仍是复用信息Probe，不是独立Validation。

| new known-held298 指标 | Cm | state-only | shuffled | 未适配旧Cm |
|---|---:|---:|---:|---:|
| 支持高度MAE(mm) | 11.027 | 14.673 | 14.644 | 11.242 |
| 末3步joint存在Brier | .11067 | .12694 | .12099 | .12000 |
| H10任一步接触损失Brier | .10512 | .13063 | .12720 | 无该受训头 |

接触损失相对state/shuffled改善19.5%/17.4%，joint存在改善12.8%/8.5%，全部
物理/支持/几何风险非劣门和旧能力保持门通过。新末段共同存在与H10任一步
损失的必要关系通过；direct仅报告受训高度10.667mm，其事件/物理/接触头不
参与科学比较。

原门仍失败：新known-held高度11.242→11.027只改善1.92%；实际生成分布
180窗口高度12.986→12.685改善2.32%，均不足预先固定5%。不能以接触部分
正向把整轮改为PROMISING，也不追加更新/seed或拿本适配模型直接宣称控制成功。

独立审计全6263原始force/mg与所有标签、世界物理目标/当前PD/旧fit归一化、
全部actual及八候选完整NN、旧冻结Cm基线、原门分类全部通过。标签差0，
物理目标归一化坐标差1.54e-4，完整NN0，统计最大差2.59e-6。

首次启动在队列检查失败后错误进入训练，已确认自己的进程并终止，exit143；
6成员3000更新完成，活动第7成员0–500额外更新不可完全观测，没有完整科学
结果，全部保留。修复命中diagnostic_hypotheses子串的队列编辑并提交93e7244后，
同seed/输入/配置r2从原checkpoint重新开始；无门限变化。早期尝试精确墙钟
未保存，成本单独以300秒预算费用报告，不冒充实测精确耗时。含工程/准备/
该费用/正式fit与审计累计预算511.60秒，旧源与模型另报。

下一步改变控制职责：利用已学到的接触风险约束独立高度评分的候选，先检验
约束是否真正改变动作及避免预测风险。该观察只授权新的机制工程，不替代
当前失败门；真实收益、策略训练和稳定抓取仍需完成。

证据：[固定卡](P-20261002-support-preserving-contact-effects.md)、
[完整结果](P-20261002-support-preserving-contact-fit-results-r2.json)、
[独立审计](P-20261002-support-preserving-contact-fit-audit-r2.json)、
[成本账本](P-20261002-support-preserving-contact-fit-completion-r2.json)。
