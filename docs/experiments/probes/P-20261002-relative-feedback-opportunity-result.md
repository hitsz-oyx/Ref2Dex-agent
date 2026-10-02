# HF24：速度反馈保留局部信号，强对照机会门未过

source_experiment_id: P-20261002-relative-feedback-opportunity
run_id: P-20261002-relative-feedback-opportunity-source-r1
run_status: COMPLETED
Probe classification: UNPROMISING

固定6seed671–676全部完成982H10/438episode，fit420/cal222/held340。
fit原规则选velocity2，它也是fit best-fixed；held实际base115/Cup116/velocity46/
position63，支持充分。未从held选候选或调gain，排除工程669不进入科学/训练。

| 留出velocity相对 | 支持高度差 | group90 | episode90 |
|---|---:|---:|---:|
| base | +22.434mm | [+9.529,+33.349] | [+8.728,+35.013] |
| strong rotation-Cup | +6.415mm | [-1.784,+16.164] | [-5.796,+17.584] |

base重复槽零差1.395mm，Cup重复零差10.976mm，固定Cup收益门21.952mm。
相对base确有超过base噪声的局部信号；相对强Cup区间跨0且不足其重复噪声门，
因此完整原门UNPROMISING，不把base正向信号替代强对照通过。这既不否定
速度反馈可能有作用，也不能证明可靠强对照收益、逐状态oracle或Cm控制收益。

相对Cup：contactloss -9.783pp(group90[-20.672,-.705]，episode90[-20.447,+1.592])，
geoloss -3.786pp，末3步joint +10.682pp（两种90下界正）。三项点非劣通过，
高度/零差/双区间门失败。held速度46窗口全部实际PD改变同观测Cup，不是未来
Cup轨迹真值。位置反馈相对Cup高度-3.205mm，接触loss+3.571pp/geoloss+2.805pp。
没有据其结果转选候选。初始clear velocity19/Cup27窗口，不足原20/20支持门，
掉落收益继续UNCLEAR，不称零风险；H10不能代表最终稳定抓取。

全部6native/6独立PD/RNG/rawforce/mesh/pre-post观测审计exit0，完整专家重放0。
独立标量统计重新从原始力、高度、实际propensity、fit选择及cluster bootstrap
复算，差最大5.98e-7，原门与标签完全一致。原HT并列保留，没有切换估计器。
科学源、工程失败/修正/静态、准备、分析与额外审计预算合计640.669秒，
144252822bytes（137.57MiB）；旧专家/共享源成本另报。全部自己源PIDterminal。

决定：HF24当前gain0.5反馈机会配方关闭1/3，不执行未使用的slot2/3，不追加
seed/gain扫描或为本配方训练Cm/PPO。新数据与base局部信号保留为后续设计输入，
不能改写完整机会门。返回Cm控制职责/物理表示的路线复盘，核心目标仍是可靠
候选机会→对应动作后果/排序→实际短片段再决策→策略训练/最终稳定抓取。

证据：[固定协议](P-20261002-relative-feedback-opportunity.md)、
[完整结果](P-20261002-relative-feedback-opportunity-results-r1.json)、
[独立统计复算](P-20261002-relative-feedback-opportunity-statistics-audit-r1.json)、
[运行与成本关闭](P-20261002-relative-feedback-opportunity-completion-r1.json)。
