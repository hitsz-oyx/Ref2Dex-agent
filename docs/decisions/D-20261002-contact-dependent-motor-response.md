# Decision Memo：Cm只学习接触状态条件下的动作增量

状态：未执行的设计草案。2026-10-02 用户准备转入新路线，本路线在当前状态
封存；以下网络设计不是已实现、已训练或已验证的方法，不自动授权恢复实验。

HD04线性响应原门UNPROMISING：速度比state改善3.33%，位移/CLR不足；动作
中心化与独立原生PD/物理审计通过。小幅速度信号及非退化聚合协方差不能证明
完整控制收益，却说明最便宜线性配方失败不等于没有动作响应。HF24固定反馈
配方保持关闭，不追加其gain/数据/Cm拟合。强rotation-Cup相对base的已有HF15
机会信号保留；任何新控制仍须实际检验强固定参考，而非复用预测当真值。

选择非线性但物理响应读出显式按动作分解：历史GRU、掌部/五指的object-frame
位置/速度、对应关节及当前收集力编码接触状态；先训练共享state基线并冻结，
再学G(contact_state)*中心native PD。中心来自分配前候选集的真实概率均值；
动作期望物理修正为零，不让Cm通过改变状态基线伪装成动作增益。
基线还观察共享候选平均PD相对当前q，表示当前候选集的平均执行责任；它不
区分具体候选。力来自对应finger intermediate/distal净力，非指尖接触对。

原设计限定第一实际控制周期，随机化中心不能沿用到已经受分配影响的后续
状态。拟议响应输出物体dv/dp/meshCLR残差和hand/object/clear的8类联合event
logit；logit与物理修正均中心化，softmax事件概率的均值不保证等于基线。
next支持高度由实际预测dp与P111计算，contactloss=1-Pjoint，clear loss仅在
当前clear时解释为局部掉落风险。没有长期V、H10任务值头或端点力积分。

原拟复用HF19原fit1899/cal989；实际标签预检发现cal下一步接触丢失16例，
不足预设24例支持门。当前clear325例，其中真正clear→unclear29例；655例
next unclear不能都叫掉落。没有启动非线性训练，也没有降低支持门。预检见
[原始标签计数](../experiments/probes/P-20261002-contact-dependent-motor-information-preflight.json)。

为解决后续状态上的随机化问题，已编写
[逐周期随机采集器草稿](../../scripts/collect_randomized_motor_cycles.py)：每次观察
后重新生成候选并独立分配动作，保存当前历史、全部候选、分配概率及真实
pre/post物理状态。仅Python语法检查通过；尚无独立审计器、受控runner、原生
GPU smoke或新采集数据。不得将其视为可用科学数据源。

原预算草案slot1<=900秒/512MiB、默认单空闲GPU5，3共享state成员各800更新
及两类响应各3x500更新；未登记HF25预算、未消费科学slot。若未来恢复，须先
固定新源的采集预算及支持门，完成独立RNG/PD/rawphysics审计，再决定是否
实现网络。Mission、claim与权限不变，无子代理，C3OPEN。
