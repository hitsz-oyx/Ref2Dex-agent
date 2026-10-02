# 当前路线封存：随机化动作后果与控制

2026-10-02，用户准备开始新路线；本记录保存当前成果和未完成边界。
分支：`agent/cm-randomized-motor-effects`。封存前最新已完成结果提交：`196389b`。
不启动新的采集或训练，不自动恢复HF25草案。

## 已完成并保留的证据

- HF15保留rotation-Cup相对base的局部候选机会；它不构成Cm增益。
- HF24完成982个实际H10窗口。速度反馈相对base支持高度+22.434mm；相对强
  Cup+6.415mm、区间跨零且低于预设重复噪声门，原机会结论UNPROMISING。
  [完整结果](../experiments/probes/P-20261002-relative-feedback-opportunity-result.md)。
- HD04完成GPU线性动作响应拟合和独立审计。速度误差比state-only低3.33%，
  位移改善不足、mesh间隙更差，原资格门UNPROMISING。动作中心化、原生PD和
  独立物理重建通过；小幅速度信息不能替代控制收益。
  [完整结果](../experiments/probes/P-20261002-randomized-motor-response-result.md)。
- HF25拟议非线性接触响应的源支持预检完成：cal989，接触丢失16<24，真实
  clear→unclear29。没有训练新模型、降低门或将next unclear全称为掉落。
  [预检证据](../experiments/probes/P-20261002-contact-dependent-motor-information-preflight.json)。

## 未执行草案

[接触状态动作响应设计](../decisions/D-20261002-contact-dependent-motor-response.md)
及[HF25草案](../experiments/probes/P-20261002-contact-dependent-motor-information.md)
保留作设计输入，HF25未登记预算、未消费科学slot。

[逐周期随机采集器](../../scripts/collect_randomized_motor_cycles.py)从每个当前
观察独立生成/分配候选，保存所有周期的当前状态、历史、概率和真实后果。
目前仅语法检查通过，尚无原生执行、独立审计器、runner或新数据。不能把草稿
当作可靠数据源或已完成的机制成果。

## 新路线仍须解决

真实候选机会→动作后果和排序（state-only、shuffled、base、best-fixed）→实际
短片段控制并重新观察，仍未完整通过。最终Cm策略学习收益与稳定抓取验收
也未完成；North-star C3保持OPEN。旧失败结果、资源成本和全部审计保留，
新路线不能通过改名或改变判定门把它们升级为正向结论。
