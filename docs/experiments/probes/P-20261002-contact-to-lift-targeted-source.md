# HF18 slot1 内的定向训练来源补充

原源资格检查：fit1151 early窗口仅23 positive，cal386仅11positive，held971
有41positive/21ep/8组；原cal>=12门未过。不降低门，不先拟合。补充目的为
使训练/校准具有真正形成抬升的监督，而非泛化地再采随机静态桌面状态。

采集仍使用原六专家与两个整片段rotation-anchor程序，完整10步观察反馈、
anchor在窗口开始固定，和原HF15实际反馈律合同一致。pre-state限定：连续3
归一化净力存在代理、中心rest>=5mm、object vz>=.01m/s、尚未同时满足
中心rest+3cm与meshCLR2mm；warmup10、8window/firstepisode/6cooldown。
这些是当前观察条件，不按未来是否成功选择或删除轨迹。

只允许原SHA9851的fit/cal初始组(<70)，所有held组禁止进入补充采集；原
held测试记录保持不动。科学4seed551–554，私有分配14551–14554；工程550/
14550。96env650tick，uniform9分配槽（4/8重复base），actual base p2/9，
其余1/9；先观察再私有抽签。完整窗口任何reset/partial使phase失败。每phase
至少一完整窗口只属执行最低检查，最终监督支持按原HF18全局资格门判定。

采集不拟合NN，不运行policy学习。原生PD/完整源geometry/实际force/专家
反馈逐步保存，源hash固定；独立审计通过才用。原源及新源仅fit/cal合并，
holdout仍原source；整个初始组固定分离，不以held挑候选或阈值。
原支持仍不足就停止，不续seed；足够才九个GPU匹配H10预测模型。
成本并入HF18 slot1 60min/8GiB，包括初始资格检查、工程及失败、4科学phase、
audit、最终qualify、所有拟合与标签归一化审计。不为补充数据开新family预算。
单GPU0/1每phase重新admit，native240s/parent290s。C3OPEN，无正式安全/泛化
或稳定抓取结论。净力是presence代理，source mesh/plane是VHACD近似。
