# PTv3-Shared-WAM v2 当前状态

2026-10-14：完成用户已批准的独立v2预训练实现，代码checkpoint95c1436a。
模型98,591,142参数，joint geometry11,092,000、shared85,054,464。
输入固定物理尺度；TRAIN4096窗共享逐horizon统计；hand进入PTv3并在之后显式concat。
61项新旧合同测试、真实GPU梯度/数值、四步训练、step2→4恢复与双卡global256负载通过。
工程恢复RNG一致，预测hand/scene最大差0.01515/0.00117mm；不声称逐位确定。

已启动 `pretrain-main3-probe-20261014-r1`，GPU4/6、每卡128/global256、1000更新，
最多2小时/8GB；独立监督器PID3444957。旧Task660微调仍在GPU5/7继续。
当前尚无v2学习质量、机器人抓取或Cm增益结论；下一决策依据固定三域dev的物理EPE。
详见[实验卡](experiments/probes/P-20261014-ptv3-shared-v2-pretrain.md)。
