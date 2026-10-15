# 九源统一预训练状态

用户已要求第一轮合并九源并尝试大规模训练。九源 TRAIN 共4,806,144窗，
统计 `nine-source-recipe-20261015-r1` 已完成；HOcap、H2O TEST 和encoder越界窗排除。

已通过5项合同测试、九源mixed collate审计及真实双卡满batch启动检查。
`startup-nine-ddp-20261015-r1` GPU1/3、每卡128/global256，完整模型98,591,142参数，
4次更新、step1/step4模型与optimizer和rank RNG保存成功。
最终峰值分配12.96GiB，reserved17.62GiB（rank0），loss/grad有限。

长训 `pretrain-nine-80k-20261015-r1` 从step4完整恢复，目标80000步，
固定代码67f7af78，独立工作树 `tmp/nine-source-code-20261015-r2`，supervisor1584518。
GPU1/3，与已有GPU4/5/6/7训练合计6卡。48小时/8GB预算，可用磁盘下限20GiB。
已确认step4→60持续更新，稳态0.918–0.933步/秒，约235–239窗口/秒；
预计80k约24小时。两卡利用率100%，rank0峰值分配13.64GiB，reserved21.15GiB，
物理显存约21.59GiB/15.58GiB，无OOM/非有限量。预算内继续，不形成模型效果或机器人收益结论。

用户追加远端九源数据准备，已并行检查远端已有数据及可迁移闭包；本地长训不等待传输。

详见 [实验卡](experiments/probes/P-20261015-nine-source-pretrain.md)。
