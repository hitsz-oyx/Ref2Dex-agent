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

质量观察：step140出现由少量大位移标签解释的孤立scene loss峰值。CPU回放已定位
UniDex未来24帧最大7.387m；后续loss恢复通常范围。暂未修改冻结数据或loss，
不把大位移直接视为坏标签，待固定DEV读出决定是否追加质量Probe。

## 当前：本地已停止，转入质量审阅和远端准备

用户要求停止本地v3。GPU1/3已释放，最后记录step440；未到1000步首次固定DEV，
模型有效性仍UNCLEAR。已有完整恢复点保留至启动step4；本地日志和停止回执保留。
运行同期资源guard因其他GPU任务出现而终止own进程组，原始FAILED回执不改写；
这不构成方法负结果。本地不再恢复；远端数据传输与CPU准入检查继续，后续训练使用四卡。

已加入九源viser完整序列审阅脚本，基于旧多域viewer的源数据读取/回放方式，
新增六源canonical/compact适配、固定检查帧坐标、速度/加速度、人工质量标记。
默认异常定位source4 row14909 tick75。精确raw独立复核支持原仿真高速飞离，
尚不能定位具体动力学/穿透校正原因。
