# PTV3 shared WAM v3

九源统一预训练：OakInk2、GRAB、ARCTIC、UniDex、TACO、HOT、DexYCB、H2O、HOI4D。
此 Task 接管训练、九源采样和归一化入口；清洗实现与冻结数据归 3d-data-preparation。
旧六源工程脚本保留，避免破坏已记录的运行链接；v2 的活动训练源码不修改。

模型从 v2 固定架构派生，保持 98,591,142 参数，改为明确的九源 normalizer/schema。
从头初始化；不使用旧 parent 的权重。统计只来自 TRAIN，九源共享 hand/scene 各 24×3
mean/std，std 下限 1mm。source 概率正比于窗口数开平方，DexYCB 上限1%；main3 内
保持已有 .6/.2/.2 运动分层，native 内均匀冻结窗口。loss 为全局 masked-point mean，
hand/scene 1:1，scene motion tau2mm、floor0.1，无额外 source loss 乘数。

入口：`tools/run/prepare_nine_recipe.py`、`tools/run/launch_pretrain.py`、
`tools/run/train_pretrain.py`。通过独立 detached 工作树冻结代码后运行，输出到
`outputs/ptv3-shared-wam-v3/<run_id>/`。模型/optimizer/各rank RNG一起恢复，恢复创建新目录。

预算：最多2卡且项目合计不超过6卡；每卡batch128、80000更新、48小时、8GB/run，
可用磁盘至少20GiB。前期执行全模型梯度、大批量、保存/恢复检查；异常只停止自己的进程组。
这是大规模离线 Probe，不据此宣称机器人收益。

完整九源viser审阅入口为 `tools/audit/review_nine_sources.py`，默认8769端口，
可按来源/划分/序列选择、播放完整轨迹并保存独立质量标记。
详见 [浏览说明](docs/nine-source-viewer.md)。默认直接打开7.387m的UniDexGrasp++仿真异常样本。
本地v3已按用户要求停止；后续转远端四卡，先完成数据迁移和质量审阅。
