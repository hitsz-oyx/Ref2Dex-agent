# PTV3 shared WAM v3

九源统一预训练：OakInk2、GRAB、ARCTIC、UniDexGrasp++（仿真）、TACO、HOT3D、DexYCB、H2O、HOI4D。
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

预算：本地启动检查2卡、后续远端4卡；每卡batch128、80000更新、48小时、8GB/run，
启动时核对CAMPAIGN和已授权资源，入口保持单机合计6卡保护，
可用磁盘至少20GiB。前期执行全模型梯度、大批量、保存/恢复检查；异常只停止自己的进程组。
这是大规模离线 Probe，不据此宣称机器人收益。

完整九源viser审阅入口为 `tools/audit/review_nine_sources.py`，默认8769端口，
可按来源/划分/序列选择、播放完整轨迹并保存独立质量标记。
详见 [浏览说明](docs/nine-source-viewer.md)。默认直接打开7.387m的UniDexGrasp++仿真异常样本。
本地v3已按用户要求停止；远端四卡全量训练已运行，当前质量切换及进度见 [STATE](docs/STATE.md)。


当前严格准入输入为 `nine-source-strict-quality-20261015-r1`；九源全序列运动审计后，按用户要求隔离高幅度/模棱两可的窗口。使用独立全局窗口排除索引，不改原始数据；训练采样和DEV面板都实际应用。远端 `--recipe`：

```text
/mnt/ugreen_nas/storage/oyx_storage/ptv3-shared-wam-v3-nine-source-20261015-r1/data-strict-r1/recipe
```

该recipe用于继续既有训练，保留父模型的24-horizon数值normalizer和固定来源概率；它们是此前TRAIN拟合的参考，并非严格子集重新拟合值。跨recipe恢复必须显式使用 `--resume-reviewed-subset`，检查配置、payload、排除集合和normalizer不变，并保留optimizer/各rank RNG。DEV窗口已变更，切换前后曲线不能视为匹配对照。规则、用户判定和切换记录见 [运动复核](docs/motion-followup-20261015.md)。旧recipe、原始payload和checkpoint均保留。
