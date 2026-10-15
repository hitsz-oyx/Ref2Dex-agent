# 当前：远端四卡motion九源全量预训练已稳定更新

用户已授权直接开始。运行`pretrain-nine-motion-remote4-80k-20261015-r1`，远端supervisor1877848，固定代码6e535bc1、短目录`/home/wbcd/workspace/oyx_ws/wam-v3-20261015/repo`。使用下述motion清洗recipe，从头初始化、GPU0/1/2/3、每卡128/global512、80k更新/48h/8GB预算；本地GPU当前空闲。启动检查已通过：step120，四卡98–100%利用率、显存约22–32GiB，loss/grad有限；完整step1 checkpoint已保存。近期1.4–1.47更新/秒（global512），80k暂估16–18h；训练后首次固定DEV在step1000。当前具体步数/失败原因以远端process/progress.json及本地`tmp/nine-remote-20261015-r1/training-startup-monitor.jsonl`为准。

按用户“先筛查清洗”要求，全量扫描UniDex33,056条，版本化规则隔离25条中的136窗（TRAIN114/DEV22）。保留15条的可信完整前缀，10条无剩余窗；不按抓取成功过滤、不改原始payload，其余八源窗集合不变。

新native view：`outputs/3d-data-preparation/native3d-clean-motion-20261015-r1`；新九源recipe：`outputs/ptv3-shared-wam-v3/nine-source-motion-recipe-20261015-r1`，TRAIN4,806,030/DEV821,613。原三源4096/source统计抽样保持，扣除114个TRAIN窗的精确贡献后重新计算配比和24-horizon hand/scene mean/std；DEV不参与拟合。114窗独立reader重算并回加统计误差≤7.28e-12；已知row14909移除59/67/75、保留3..51。

规则针对高冲量、共同手物强冲量、最初两个转换已高速飞行、极端转速的保守疑点隔离；不宣称排除了所有物理异常。纯旋转接触与正常掉落要继续视觉复核。详见[清洗记录](../../3d-data-preparation/docs/research/2026-10-15-unidex-motion-cleaning.md)。原清洗view、raw数据和本地旧训练全部保留。

远端验收已完成：`/mnt/ugreen_nas/storage/oyx_storage/ptv3-shared-wam-v3-nine-source-20261015-r1/data-motion-r1/recipe`。仅传约3.24MB元数据，复用37,591 payload/2,638 CAD；九源36 TRAIN/八源32 DEV真实mixed-reader通过，数值统计与本地完全一致，14909窗口准入和原始坏窗仍可读另检通过。以上为启动前CPU验收；证据`tmp/nine-remote-20261015-r1/motion-r1-evidence/`。

Viser已切换同8769新清洗recipe，PID1686665、固定代码494c2111；10条整段隔离仍可选择回看，真实UI检查通过，HTTP200。新标记目录`nine-source-motion-review-20261015-r1/annotations`；旧标记保留。

以下为先前运行与部署记录；当前状态以上述清洗版本为准。

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

远端四卡准备：已提供显式world4配置`pretrain_nine_remote4_80k.json`（每卡128/global512），
launcher允许1–4卡，保持单机合计6卡保护；只提升自己的soft nofile至hard内最多65536。
训练代码部署短路径以满足loader Unix socket108字节限制。当前没有启动远端GPU训练。

## Viser已部署

`http://127.0.0.1:8769`（转发8769），PID1622883，固定代码cfc45d24，
工作树`tmp/nine-source-viewer-code-20261015-r1`，默认row14909/frame75。
脚本`tools/audit/review_nine_sources.py`。覆盖全量39,528条TRAIN/DEV完整序列，
4,806,144个TRAIN窗/821,635个DEV窗，不是抽样面板。
九项Task测试通过；九源真实Viser控件切换、空DEV、坐标、播放、标记通过；
HTTP200、版本匹配的WebSocket下发九源控件与曲线并接受交互事件通过。
服务CPU运行、最长24小时；标记保存在独立`nine-source-review-20261015-r1/annotations/`。
