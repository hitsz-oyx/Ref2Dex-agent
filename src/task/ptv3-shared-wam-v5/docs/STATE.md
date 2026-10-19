# WAM v5 — 2026-10-19 engineering state

用户授权先准备远端数据、同步实现v5；需要GPU时可短暂停已确认的远端WAM训练并立即恢复。
代码在main，独立Task，不替换v4模型、缓存或训练权重。

## 当前决策：先审计数据与C

按[ref1](user/ref1.md)，先运行[contact定义Probe](experiments/probes/P-20261019-v5-contact-definition-audit.md)：
全量有效cache的分source contact比例，以及至多324个TRAIN帧/物体的法线方向候选比较。
先看结果与可视化，再决定teacher或BCE；保持原17通道C、21点τ/K24逐horizon归一化，
不加入C→PPO、重定向或CEDex抓取数据。法线尚未跨source验证向外，离散MANO/Shadow
采样密度误差也尚未统一，不把候选变化幅度直接当正确性证据。

已增加独立审计工具，13项v5测试通过。新检查确认手缓存39526/39526完成、隔离0；
contact仍生成，完整数据与审计结论以对应运行状态为准。

## 已完成

- 21点τ、778手表面输入、2048物体/接触查询、16部位+C概率头；PTv3候选网格5mm。
- 2747个物体的2048个不同CAD表面点全部完成，保留原32个flow点。
- 九源各一条真实序列已完成21点、表面、contact和reader验证；初始工程快照9334 TRAIN窗口，DEV为空。
- 九个合同测试通过，包括未来标签隔离、无效标签mask、contact分块梯度、原始Shadow FK和运动分层。
- 实际4090/1024宽/bf16，batch128完成8步AdamW，九源均出现：GPU前向+loss约311ms，
  反向746ms，optimizer43ms，合计1.10s/步；峰值allocated18.94GiB。
- 有界缓存+4 workers的8步总耗时18.59s，约2.32s/步，含启动/读取；这是短测，不是全量/DDP稳态吞吐。
- 原WAM训练已恢复，同一supervisor2975536/torchrun2975737，检查时step29480，四卡99%，
  约10.5GiB/卡；两轮暂停共95.70秒，未写v5模型checkpoint。

## 正在运行

远端根目录：
`/mnt/ugreen_nas/storage/oyx_storage/ptv3-shared-wam-v5-20261019-r1/`。

| 阶段 | 路径/状态入口 |
| --- | --- |
| 冻结输入身份 | `data/plan/plan.json`：39526序列、3764199个原准入TRAIN/DEV窗口 |
| 人手原始参数提取/上传 | 本地`outputs/ptv3-shared-wam-v5/hand-packets-20261019-r1/`，剩余OakInk参数解析 |
| 21点与手表面 | `data/hands/progress.json`；机器人直接读取既有远端q/root，通过原始FK重建 |
| 表面接触 | `data/contacts/progress.json`；CPU离线KD树，每帧复用，bbox剔除不改变正接触标签 |
| 自动收尾 | `outputs/finalize-data-r2/progress.json`；完成后写`recipe/manifest.json`和`stats.json` |

进度会继续变化，以这些JSON为准。最后检查时手缓存36704/39526、隔离0；contact3130序列、
约153万个有效物体帧。全量尚未完成。剩余耗时主要取决于原始OakInk pickle解析和contact生成。
各运行有6小时边界和20GiB空闲空间停止条件；完成产物保留。最终收尾区分完整/部分覆盖，
不把缺失标签变负样本，也不自动启动v5长训。

## 适用范围与下一步

contact是手表面顶点距离派生的邻近标签，非仿真真实接触力，也非CEDex法线胶囊公式。
最终recipe保留v4的source概率和main3运动分层，不混入采样策略变化；新21点归一化只用TRAIN。
初始GPU工程快照在分层绑定前使用source内均匀采样，不能据其loss或吞吐声称训练效果提升。

v5尚未长训；分布式启动、逐位续训等价、BC/RL的21点接口与C执行消费尚未验证/接入。
完整数据完成后先检查source覆盖、mask和全量输入/DDP吞吐，再决定训练配置。
