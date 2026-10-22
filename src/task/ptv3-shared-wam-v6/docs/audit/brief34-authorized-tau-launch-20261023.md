| 结果解读表：读数 / 可能结果 | 含义 | 决定 |
|---|---|---|
| 固定 τ 面板前 8 帧 vs 旧 B1（9.357826 mm），母序列聚类 paired CI | 短期预测变化；B1 是旧配方参考，多项条件来源同时变化，不能归因为事件消融 | 改善且 contact / overall 无明确退化才进入执行器验证；跨 0 则不宣称优势 |
| contact window vs 旧 B1（19.834665 mm），同一固定面板 / 聚类 CI | 接触附近预测是否改善，不能单独代表 approach 或任务成功 | 与前 8 帧、整体联合判断，不以 contact 单项放行 |
| 整体 vs 旧 B1（23.409973 mm），同一固定面板 / 聚类 CI | 完整预测时域的可比读数；旧 B1 仅作旧配方参考 | 不新增基线；保留分阶段结果，尤其 approach（B1 29.897742 mm） |
| 执行器内实际任务进展 / 完成率改善、持平或退化 | 模型误差是否转化为任务进展，τ 读数本身不保证成功 | 改善才支持下一阶段；持平或退化不宣称部署收益；本轮不新开执行器实验 |
| 训练发生 OOM / 非有限 loss 或 gradient / 磁盘告警 | 运行或基础设施失败 | 停止并报告，保留现场；不静默重启、不换参数、不消耗第二次训练 |
| 本次 GPU 授权池 4/5/6 实测空闲，spare 显式豁免；其他门仍逐项检查 | 资源分区按 brief34 执行，不豁免输入合同或磁盘阈值 | 生产事件 mask / curated 索引接线已补；等待 waypoint 缓存绑定，尚未开训 |

## brief34 范围与授权

本轮承接 brief33，只修改冻结政策的 resource_policy；其余字段与 ca640870 版本逐项相等，由 regression test 核对。授权项目池为 4/5/6/7，本次训练池固定 4/5/6，7 属执行器优先级②，0–3 属 zyc。spare 仅在授权池内有富余时要求，本次三卡 DDP 在 4/5/6 上要求 spare=0。禁止触碰 GPU 0–3 / 7 及 zyc PID 2730517/2730518/2730519/2730520；20 GiB stop、30 GiB warn、32 GiB artifact alarm 保持原值。

## 生产接线与输入边界

本轮修改范围已扩到整个 v6 Task，所以把共同 unknown packet / mask 从审计原型接入 events、V6Windows、SharedWAMV6.encode 以及 OnlineV6Runtime.online/predict/_trajectory。两条运行模式（完整 forward 与 forecast-only）都按 checkpoint 的 event_input_policy=既定政策强制 unknown，历史 checkpoint 未声明政策时保持历史行为。模型结构、state_dict 键、goal sanitizer、监督和执行器接口不变。

新增 curated_training 绑定 r7 有序 6,000 元组与既定 CuratedTauSampler；不改 curated_sampling.py，不改组成 / 比例 / seed / cap。TRAIN reader 仍必须绑定 scheduler waypoint 条件后才能返回这些样本。当前缺失时在读取未来监督前抛错，训练入口在 CUDA / optimizer 创建前检查，避免旧 realized endpoint 进入本次训练。

这不是解除后的 GPU 门再设阻塞：冻结政策的 goal_missing_policy 仍显式要求 scheduler-conditioned identities，且禁止 endpoint substitution / goal ablation。本轮已异步询问机主本次应绑定的合规 cache / receipt 路径；未收到前不能更改 goal 合同。既有 scheduler 0/6,000 结论只读继承，不重跑审计。

## 验证与 preflight

新增与相关回归测试：**41 passed, 1 warning in 11.08s**。包括：政策只改 resource_policy、授权池显式 spare 豁免及硬盘门负控制、完整 / forecast-only 实际 predict 模型入口、事件 encoder 非零 bias 下仍全零 residual、curated 索引和 sampler、缺失 scheduler binding 在标签读取前拒绝。没有运行真实 τ 训练或 GPU 作业；工程测试使用 CPU 小模型。

生产代码冻结 HEAD：`eaa8832810a20ea4b2c58b73e3faefcccf032ade`。完整审计证据在 [preflight.json](brief34-authorized-tau-20261023-r1/preflight.json)，有序清单在 [window-tuples.json](brief34-authorized-tau-20261023-r1/window-tuples.json)。

| 启动门 | 实测结果 | 证据 / 解释 |
|---|---|---|
| GPU 授权分配 | **PASS** | 4/5/6 各使用 2 MiB、0%、无 compute PID；每卡空闲 24,574 MiB；available_authorized_gpus=[4,5,6]，world_size=3，required spare=0 |
| 硬盘 | **PASS** | `/home2` 可用 **147,598,905,344 B（137.46 GiB）**；按冻结预算预测写盘 **24,476,537,760 B（22.80 GiB）**；扣除后仍高于 20 GiB |
| 生产事件 policy | **PASS** | 九来源实际 reader / collate 与 runtime policy 包逐位相等；[9,5] event features 非零数=0，valid=0、known-NONE=0；新 reader 读取未来 event sidecar 数=0 |
| curated reader 索引 | **PASS（索引 / sampler 层）** | 有序 6,000 元组无重复，原 sampler 一批 96 次 draw；监督 / goal 与原数据逐位不变。不是声称已经有可返回训练样本的完整 goal provider |
| scheduler waypoint cache | **FAIL** | 冻结 r9 receipt 仍为 0/6,000；新 reader 明确拒绝 `scheduler waypoint cache binding absent ... refusing realized-endpoint fallback before training`；只有此门未过 |

资源测量时间是 **2026-10-23 05:15:05 +08:00**，不是训练启动时刻。0–3 的计算 PID 仍为 zyc 的 2730517/2730518/2730519/2730520，显存 15,621 / 15,697 / 15,569 / 15,539 MiB；7 有执行器进程、占用 4,884 MiB。没有向这些进程发任何信号。receipt 中 legacy `available_unreserved_gpus=[]` 仅保留旧字段，**不参与本次授权门判断**；实际使用的是显式 owner allocation，GPU 门已 PASS，不再索要额外空闲两卡。

| 冻结身份 | SHA256 |
|---|---|
| 更新后的政策（仅 resource_policy 改动） | `7bfcd6e569bb205b88439a9d2bf056eff8421198a03fe614c69d1538d6308044` |
| curated manifest（未改） | `5bda6a99678b0586b496eff7b4b3ba3afddda996cdc16c43603341ed1264225f` |
| 有序 6,000 元组（未改） | `56b9fac09e35e41c3922041e8794232fdeb4056b1e23838928afcd31fc49aaa3` |

当前裁决：**BLOCKED_PREFLIGHT，真实 τ 训练 0 updates**。训练 PID、启动时刻、吞吐、前 200–300 updates loss / 各监督项 / LR、剩余 ETA、checkpoint 增长率均 **未产生**。历史 checkpoint 实测大小 1,217,116,001 B；写盘预测沿用其 1.25 倍 × 最多 16 次保存 +128 MiB 的保守预算，不冒充新 run 实测吞吐。

最小剩余输入是本次应绑定的、覆盖上述元组的合法 scheduler waypoint 条件缓存及其 receipt。已向机主询问路径；**尚未收到**。`configs/tau_event_free_v1.json` 的 `goal_missing_policy` 原文仍是 `preserve frozen scheduler-conditioned identity requirement; no endpoint substitution or unapproved goal ablation`。本轮没有权限把 0/6,000 改写为通过、把 endpoint 改名为 waypoint，或默默启用 unknown-goal 消融。部署 goal sanitizer 的已通过结论保持不变；这里缺的是训练条件输入绑定。

## 启动与监控约束

唯一一次训练若能启动，必须先冻结启动 HEAD、recipe / data SHA256、4/5/6 GPU UUID、torchrun 与子进程 PID、时刻、输出预算与初始 ETA，再记录最初 200–300 updates 的 loss / 各监督项 / LR / 吞吐与剩余 ETA。不得为了交付读数先运行 endpoint 条件训练。

磁盘 watchdog 应常驻于该次训练的 supervisor：<30 GiB warn，<20 GiB stop；产物 >32 GiB alarm，按冻结配方停止；OOM / 非有限数值停止并保留现场，不自动第二次启动。只能向本次登记的训练进程组发信号，不得按 GPU 全局 PID 清理，0–3 与 7 均不动作。本轮若没有启动训练，则不伪造 resident watchdog / PID / 200-update 读数。

## 复现与写盘

```bash
# 复现时将 pytest 临时目录限定在本 Task 内。
PYTHONDONTWRITEBYTECODE=1 /home2/wyy/miniconda3/envs/dexplore_repro_py38_torch222_cu121/bin/python -m pytest -q -p no:cacheprovider \
  --basetemp=src/task/ptv3-shared-wam-v6/tmp/brief34-tests-reproduce \
  src/task/ptv3-shared-wam-v6/tests/test_event_free_preflight.py \
  src/task/ptv3-shared-wam-v6/tests/test_brief34_authorized_training.py \
  src/task/ptv3-shared-wam-v6/tests/test_v6_training.py \
  src/task/ptv3-shared-wam-v6/tests/test_v6_executor_wiring.py
# 41 passed, 1 warning in 11.08s

# 必须使用不存在的新输出目录；这个工具不会启动训练。
PYTHONDONTWRITEBYTECODE=1 /home2/wyy/miniconda3/envs/dexplore_repro_py38_torch222_cu121/bin/python \
  src/task/ptv3-shared-wam-v6/tools/audit/freeze_event_free_tau.py \
  --output src/task/ptv3-shared-wam-v6/docs/audit/brief34-authorized-tau-20261023-r1-reproduce
```

审计写盘前上限 2 MiB，receipt + 元组清单实际 **131,356 B**；新增训练 checkpoint / cache / 视频 **0 B**。本轮交付 12 个文件逻辑大小 **242,245 B**，磁盘分配 **270,336 B**；不包含 Git object、工程测试临时 fixture 或他线写入。
