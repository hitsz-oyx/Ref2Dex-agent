# VAL-20261023-wam-full-scale

| 结果解读表 | 当前结果 | 判定 |
|---|---:|---|
| 指标一：20 个新物体上的 τ 预测误差（相对 step011000） | 新权重尚未产生 | 待完整 v6 recipe 与 GPU4/5/6 释放后执行 |
| 指标二：同一 40 初态面板的 `OnlineV6Runtime` 成功率 | 新权重尚未接入 | 待完整 v6 recipe 与 GPU4/5/6 释放后执行 |
| 判据：预测误差下降但成功率不变 | 尚无新旧配对结果 | 不能提前下结论 |
| 判据：预测误差和成功率都上升 | 尚无新旧配对结果 | 不能提前下结论 |

## 事实清单（训练前）

`flowmatching-overnight-20261022-r2` 的 `step011000.pt` 使用的是本地 v6 overlay，而不是
`data/processed_data` 下的 341G 数据池。`training/run-manifest.json` 记录了 strict-FP32、
global batch 288（3 ranks × 96）、seed 170、validation seed 171、flow seed 172 和
9 个 source ID `[0,1,2,4,5,6,8,9,10]`。冻结 parent 的 train/dev 索引分别为 207,579/3,356
windows；训练进度到 step 11,380，但最近保存的 checkpoint 是 step 11,000。按 global batch
计算，step 11,380 暴露 3,277,440 个 train windows，约 15.79 个 train-window epoch。

当前 overlay 的实际覆盖如下：

| source | 序列 | 帧 | 物体 | train windows | dev windows |
|---:|---:|---:|---:|---:|---:|
| 0 | 6 | 2,311 | 8 | 96 | 96 |
| 1 (GRAB) | 1,132 | 343,900 | 45 | 207,179 | 2,944 |
| 2 | 6 | 3,600 | 12 | 96 | 96 |
| 4 | 6 | 600 | 2 | 30 | 30 |
| 5 | 6 | 451 | 8 | 18 | 21 |
| 6 | 6 | 5,772 | 23 | 66 | 86 |
| 8 | 3 | 212 | 6 | 12 | 0 |
| 9 | 6 | 231 | 4 | 4 | 7 |
| 10 | 6 | 3,594 | 4 | 78 | 76 |
| **合计** | **1,177** | **360,671** | **114（source 内去重合并前）** | **207,579** | **3,356** |

冻结 parent plan 其实有 39,526 条候选序列、38,349 条未进入本地 overlay；计划 windows 按 source
为 `0:2,694,722; 1:225,215; 2:351,309; 4:328,731; 5:38,185; 6:51,870; 8:351;
9:6,492; 10:67,324`。因此 r2 不是“已吃满”：source 0/2/4/5/6/8/9/10 都只取了工程序列，
source 1 是唯一接近完整的 GRAB 子集。

## 341G 数据根的独立盘点

这些根均为只读 legacy 格式；r2 的 manifest、路径和 cache binding 均没有直接引用它们。
盘点数字来自各根自己的 manifest/index 或轻量 `shared.npz`/`motion.npz` 元数据：

| 数据根 | 磁盘 | 序列/任务 | 帧 | 物体 | r2 直接使用 | 尚未被 r2 吃入 |
|---|---:|---:|---:|---:|---:|---:|
| `ObjectInteractionCmv2` (v114d GRAB MANO) | 150.29 GiB | 1,335 | 400,616 | 51 类 | 否 | 全部 |
| `stage4` (`data/grab`) | 75.50 GiB | 1,335 | 400,616 | 51 类 | 否 | 全部 |
| `coupled_geometric_cache_v1_20260912` | 43.97 GiB | 285（255 train/30 val） | 72,935 | manifest 未给独立物体总数 | 否 | 全部 |
| `cm_decoder_v2` (`dexplore_rl_v1_3_full10135`) | 34.88 GiB | 348（255/30/63） | 88,498 | 50/25/32（split 内） | 否 | 全部 |
| `dexplore_grab` | 16.55 GiB | 1,335 | 406,264 | 51 类 | 否 | 全部 |

五根合计约 321.2 GiB（`du` 的 GiB 口径）；“尚未被 r2 吃入”按直接路径是全量。它们不能直接
替换 v6 的 `V5Windows` overlay：输入字段、contact/event 标注、object-surface registry、
normalizer 和 source split 合同不同。把这五根接入必须先做一个新的、可验证的 v6 recipe；不能
用旧 cache 的文件数冒充已纳入训练。

## 规模方案与当前阻塞

已留下 [data-audit.json](../../../../../outputs/ptv3-shared-wam-v6/full-scale-20261023-r1/data-audit.json)、
[launch-config.json](../../../../../outputs/ptv3-shared-wam-v6/full-scale-20261023-r1/launch-config.json)、
完整的 [config-b96-w4.json](../../../../../outputs/ptv3-shared-wam-v6/full-scale-20261023-r1/config-b96-w4.json)
（路径以仓库根为准；实际文件名见同目录）和复现命令草案。目标是沿用 step011000 的结构/损失，
4 卡 `[4,5,6,7]`、每卡 batch 96、global batch 384、seed 170、至少 20k、目标 40k 更新，
每 500 步保存并保留最近两个 checkpoint。

本轮尚未启动训练，原因是可核实的外部状态阻塞：GPU4/5/6 正被另一条 `wam-rl-policy-v3`
线程使用（本任务没有触碰该目录或其进程），而完整 v6 recipe 所需的 38,349 个 overlay 序列
位于未挂载的 `/mnt/ugreen_nas` 路径；本机剩余空间也不足以无界复制 341G 数据根。没有这些
条件，启动会变成“旧 1,177 序列子集续训”，不能声称是用户要求的全量重训。

## 复现与证据

```bash
cd /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent
python3 src/task/ptv3-shared-wam-v6/tools/audit/audit_full_scale_data.py \
  --output outputs/ptv3-shared-wam-v6/full-scale-20261023-r1/data-audit.json
python3 -m json.tool outputs/ptv3-shared-wam-v6/full-scale-20261023-r1/config-b96-w4.json >/dev/null
```

Baseline checkpoint and its run manifest remain at
`outputs/ptv3-shared-wam-v6/flowmatching-overnight-20261022-r2/training/step011000.pt` and
`training/run-manifest.json`. The target run directory is intentionally launch-ready but marked
`BLOCKED_BEFORE_LAUNCH`; no fake checkpoint or fabricated metric is reported.
