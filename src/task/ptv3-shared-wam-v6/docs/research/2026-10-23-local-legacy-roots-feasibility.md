# Local legacy roots feasibility for τ training

| 本轮结果 | 解读 | 改变哪个决定 |
|---|---|---|
| 非 GRAB 两根（`cm_decoder_v2` / `coupled_geometric_cache_v1_20260912`）能产出通过现有 v6 准入的合法窗口，且新增非 GRAB 覆盖显著（非 GRAB 窗口占比、物体实例数、approach 占比至少一项有量级变化） | 本地存在一条**不与 NAS 冲突**的扩集路径 | 下一轮做“绑定本地 legacy 根的 v6 recipe”（先小批后全根），并把 6k/12k 计划改为“按实际可绑定源切片” |
| 能转换但新增覆盖极小（仍<5% 非 GRAB 窗口）或物体与现有重叠 | 本地盘上数据**不能**解决跨源泛化 | 停止在本地承诺“训练集做好”；把决定权交回机主（挂 NAS / 腾盘 / 接受 GRAB-only），并只做采样与相位重平衡 |
| 转换不可行（字段/标注/注册表/归一化缺失不可弥补） | 同上，但附带明确的“缺失清单” | 报告必须给出“要补什么、大概多少工作量”，不许泛泛说“格式不同” |

本轮试点结果落在第三行的**工程准入阻断**分支：6 条 pilot 的 1,564 个 H4/K24 候选窗口中，现有 `sample_window` 准入为 **admitted 0 / rejected 1,564**。这不是把 legacy 数据判为“不可用”：阻断是 v6 语义适配器、有效性和标签绑定尚未存在；数据本身仍可作为后续适配输入。本轮没有启动长训，也没有触碰 goal contract 或 `wam-rl-policy-v3`。

## 1. 本地子集与全配方

下表的窗口是 TRAIN 窗口；本地缓存另有 DEV-only 序列，因而“本地缓存序列总数”仍为 1,177。全配方数字来自冻结 parent plan，非 GRAB payload 的路径仍指向未挂载的 `/mnt/ugreen_nas`。

| source | 本地 TRAIN 序列 | 本地 TRAIN 窗口 | 本地物体 | 本地 TRAIN phase | 全配方序列 | 全配方窗口 | 全配方物体 | 全配方 phase |
|---:|---:|---:|---:|---|---:|---:|---:|---|
| 0 OakInk2 | 3 | 96 | 3 | 未单独测量 | 568 | 2,694,722 | 106 | 冻结计划未物化 |
| 1 GRAB | 1,040 active（1,132 cached） | 33,272 | 45 | approach 4.122%; close 8.354%; contact 22.654%; hold 64.597% | 1,132 | 225,215 | 45 | 冻结计划未物化 |
| 2 | 3 | 96 | 12 | 未单独测量 | 301 | 351,309 | 22 | 冻结计划未物化 |
| 4 | 3 | 30 | 2 | 未单独测量 | 33,055 | 328,731 | 2,066 | 冻结计划未物化 |
| 5 | 3 | 18 | 8 | 未单独测量 | 2,308 | 38,185 | 161 | 冻结计划未物化 |
| 6 | 3 | 66 | 23 | 未单独测量 | 287 | 51,870 | 33 | 冻结计划未物化 |
| 8 | 3 | 12 | 6 | 未单独测量 | 100 | 351 | 20 | 冻结计划未物化 |
| 9 | 3 | 4 | 4 | 未单独测量 | 524 | 6,492 | 8 | 冻结计划未物化 |
| 10 | 3 | 78 | 4 | 未单独测量 | 1,251 | 67,324 | 286 | 冻结计划未物化 |
| **合计** | **1,177 cached** | **33,672** | — | **GRAB 98.812%** | **39,526** | **3,764,199** | — | **GRAB 6.0%** |

本地子集与全配方**不是同一分布**（本地 GRAB 98.8% vs 配方 GRAB 6.0%）。因此“从本地挑 6k curated”只能重平衡 GRAB 内部的阶段覆盖与采样；本轮仅调整本地 GRAB 类数据的阶段覆盖与采样，没有新增跨源覆盖；跨源改善尚未验证，也不作为本轮验收门槛。全配方中 source0 单独占 2,694,722 / 3,764,199 = 71.6% 窗口，而本地只有 96 个 source0 TRAIN 窗口；按本轮已核实口径，本地 source0 仅 3 序列 / 96 窗口 / 3 物体。

## 2. 先做的母序列差集

三个 GRAB 派生目录按 `parent_sequence_id` 去重后不能相加：

| 集合 | 唯一母序列 | 已在当前 1,132 条 GRAB TRAIN 计划 | 真正未入当前 TRAIN |
|---|---:|---:|---:|
| 本地 `dexplore_grab` | 1,335 | 1,132 | 203 |
| `cm_decoder_v2` | 348 | 297 | 51 |
| `coupled_geometric_cache_v1_20260912` | 285 | 244 | 41 |

`cm_decoder_v2` 与 coupled 的交集仍按同一母序列计数；不能把 348、285 或 1,335 重复相加为新增演示。对已经入训且仍有本地 v6 overlay 的 1,040 条 active TRAIN 母序列，按现有 `history_admitted` 加 H4/K24 未来 object-pose 有效性做了便宜上界审计：当前窗口 33,272，满足这两个门槛的完整 tick 287,857，其中 254,585 个不在当前窗口构造中。它们还没有 contact/event 标签，因此只是“可能扩展”的上界，不是可直接加入的监督数量。下一步若只做 GRAB 扩展，应先从这些差集/未吃入 tick 中生成并审计标签，再谈 6k/12k 配方。

## 3. 两根非 GRAB legacy 的合同审计

### `cm_decoder_v2/dexplore_rl_v1_3_full10135`

- **已有输入**：`q_native[T,18]`、`q_finger_real[T,6]`、`wrist_pose_world[T,4,4]`、`wrist_pose_object_t[T,4,4]`，30 Hz；窗口使用连续 4 帧 Cm cache。其几何指向 Inspire `knn_hand_points_world[T,10135,3]` 与 4096 点 object pool。
- **hand/joint**：缺少 v6 的 `joints[T,2,21,3]`、`joint_valid[T,2,21]` 及其 MANO-style 点位语义。用 URDF FK 可以得到 Inspire link/表面点，但数组尺寸对齐不等于 21 点语义兼容，不能直接准入。
- **contact/event**：缺少 `contact.npy`、`part.npy`、`valid.npy` 和 v6 event sidecar。`prepare_surface_contact.py` 只有在 hand/surface 已转换为 v6 合同后才可重算 contact；随后可用 `prepare_events.py` 重算 event，当前代价是每条序列一次完整几何接触扫描。
- **object pose**：该 root 没有 `object_pose_valid`；coupled 的同名母序列有 `obj_pose_world`，但必须显式绑定、逐帧有限性检查并生成 v6 validity mask。
- **object-surface registry**：已有 4096 pool 和每序列 sampling manifest，没有 v6 2048 `surface/normals/query` registry 及 metadata identity；需按 `object-surface` 工具重建/绑定。
- **normalizer**：缺失。准入只能继承冻结 v6 TRAIN normalizer；不能用这两根的统计量替代，也不能在 pilot 中重估。
- **source split**：有 train/val/test，需映射到 v6 train/dev 且继承母序列分割，防止同一演示跨集合。

### `coupled_geometric_cache_v1_20260912`

- **已有输入**：`obj_pose_world[T,4,4]`、4096 object pool、`obj_knn_indices[T,4096,32]`、`knn_hand_points_world[T,10135,3]`、`hand_points_world[T,1538,3]`、2 cm candidate/supervision mask，30 Hz，`coordinate_frame=object_pose_t`。
- **hand/joint**：没有 v6 21 点 joints/validity；10135/1538 是 Inspire 表面点流，不能只按 shape 对齐。
- **contact/event**：candidate mask 和 `hand_supervision_mask_2cm` 是距离 proxy，不是 v6 contact confidence、part label 或 release/contact event。contact/event 仍需先完成 v6 hand semantic adapter，再运行 `prepare_surface_contact.py` / `prepare_events.py`。
- **object pose**：有 finite `obj_pose_world`，但没有 `object_pose_valid`；需要把有效性判定、单位和 t+24 剩余帧数写入新绑定。
- **object-surface registry**：manifest 内嵌 4096 pool 的采样 correspondence，不等于 v6 的 2048 surface/normals/query registry；需重采样或建立可审计的 v6 registry identity。
- **normalizer/source split**：同上；train/val 可映射，但必须继承 v6 train/dev 分割并复用冻结 TRAIN normalizer。

两根都满足 30 Hz、`t+24` 帧数可检查这一层，但这不能替代坐标系/单位/事件剩余帧数一致性证明。Inspire 点流和现有手部表示的点位语义也未证明兼容。现有工具可以重算 FK、surface contact 和 event sidecar；不能在不增加新语义适配器的情况下补出 v6 21 点监督。

## 4. 六序列 pilot（两源各三条）

选择规则是每个 root 的 TRAIN manifest 按 `parent_seq_id` 字典序取前三条，不按物体身份筛选。两根选择的是同三条母序列（`s1/airplane_lift`、`s1/alarmclock_see_1`、`s1/apple_pass_1`），所以 6 条记录只有 3 条独立母序列；不能把两种 cache view 算成 6 条独立演示。

写盘前估算 required v5 arrays、contact/event arrays、metadata 和 staging padding 共 **40,259,322 bytes（约 0.0375 GiB）**；写前 `/home2` 可用 **139.0233 GiB**，估算后仍远高于 20 GiB floor。实际只写 6 个候选记录和审计 JSON，pilot 目录 allocated **24,576 bytes（约 24 KiB）**；写后可用 **139.0233 GiB**。GPU 使用 **0 秒**。

| root | 序列 | 帧数 | H4/K24 候选窗口 | admitted | rejected | object instance |
|---|---|---:|---:|---:|---:|---:|
| cm_decoder_v2 | s1/airplane_lift | 432 | 405 | 0 | 405 | 1 |
| cm_decoder_v2 | s1/alarmclock_see_1 | 254 | 227 | 0 | 227 | 1 |
| cm_decoder_v2 | s1/apple_pass_1 | 177 | 150 | 0 | 150 | 1 |
| coupled | s1/airplane_lift | 432 | 405 | 0 | 405 | 1 |
| coupled | s1/alarmclock_see_1 | 254 | 227 | 0 | 227 | 1 |
| coupled | s1/apple_pass_1 | 177 | 150 | 0 | 150 | 1 |
| **合计** | **6 records / 3 unique parents** | — | **1,564** | **0** | **1,564** | **6 records / 3 unique parents** |

准入尝试调用现有 `ptv3_shared_wam_v5.data.sample_window` 边界，首个缺失字段即拒绝；记录中的完整缺失清单包含 `joints[...21,3]`、`joint_valid`、`object_pose_valid`、`contact/part/valid` 和 v6 object-surface registry binding。因没有合法 V5Windows，语义 phase 分布为 **N/A**，不能把 legacy proximity mask 当 phase 标签。作为只读诊断，6 条记录的距离 proxy 合计为 `approach_gt_5cm=243`、`close_2_to_5cm=10`、`contact_proxy_le_2cm=610` 帧；这组数字不进入训练，也不作为 phase 覆盖验收。

因此 pilot 的**新增窗口数为 0**，新增非 GRAB 窗口占比为 **0.0%**（没有合法新增窗口，分母按工具 receipt 固定为 0）；approach/close/contact/hold 的 v6 语义分布均为 N/A，物体记录数为 6、去重后的母序列/物体实例为 3。

## 5. 决定与复现

本轮没有达到“新增合法非 GRAB v6 窗口”的第一行门槛，也没有证据说明数据本身不可用。近期可执行路径按成本排序为：

1. 先对 203 条真正未入训 GRAB 母序列和 254,585 个未吃入的 H4/K24 候选 tick 做标签绑定审计；这只增加同源实例/阶段内容，不宣称跨源覆盖。
2. 若要使用两根 Inspire legacy，先补一个显式版本化的 Inspire→v6 21 点语义 adapter，并为 `object_pose_valid`、surface registry、contact/part/event 生成可追溯 sidecar；完成 6 条 pilot 后再扩大到不超过 50 条。
3. 跨源覆盖仍需要可访问的 source0/source2/source4/source5/source6/source9/source10 payload。腾盘只解决容量；`/mnt/ugreen_nas` 不存在时不能解决 pending payload 不可访问。

复现（不启动训练，源数据只读）：

```bash
cd /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent
python3 src/task/ptv3-shared-wam-v6/tools/audit/audit_local_legacy_roots.py \
  --output outputs/ptv3-shared-wam-v6/local-legacy-feasibility-20261023-r1
cat outputs/ptv3-shared-wam-v6/local-legacy-feasibility-20261023-r1/audit.json
```

完整 JSON receipt：`outputs/ptv3-shared-wam-v6/local-legacy-feasibility-20261023-r1/audit.json`；pilot 候选记录位于同目录 `pilot-candidates/`。本工具只新增本报告和 `audit_local_legacy_roots.py`，不修改 `curate_tau_training_set.py`、既有 tau 报告或 `wam-rl-policy-v3`。
