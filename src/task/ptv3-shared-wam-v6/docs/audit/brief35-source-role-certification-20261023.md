| 可能结果 | 含义 | 决定 |
|---|---|---|
| 6,000 全可认证：先存在 source artifact + role/reset 绑定 | goal 是合法部署前输入 | 构造 cache/receipt 并报告认证；本轮明确约束只做认证、训练 0 updates，后续启动另按授权执行 |
| 部分可认证，覆盖 C 较大（如 ≥80%） | 条件合法性与覆盖度冲突 | 报 C、缺失构成、按 C 训练 / 补记录两条量化代价，等机主裁决 |
| 部分可认证但 C 很小（<50%） | 现有录制不足以支撑合法条件训练 | 不训练，报补记录清单与规模，等裁决 |
| 全部确证来自对应 rollout 自身未来 | 现有 goal 条件非法 | 不训练；统一诊断缺部署前意图绑定；仅列 unknown-goal 零填充为明确配方变更选项，不启用 |
| **实测：合法 0 / 确证自身未来 0 / 不确定 6,000；覆盖 1,064 母序列** | 候选 waypoint 的角色与目标 reset 绑定缺记录；不能推导所有计划非法 | 不构造冒名 cache，不训练；报最小补证清单，不改生产代码 |

## 预先冻结的认证口径

Experiment `P-20261023-source-role-certification`，run `brief35-source-role-20261023-r1`。
本轮为 Blocker / Decision 审计：区分已知参考计划与对应执行自身未来，决定能否构造合法条件 cache。最便宜方法是沿 r7 有序 6,000 元组追查已绑定 metadata、producer、原始 artifact 和执行记录，不跑模型或仿真。

不能把 r9 的“纯观测历史不可恢复指令”外推成“所有参考轨迹不合法”。旧录制可以成为新执行的事前计划，但必须证明它与目标执行的来源角色、内容及 reset 绑定；文件存在 / mtime 更早 / 同名对象 / producer 推断都不足以补上该绑定。当前 reader 使用未来 endpoint 的代码事实与“是否存在合法候选 waypoint 来源”分开记账。

合法证据至少包含：窗口目标执行身份、source artifact 内容身份与可读引用、明确 source_role、source 对目标执行独立的 provenance、source 冻结/声明于目标 reset 之前的证据、引用该 source 的 waypoint 派生链。确证未来泄露须有对应目标执行的同身份未来读取证据；缺一条合法证据不能自动判成泄露。

训练 0 updates，无 GPU；写盘上限 1 GiB，预计审计 manifest / CSV / receipt 不超过 64 MiB，写前实测空间；不改 production、配方、goal_missing_policy 或 config status。不新增采集，不删除历史产物。以下为本次全量追查的实际结果。


## 全量认证结果

**认证对象是拟绑定的 scheduler waypoint 来源**。可认证合法 **0/6,000**，确证该 waypoint 只能来自对应目标自身未来 **0/6,000**，缺记录无法判定 **6,000/6,000**。合法-only 清单为空，覆盖 C=0；不把缺记录升级成合法，也不把它降格为确证非法。

这与 legacy reader 的另一条代码事实分开：V5 的 `future_flow[-1]` 依赖所预测记录的 t+24，覆盖这 6,000 元组。这个 endpoint 依赖事实不能排除“另有一份旧 source 在新执行前被声明为计划”的可能，也不能为未绑定的 scheduler waypoint 补身份。CSV 单列 `current_legacy_endpoint_frame` / `current_endpoint_reads_target_future` 与 `waypoint_status`，不混用分母或偷换认证对象。本轮没有让模型实际消费任何窗口。

| source | 母序列 | 窗口 | 合法 waypoint | 确证自身未来 waypoint | 不确定 |
|---|---:|---:|---:|---:|---:|
| 0 | 3 | 96 | 0 | 0 | 96 |
| 1 | 1,040 | 5,632 | 0 | 0 | 5,632 |
| 2 | 3 | 96 | 0 | 0 | 96 |
| 4 | 3 | 30 | 0 | 0 | 30 |
| 5 | 3 | 18 | 0 | 0 | 18 |
| 6 | 3 | 66 | 0 | 0 | 66 |
| 8 | 3 | 12 | 0 | 0 | 12 |
| 9 | 3 | 4 | 0 | 0 | 4 |
| 10 | 3 | 46 | 0 | 0 | 46 |
| 总计 | **1,064** | **6,000** | **0** | **0** | **6,000** |

逐窗口完整表：[window-certification.csv](brief35-source-role-20261023-r1/window-certification.csv)。顺序严格沿用 r7，有 ordinal、source_id、sample_id 三分量、tick / window_start_frame、phase、实际标签 artifact 路径与内容 hash、上游 pose artifact、角色 / reset 缺项、认证类别与母序列证据索引。人工没有抽取“容易认证”的窗口。额外完整性核对 6,000 行逐行匹配 r7，没有丢失 / 重复 / 重排。phase 仍为 approach 1,200 / close 1,200 / contact 1,800 / hold 1,740 / unknown 60。

母序列证据：[mother-provenance.json](brief35-source-role-20261023-r1/mother-provenance.json)。每条列出源序列 / 原 episode、目标标签路径及 hash、上游 artifact ID、原始 metadata 命中与缺项。通过 ID 可查 [artifact-inventory.json](brief35-source-role-20261023-r1/artifact-inventory.json) 的真实路径 / resolved path、内容或声明身份、mtime / ctime / birthtime、原始字段与 JSON pointer。汇总与环境：[receipt.json](brief35-source-role-20261023-r1/receipt.json)。

## 真实派生链追查与边界

全量 fresh scan **8,482 个 artifact / metadata 节点**，文件缺失 **0**、读取错误 **0**。旧 brief32 inventory 仅用于定位入口路径；本次重新读取实际 metadata、archive 字段、文件 stat，并对目标 object_poses 内容与 JSON metadata 求身份。也追踪了本地 preparation plan 中的 source manifest / snapshot / parent manifest、processed metadata 的 raw dependencies，以及 Shadow producer 的 launch / runtime patch / 原代码。没有将 brief32 的 full-event-chain=0 或缺 force 信号用作本次 goal 身份认证的否决条件。

| 适用 source | 实际路径链 | 代码证据 |
|---|---|---|
| 0/1/2，合计 5,824 窗口 | 原数据 / raw annotation → processed `poses.npy` → hand overlay `object_poses.npy` → r7 元组 | `v5/preparation.py:7–14` 按 job.sequence_folder 读 poses；`tools/run/prepare_hand_overlay.py:55–67` 复制位姿并记 source_sequence |
| 4，30 窗口 | 原 Shadow replay NPZ 的 `object` → SE(3) overlay → r7 元组；NPZ 的 `goal` 是另外的列 | `v5/preparation.py:17–25` 只用 object7 重建观测 pose，没有将 goal 列写成 scheduler 指令 |
| 5/6/8/9/10，146 窗口 | job.source_row / source manifest / canonical native NPZ `object_poses` → overlay → r7 元组 | `v5/preparation.py:15–16,26–27`；overlay producer 同上 |
| legacy goal，6,000 窗口 | 当前目标记录 `object_poses[t-3:t+25]` → 当前 anchor 下 `future_flow` → t+24 endpoint | `v5/data.py:25–30,44,60–61`；该表达式仍是标签来源，不认证为 scheduler waypoint |

此处 v5 路径均相对 `src/task/ptv3-shared-wam-v5/src/ptv3_shared_wam_v5/`；producer 相对该 Task。逐窗口具体 artifact URI / hash 在 CSV 与母序列表中，不只给“某类文件”。

**文件年龄不能补 reset 绑定。** 所有节点均记录真实 filesystem mtime/ctime；8,482 个节点的 birthtime 均不可得（null）。mtime 是该文件副本最后修改时间，ctime 是 inode 元数据变更时间，二者均不是原始创建时间或目标执行 reset 时间。现有文件确实早于现在、也可以供未来新执行使用，但这不足以证明它曾是这批目标记录执行前已知的计划。目标执行 reset 身份 / 顺序证明在 6,000 窗口均未绑定，故 `available_before_reset=UNPROVEN`，不能填 False 或 True。

**角色字段不能靠默认值补齐。** 本次被绑定的来源 metadata 中 `source_role` 明示命中 **0**，`available_before_reset` 明示命中 **0**。`wam_rl_policy_v3/task_spec.py:48` 的 dataclass 默认与 `:120` 的 `packet.get(...,'explicit_task_instruction')` 会自动填角色；本工具直接读原 packet / JSON / archive，不调用这些默认构造。旧轨迹的内容即使包含它自己的“未来”，只要有独立新目标执行和事前绑定，仍可合法；17 项测试中有该正例。反过来，同名对象、source_sequence 名字、mtime 早、默认 source_role 或论文中的任务名称不产生认证。

GRAB 的原始 `framerate` / `motion_intent` 在 1,040 条母序列均能读取（共 5,632 窗口），是时钟与动作意图标签的正证据；这些字段没有给出被引用的 waypoint source 路径与目标 reset 绑定。NPY object dtype 的嵌套 annotation 没有执行 pickle，未见的语义保留不确定，不能用这一限制宣称它们绝不存在。

### Shadow 目标列：保留正证据，但不补造 waypoint

3 个母序列 `s4-00000/1/2` 各 10 窗口，原 NPZ `goal` 均为 `(100,3)`，各 tick 恒定 `[0,0,0.8999999761581421]`。原 collector 的冻结 launch commit `6141c34e592085f1f102205eab45dd643429e0ae` 中 `replay_unidexgrasp.py:219,226–228,267` 表明先 reset / warm step，再 snapshot native `task.goal_pos`，最后写 NPZ。

该次运行保存的 `tmp/shadow_hand_grasp_capacity.py:393–404,506–508,548–554,807–810` 明确把目标位置由初始物体位置加固定 displacement 构造，并在 reset_target_pose 中恢复 goal_init_state；其内容不需要读 rollout 未来。这说明现有记录**并非所有任务目标都只能从未来推断**。但它仍只保存 native position target，没有对应 scheduler waypoint 派生 / TaskSpec source-role / reset 前 source artifact 绑定。把它改造成新的 `TaskSpec.shadow_goal`、补指定姿态或规划路径会加入一个新的条件构造规则，不能在本轮认证中偷偷执行。因此 30 个窗口仍记不确定，不记“非法”，也不记合法 cache。

### 已存在其他执行的计划 / 标签，不能替换 r7 的目标身份

只读核对了一个已知 teacher-prefix admission manifest（501 条记录）及 source-ceiling r1–r14 共 **14 个 registration，560 个注册 job 条目**（含历史重复，不是 560 个独立数据样本）。teacher-prefix 明示 `goal_source=same_execution_teacher_object_rollout`；其 501 条记录的 source 标签路径与 r7 exact join 为 **0**。14 个 registration 的 560 个 job 条目指向 **40 个唯一 reference 路径，40/40 文件实际存在**；job 中 `target_execution_id / target_rollout_id / target_label_path` 明示字段命中 **0**。这些 registration 没有可用于绑定 r7 目标执行的字段，因此 target join 是缺记录，不把它伪装成一次有完整 target 键的“0 匹配”。它们可证存在 reference artifact，不可据此把 reference 的合法新执行用途转贴给 r7 标签。

这些声明作为“其他执行 / 不可借用”的证据保存在 receipt.external_declarations；没有因名字像同一个 GRAB sequence 就借入角色或 reset，也没有把别次 teacher 的明确同执行标签结论扩散成 r7 全部泄露。扫描范围是 r7 已绑定 lineage 与明确上游引用，外加这批直接相关的部署对照声明；不声称穷尽整机未索引、未引用的所有文件。

## Cache 与决定

可认证子集 **C=0**，[legal-only-tuples.json](brief35-source-role-20261023-r1/legal-only-tuples.json) 是空集合。缺的就是 CSV 中全体 6,000 个元组；逐项原因均包括 waypoint-source 派生绑定、明确角色、目标执行/reset、内容冻结先于 reset、目标独立性缺证据。没有为了凑覆盖放宽任何条件。

**没有构造合规 waypoint tensor cache**，receipt.cache 为 `NOT_CONSTRUCTED_NO_CERTIFIED_WAYPOINTS`，窗口数 0；不生成伪 PASS receipt。另需如实说明：brief34 reader 目前只有 curated index / sampler 和缺失 cache 的 fail-closed 检查，尚无实际实现的外部 cache loader schema。生产接口不能仅凭之前文档的一句“绑定”宣称已经完成；本轮没有认证子集，因此没有新造一个格式掩盖这个实现缺口，也没有改生产代码。

本轮落在“无法判定”分支，**不是“全部确证未来泄露”分支**。不据此宣称所有 goal 都非法，不把它升级为 goal/event 的统一因果诊断，不擅自启用 unknown-goal 配方。旧源轨迹作为新执行已知计划这一路径仍成立；缺的是当前 r7 目标记录与该角色的证明。

## 最小补证清单与规模：先找旧日志，不立即新增采集

沿用 brief32 的逐信号分母，区分“观测信号存在”和“意图身份绑定存在”：

| 要补的记录 / 信号 | 已有证据 | 需要补证的绝对规模 |
|---|---|---:|
| source artifact URI + 固定内容身份 + 意图生产来源 | 1,064 母序列的观测数据链可定位；独立 waypoint source 指令链未绑定 | **1,064** 个母序列级 source 指令绑定；不能仅改 source_role 字符串 |
| target execution/reset identity + source 已冻结/读取的顺序记录 | 序列边界 / 文件 mtime 不等于 reset 日志；Shadow 有运行启动但无完整 waypoint 指令绑定 | **1,064** 个目标执行级 proof 包；可与上一行合成一条 manifest 记录 |
| source_role + independent source/target provenance | 原 packet 无明确角色；TaskSpec 默认不计证据 | **1,064** 个双身份/角色记录 |
| object slot / source_frames / source_fps / query IDs / hand slot 的绑定 | 观测 pose、hand、clock 在 brief32 为 1,064/1,064；source4 有 3 条 native position 目标，但不等于完整指令 | **1,067** 个对象流映射，覆盖全部 **6,000** 元组 |
| waypoint producer、版本、配置、对应 source 内容 hash / reset cursor | 当前无可追溯的 waypoint condition，不能用旧 future endpoint 替代 | 绑定上述 **1,067** 对象流；认证后才计算条件与逐窗口 hash |
| 窗口有序身份 | 本轮 6,000/6,000 已有，无需重抽 | **6,000** 条现有元组与指令链做 join；本轮未填条件 |

已有 archive 若保留这些真实生产日志，可补导出绑定而不重新跑轨迹；事后生成一份自称 available_before_reset=true 的 JSON 不构成补证。元数据的**规划载荷估算**（不是实测恢复成本）：每母序列 1–4 KiB proof + 每对象流 128–512 B 映射 + 每窗口 64–128 B，合计约 **1,610,112–5,522,688 B**，不含需恢复的原计划文件。恢复 CPU / 人工时间取决于 archive 可定位性，当前不能承诺免费恢复或给假吞吐。

如果旧日志确实不存在，无法靠补写 manifest 对旧目标执行追认；这时新的计划→reset→执行证据只能在新执行时记录。等规模条件信封沿用 brief32：**1,064 母 rollout、1,067 对象流、261,720 对象 tick、30Hz 共 8,724 秒对象流（2.423h）**；此前 0.83–5.37h 墙钟是实时倍率 / reset 开销的条件假设，原生完整反馈载荷 **134,000,640–536,002,560 B**。它是完整新录链的参考预算，**不是当前 goal 身份认证必须补 force 的结论或已测采集成本**，也不保证新数据恰好保留 6,000 个准入窗口。新条件只能绑定相应新执行，不能贴回旧 human 标签宣称修复。是否新录由机主裁决；本轮没有新增采集。

## 运行、身份与复现

冻结审计代码 HEAD：`8483df633a27e746a5f53aa2867dba8206cceebf`。环境 `dexplore_repro_py38_torch222_cu121`，Python 3.8.20、NumPy 1.24.4；CPU-only，实测 **13.334 s**。写前 `/home2` 实测可用 **147,366,625,280 B**；先声明预计写盘 64 MiB，硬上限 1 GiB。训练 **0 updates**，GPU **0**，生产代码 / goal_missing_policy / config status 均未改。

r7 manifest SHA256 `5bda6a99678b0586b496eff7b4b3ba3afddda996cdc16c43603341ed1264225f`；有序元组 hash `56b9fac09e35e41c3922041e8794232fdeb4056b1e23838928afcd31fc49aaa3`，均与 brief34 冻结身份一致。政策文件 hash 仍为 `7bfcd6e569bb205b88439a9d2bf056eff8421198a03fe614c69d1538d6308044`。

```bash
PYTHONDONTWRITEBYTECODE=1 /home2/wyy/miniconda3/envs/dexplore_repro_py38_torch222_cu121/bin/python -m pytest -q -p no:cacheprovider \
  --basetemp=src/task/ptv3-shared-wam-v6/tmp/brief35-tests-reproduce \
  src/task/ptv3-shared-wam-v6/tests/test_waypoint_source_certification.py
# 17 passed in 0.14s

# 必须使用不存在的新证据目录；无模型/重放/采集/训练。
PYTHONDONTWRITEBYTECODE=1 /home2/wyy/miniconda3/envs/dexplore_repro_py38_torch222_cu121/bin/python \
  src/task/ptv3-shared-wam-v6/tools/audit/certify_waypoint_sources.py \
  --curated outputs/ptv3-shared-wam-v6/tau-curated-20261023-r7/curated-manifest.json \
  --output src/task/ptv3-shared-wam-v6/docs/audit/brief35-source-role-20261023-r1-reproduce
```

实际审计证据 5 文件共 **11,674,645 B**；本轮完整交付 8 文件逻辑大小 **11,717,039 B**、磁盘分配 **11,739,136 B**，低于 1 GiB 上限。不含 Git object / 测试临时 fixture / 他线写入。无 checkpoint、视频或轨迹副本，无历史产物删除。
