# P-20261023-scheduler-cache-identities

| 可能结果 | 含义 | 会改变哪个决定 |
|---|---|---|
| 训练条件可由调度器路径重算、identity 唯一、等价性测试通过 | 门槛 (2) 达成 | 与 goal 契约一并放行唯一封顶训练，仍需冒烟门 | 
| 只有部分窗口能重算 | 条件覆盖不完整 | 按覆盖比例决定补数据或仅使用可重算子集，并把差异写入报告 | 
| 重算后条件与现有缓存普遍不同 | 现有训练条件语义错误 | 旧 checkpoint 只能作为旧契约历史基线 B1；放行前先完成差异审计 | 
| 发现多份语义实现 | 契约分层有问题 | 停止训练，先收敛到唯一语义来源 | 

本轮实际落在“只有 0/6,000 窗口能由合法 scheduler path 重算”的阻塞情形：当前
curated payload 没有独立的 planned reference path/waypoint。它只有观测到的
`object_poses`，而这些是标签；把它们当 scheduler instruction 会重新制造
realized-endpoint 条件，所以全部 6,000 条都被明确排除。没有训练、长任务或 checkpoint
写入。

## 训练条件的现状与调用链

现有训练条件是 realized t+24 endpoint：

1. `src/task/ptv3-shared-wam-v6/tools/run/train_pretrain.py:118` 调用
   `validate_recipe`，`:126` 建立 `V6Windows`，`:207` 把 `dataset` 交给
   `DataLoader`，训练循环 `:281–299` 直接取得并送入模型。
2. `src/task/ptv3-shared-wam-v6/src/ptv3_shared_wam_v6/data.py:20` 让 v6
   reader 继承 parent v5；`:43` 调用 `super().__getitem__`。
3. `src/task/ptv3-shared-wam-v5/src/ptv3_shared_wam_v5/data.py:22–65` 的
   `sample_window` 是条件产生处：`:25` 取 `tick-3:tick+25`，`:44` 计算
   `future_flow`，`:60` 写入 `goal_object_points=flow+future_flow[-1]`，`:61`
   写入 `goal_object_delta=future_flow[-1]` 和全 True 的旧 mask。这里的
   `rel[4:]` 是观测对象的 t+1…t+24，因此 `future_flow[-1]` 就是 realized
   t+24 endpoint displacement。
4. `src/task/ptv3-shared-wam-v5/src/ptv3_shared_wam_v5/data.py:155–157`
   只做 tensor stack；`src/task/ptv3-shared-wam-v6/src/ptv3_shared_wam_v6/model.py:134–143`
   消费 `goal_object_delta` 和 `object_goal_valid`。旧 reader 没有 scheduler
   import/call。

curated r7 只登记 sequence/slot/tick：选择和准入位于
`src/task/ptv3-shared-wam-v6/tools/audit/curate_tau_training_set.py:406–460`
和 `:507–515`；没有条件张量、scheduler reference 或 cache identity 字段。
因此当前 curated 窗口使用 realized endpoint，且不能声称已有 scheduler-conditioned
cache。

推理路径是另一条合同：`src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/goal_scheduler.py:430–463`
的 `update_schedulers` 产出 `decision['goal_delta']`；
`src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/batched_scheduled_inputs.py:45`
调用它，`:59` 把同一 delta 送入 WAM。scheduler 在
`goal_scheduler.py:389–403` 标注 `goal_semantics='spatial_path_waypoint'`，而不是
future endpoint。

审计没有发现第二份 scheduler waypoint 语义实现：唯一的 waypoint 产生器是上述
`goal_scheduler.py`；v6 `online_runtime.py:73–99` 只把已给定的 scheduler pose
转换成同一 query frame 的 delta、做 finite/mask sanitation，不重新决定 waypoint
语义。receipt 的 `semantic_implementations` 因此只列 scheduler source。

## 实现与证据

新增 [scheduler_cache_identity.py](../../src/ptv3_shared_wam_v6/scheduler_cache_identity.py)
提供：

- canonical float32 32×3 condition hash（dtype、shape、bytes）；
- 复合 identity：`(source_id, sample_id, window_start_frame,
  goal_semantics, condition_sha256)`，明确不把 local sample id 单独当 key；
- 通过 `OrderedGoalScheduler` + `update_schedulers` 的 scheduler replay helper，
  以及 24 steps @ 30 Hz 的合同元数据。

该 helper 只接受显式 `scheduler_reference_poses` 字段；缺失时直接报错，绝不回退到
观测 `object_poses`。

新增 [verify_scheduler_cache_identities.py](../../tools/run/verify_scheduler_cache_identities.py)
生成机器可读 receipt。当前 receipt：

`outputs/ptv3-shared-wam-v6/tau-curated-20261023-r8/scheduler-cache-identity-receipt.json`

receipt 的关键计数为：

| 项 | 数值 |
|---|---:|
| curated admitted windows | 6,000 |
| scheduler-eligible windows | 0 |
| 明确排除（无法由 scheduler path 生成） | 6,000 |
| scheduler condition hash 去重数 | 0 |
| scheduler condition hash 重复率 | 0.0（没有准入条件） |
| 旧 realized-endpoint hash 去重数 | 6,000 |
| 旧 realized-endpoint hash 重复率 | 0.0 |

旧 endpoint hash 统计只作负证据，未计入 scheduler admission。排除原因逐窗口写在
receipt 的 `excluded_windows`，统一原因是
`scheduler_reference_path_absent; observed object_poses are labels`。

新增测试 [test_scheduler_cache_identity.py](../../tests/test_scheduler_cache_identity.py)：

- 相同 local `sample_id` 在不同 `source_id` 下 identity 不碰撞；
- 训练 replay helper 产出的 scheduler delta 与 v6 `online_runtime._goal_from_pose`
  sanitizer 逐位一致；scheduler state 也由推理使用的
  `update_schedulers(..., fast_geometry=True)` 生成。结果：`2 passed`。

已有负控制也通过：训练/推理 goal tensor equivalence 与错误 endpoint semantic
拒绝测试 `2 passed, 7 deselected`。直接把错误语义送入 `v4_online_samples` 的真实输出是：

```text
ValueError: goal semantics must be spatial_path_waypoint for both training and inference
```

这证明错误 endpoint semantic 不能通过现有推理边界静默进入模型；它没有证明当前旧
curated cache 已经修复。

## 复现命令

环境为 `dexplore_repro_py38_torch222_cu121`，CPU/IO-only；本轮未使用 GPU。

```bash
PYTHONPATH=src/task/ptv3-shared-wam-v6/src:src/task/ptv3-shared-wam-v5/src:src/task/wam-rl-policy-v3/src \
/home2/wyy/miniconda3/envs/dexplore_repro_py38_torch222_cu121/bin/python \
src/task/ptv3-shared-wam-v6/tools/run/verify_scheduler_cache_identities.py \
  --curated-manifest outputs/ptv3-shared-wam-v6/tau-curated-20261023-r7/curated-manifest.json \
  --data-recipe outputs/ptv3-shared-wam-v6/local-expanded-recipe-20261021-r1/manifest.json \
  --output outputs/ptv3-shared-wam-v6/tau-curated-20261023-r8/scheduler-cache-identity-receipt.json
```

```bash
PYTHONPATH=src/task/ptv3-shared-wam-v6/src:src/task/ptv3-shared-wam-v5/src:src/task/wam-rl-policy-v3/src \
/home2/wyy/miniconda3/envs/dexplore_repro_py38_torch222_cu121/bin/python -m pytest -q \
src/task/ptv3-shared-wam-v6/tests/test_scheduler_cache_identity.py
```

```bash
PYTHONPATH=src/task/wam-rl-policy-v3/src \
/home2/wyy/miniconda3/envs/dexplore_repro_py38_torch222_cu121/bin/python -m pytest -q \
src/task/wam-rl-policy-v3/tests/test_batched_geometry.py \
  -k 'train_and_infer_goal_tensors_match_bitwise or future_endpoint_semantics_are_rejected_by_both_paths'
```

## 决策

门槛 (2) **未达成**。不能把“能重算的那部分”作为训练子集，因为当前合法可重算部分
是 0%。下一步必须先提供或物化与推理同源的 planned scheduler reference path，
再重新生成条件、identity 和 receipt；在此之前不改 `tau_curated_data_only_v1.json`
的 status，也不放行训练。
