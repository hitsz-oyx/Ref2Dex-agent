# v6 online goal contract repair (2026-10-23)

| 可能结果 | 含义 | 会改变哪个决定 |
|---|---|---|
| 掩码能逐 query 翻 False，且 40 面板成功率/τ 预测**有变化** | 硬编码有效性确实是真实训推不一致 | 放行预算封顶的 τ 训练 |
| 掩码翻 False 但 40 面板结果**完全不变** | 该不一致对这些 rollout 是惰性的 | 记为负结论，不再当阻塞，同样放行训练 |
| 无法在不改语义的前提下接入 | 契约分层有问题 | 停训练，先报给机主 |

本轮只完成真实部署推理路径的契约修复，没有启动 40 面板、规模训练或 GPU
长任务。因此上表是后续面板结果的决策解释表；本轮证据只回答“能否在保持
`spatial_path_waypoint` 语义的前提下把 mask 接到 v6 模型输入”。

## Contract

v6 的唯一 goal conditioning 语义是
`GOAL_SEMANTICS="spatial_path_waypoint"`：调度器同 ID 点在当前物体坐标系下、
24 步 @30 Hz 的位移。观测到的 t+24 终点是训练标签/审计参照，不是合法的
conditioning。`online()` 将 scheduler waypoint pose 转成每个 flow query 的
delta；缺失 pose、含非有限值的 pose，以及非有限 flow query 都在
`object_goal_valid` 中逐 query 标为 False，并在送模张量中零填充。

v6 包不能在生产安装中依赖 v4 training Task，因此
`online_runtime.py` 内含与 v4 `_prepare_goal_inputs` 逐项等价的本地 helper。
这是 v6 的部署语义来源；等价性由 targeted test 对同一输入逐位比较两处输出，
不是第三套不同语义实现。任何显式声明为其他语义（例如
`observed_t_plus_24_endpoint`）的 `online()` 或 `predict()` 调用都会抛出
`ValueError`。

## Changed paths

* `src/ptv3_shared_wam_v6/online_runtime.py`
  * 新增 v6 包内等价 goal helper 与 `GOAL_SEMANTICS`/契约常量。
  * `online()` 支持缺失/非有限 goal 的零填充和逐 query mask，并清理 goal
    point 与 scene/start flow query 的非有限值。
  * `OnlineV6Runtime.predict()` 接受缺失 goal，拒绝错误语义；每个
    `collate()` 后的实际模型 batch 运行 dtype、shape、有限性和 invalid-query
    零填充断言。
* `tests/test_v6_goal_contract.py`
  * v4/v6 helper 逐位等价；online 有效/缺失/非有限三态；真实 `predict()`
    模型边界三态；错误语义负控制。

## Reproduction and evidence

Environment: `graspenv`, Python 3.8, PyTorch CPU seam. No training or GPU job was
started.

```bash
PYTHONPATH=src/task/ptv3-shared-wam-v6/src:src/task/ptv3-shared-wam-v5/src:src/task/wam-rl-policy-v3/src \
  /home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q \
  src/task/ptv3-shared-wam-v6/tests/test_v6_goal_contract.py
# 6 passed in 2.24s

PYTHONPATH=src/task/ptv3-shared-wam-v6/src:src/task/ptv3-shared-wam-v5/src:src/task/wam-rl-policy-v3/src \
  /home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q \
  src/task/ptv3-shared-wam-v6/tests
# 153 passed, 1 skipped, 7 warnings in 18.48s
```

The three deployment-path negative controls are:

1. `online(..., goal=None)` → all 32 `object_goal_valid=False`, finite zero
   `goal_object_delta` and zero `goal_object_points`.
2. `online(..., goal=<pose containing NaN>)` → the same finite zero/masked result;
   no NaN reaches the sample.
3. `online()` and `predict(..., goal_semantics="observed_t_plus_24_endpoint")`
   → `ValueError: goal semantics must be spatial_path_waypoint ...`.

The same controls were exercised through `OnlineV6Runtime.predict()` after the
`collate()` boundary (CPU seam); the command emitted:

```text
missing: valid=[[False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]] finite=True invalid_zero=True
nonfinite: valid=[[False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]] finite=True invalid_zero=True
wrong_semantics: ValueError: goal semantics must be spatial_path_waypoint for both training and inference
```

The direct negative-control command produced:

```text
missing valid_sum= 0 finite= True zero= True
nonfinite valid_sum= 0 finite= True zero= True
wrong_semantics ValueError: goal semantics must be spatial_path_waypoint for both training and inference
```

The `predict()` test captures the batch after `collate()` and verifies all three
cases on the tensors passed to the model: valid mask is all True; missing and
non-finite cases are all False; every invalid query has exactly zero delta.

## τ training gate

The config remains
`configs/tau_curated_data_only_v1.json: status=HOLD_UNTIL_GOAL_CONTRACT_REPAIR`.

| Gate item | Status | Command/evidence |
|---|---|---|
| Goal contract tests | **达成** | The targeted command above: `5 passed`; full v6 suite: `154 passed, 8 warnings`. The captured post-`collate()` assertions cover valid, missing, and non-finite goals. |
| Scheduler-conditioned cache identities | **未达成** | `python3 - <<'PY' ... json.load(Path('outputs/ptv3-shared-wam-v6/tau-curated-20261023-r1/curated-manifest.json')) ... PY` reports the manifest's `goal_contract.current_training` is realized t+24 endpoint, `deployment_current` is scheduler waypoint, and `must_recompute_before_training` includes all goal deltas/points and cache identities. The r2/r3 manifests likewise retain `object_goal_valid_hard_code: not changed` and `training_allowed: false`. No scheduler-conditioned identity receipt exists. |

Because the second gate is still unmet, the status stays HOLD and τ training is not
released by this repair alone. A future cache rebuild must derive scheduler-style
conditions and publish identities before the config can change.
