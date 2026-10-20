# WAM-RL Policy v3：显式任务 goal 的 BC/PPO

本 Task 承载 `teacher rollout TaskSpec → TaskController → task10 / WAM goal → BC/PPO`。
当前执行器是 Inspire current421 / scheduled686，包名 `wam_rl_policy_v3`；
Shadow current300 / scheduled565 的主要实验继续位于 [v2](../wam-rl-policy-v2/README.md)。
共同容量、官方 PPO 适配与几何工具在此保留独立代码副本，不从 v2 包动态导入。

- `tools/run/train_measured_bc.py`：完整准入语料的 Current421 BC。
- `tools/run/generate_measured_tau_cache.py`：与 task10 同源的 WAM 条件，layout v2。
- `tools/run/train_measured_scheduled_bc.py`：严格零扩展的 Scheduled686 BC，packet v2。
- `tools/run/inspire_measured_bc_eval.py`、`inspire_scheduled_bc_eval.py`：有界单回合原生评估及采样 PPO collector smoke。
- `tools/run/inspire_physics_host.py`：同轮实测 state/force/support 与原生 PD。
- `tools/run/closed_loop_scheduled_shadow.py --task-controller --evaluate`：UniDex575维对照，
  复用已训练565维PPO，task10零扩展且与WAM同一决策；仅评估，不迁移optimizer。
- 同入口 `--bc-evaluation --evaluate --official-ppo`：直接测 `--policy` 的BC均值，
  禁止传 `--rl-checkpoint`，不创建PPO/critic/optimizer；可选 `--task-controller` 检查575维接线。
- [接口与证据](docs/research/2026-10-19-measured-task-interface.md)。
- [当前状态与新缓存](docs/STATE.md)。
- [迁移清单](docs/migration.json)。
- 用户审计参考原目录的 `docs/user/ref1_2.md`；用户原文仍在 v2 目录。

新输出只写 `outputs/wam-rl-policy-v3/<run_id>/`。历史数据保持原位只读复用：
`outputs/wam-rl-policy-v2/inspire-teacher-rollout-20261019-r5/` 及
`inspire-teacher-rollout-audit-20261019-r2/manifest.json`；其软链接依赖 r4/audit-r1，不能删除。
旧audit v1存在桌板轴向错误，仅供历史追溯。当前准入为
`outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3/manifest.json`（schema v2），
修正桌板薄轴/朝向后的支撑判断；当前loader拒绝旧v1，旧τ缓存不得直接配新准入。
不复制或搬动已有 checkpoint/cache，不复用旧双调度源 τ 作为新条件。
实验 ID 保持不变，所属 Task 和代码路径迁至 v3，历史 commit/run 路径保留用于追溯。

已修复：新增265列清零；task10/WAM共享目标；训练采样动作；单回合终止收尾；
实时支撑反馈；按环境绑定几何；shaping/GAE统一 γ=.96。
桌面支撑使用实际mesh薄轴向上的面、切向范围和法向力；Shadow任务反馈使用净接触力，
不将包含内部solver力的六维力传感器读数冒充接触力。
多回合训练仍需显式 reset 后重新观察/采样；旧 Inspire launchers 只用于历史路线，
不作为新421/686长训入口。当前不据旧 Scheduled BC 的微小 DEV 差异推断 WAM 收益。

```bash
PYTHONPATH=src/task/wam-rl-policy-v3/src python3 -m pytest -q src/task/wam-rl-policy-v3/tests
```
