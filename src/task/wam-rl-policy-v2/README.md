# WAM-RL Policy v2：scheduled current565 → BC warmstart PPO

本 Task 承载新的执行器观测路线：当前 UniDex Shadow 状态300维 + WAM 世界系
8×11×3 手轨迹264维 + 历史有效位1维，共565维。包名为 `wam_rl_policy_v2`。

任务 goal → 独立 scheduler → 冻结 WAM-v4 → tau → 全量BC初始化的PPO actor →
24维原生动作。前6拍无完整H4时 tau置零、有效位为0，仍由BC执行。
空间 scheduler waypoint 尚不是经验证的0.8秒可达目标；phase尚未作为学习条件。

- `tools/run/train_current_bc.py`：全量条件缓存及BC训练。
- `tools/run/parallel_scheduled_cache.py`：互斥分片多GPU缓存。
- `tools/run/closed_loop_scheduled_shadow.py`：native闭环及批量PPO试跑。
- `tools/audit/check_control_pipeline.py`：真实模型、归档实测输入接口检查。
- [控制合同及审计](docs/research/2026-10-18-control-interface.md)。
- [全量BC结果](docs/experiments/probes/P-20261018-full-scheduled-current-bc.md)。
- [批量PPO试跑结果](docs/experiments/probes/P-20261018-scheduled-batch-ppo.md)。

新输出使用 `outputs/wam-rl-policy-v2/<run_id>/`。
已完成产物仍在 `outputs/wam-rl-policy/full-scheduled-current-bc-20261018-local-r1/`，
初始策略直接读取其中 `fit-fast/best.pt`，无需重新生成缓存或训练BC。
WAM模型继续复用 `outputs/ptv3-shared-wam-v4/local-control-snapshot-20261018-r1/`。
checkpoint schema和数值合同保持兼容，Task目录v2不代表重新编号WAM模型或实验。

旧Task的scheduled专用入口只保留兼容跳转；其旧观测训练入口继续保留。
通用scheduler、WAM adapter、PPO数学与BC模型在本Task保留迁移时实现，
后续新观测路线在此维护；不依赖旧Task的Python包。数据准备/模型Task仍作为显式依赖。

验证（graspenv）：

```bash
PYTHONPATH=src/task/wam-rl-policy-v2/src python -m pytest -q src/task/wam-rl-policy-v2/tests
python src/task/wam-rl-policy-v2/tools/run/closed_loop_scheduled_shadow.py --help
```

迁移验证：20项本Task测试、9项旧入口兼容测试通过。真实WAM+父BC的12拍归档输入
检查通过（GPU6，11.1秒，PyTorch峰值417MiB）；BC均值保持误差2.38e-7、
PPO log-prob回放误差0。记录见
`outputs/wam-rl-policy-v2/migration-interface-20261018-r1/report.json`。
这是模型接口检查，不是native物理成功率或批量PPO训练结果。

批量PPO首轮已完成：16环境、4次更新、11520训练交互；详见上述实验卡。
当前只确认训练接口可用，尚未证实性能提升或跨物体泛化。
