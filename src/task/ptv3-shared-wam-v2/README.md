# PTv3-Shared-WAM v2

用户已确认的新预训练路线：[架构与数值合同](docs/user/架构.md)。复用三域数据与
加载工具；独立联合PTv3、hand双路编码、768D×12 shared和horizon标准化。
旧Task仍承担历史 v1 运行；本Task现在提供显式 q 三维合同的 robot/Chunk-R
适配入口。`q_ref`（几何源）、`q_pd`（RL660 feedback-compensated target）和
`q_real`（执行状态）在窗口中分开保存，动作损失只使用 q_pd 的
`wrist-geometry-continuous-lift.v1` 编码。远端 WAM 只需提供兼容的 v2
pretrain parent checkpoint，RL660 与几何源仍由本机只读加载。

```bash
# 所有命令使用graspenv，TMPDIR指向项目tmp，先确认GPU空闲。
/home2/wyy/miniconda3/envs/graspenv/bin/python src/task/ptv3-shared-wam-v2/tools/audit/fit_horizon_stats.py --samples 4096 --seed 18 \
  --output outputs/ptv3-shared-wam-v2/horizon-stats-main3-20261014-r1/stats.json
```

全部命令从仓库根目录运行。

```bash
/home2/wyy/miniconda3/envs/graspenv/bin/python \
  src/task/ptv3-shared-wam-v2/tools/run/train_pretrain.py --smoke --gpu 4 \
  --stats outputs/ptv3-shared-wam-v2/horizon-stats-main3-20261014-r1/stats.json \
  --output outputs/ptv3-shared-wam-v2/<fresh-engineering-run>

/home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest src/task/ptv3-shared-wam-v2/tests -q
```

训练支持torchrun与Task-local实验卡；有界监督器为tools/run/launch_pretrain.py，
必须指定config、stats、experiment-card、gpus和新output。
resume显式指向v2 checkpoint，在新目录恢复相同config/world size/optimizer/RNG；
工程checkpoint不能进入科学运行，不能把v1权重隐式迁移到v2。

当前实现已通过61项新旧Task合同测试。最终模型98,591,142参数：geometry11,092,000，
shared85,054,464；其余为输入融合、goal与双预测头。
单卡batch64工程负载峰值分配约7.1GiB，物理显存约8.6GiB；工程检查不代表预测有效。
当前研究运行与后续结果见[实验卡](docs/experiments/probes/P-20261014-ptv3-shared-v2-pretrain.md)。

机器人适配工程 smoke / Probe 入口：

```bash
/home2/wyy/miniconda3/envs/graspenv/bin/python \
  src/task/ptv3-shared-wam-v2/tools/run/train_robot.py \
  --config src/task/ptv3-shared-wam-v2/configs/robot_remote_finetune.json \
  --stats outputs/ptv3-shared-wam-v2/horizon-stats-main3-20261014-r1/stats.json \
  --source outputs/ptv3-shared-wam/rl660-corpus-20261014-r2 \
  --init-shared outputs/ptv3-shared-wam-v2/<remote-parent>.pt \
  --output outputs/ptv3-shared-wam-v2/<fresh-run> \
  --experiment-card src/task/ptv3-shared-wam-v2/docs/experiments/probes/P-20261015-ptv3-shared-v2-robot-finetune.md
```

先用 `--smoke --gpu 4` 验证 wiring；科学运行需使用远端 parent，并以
`torchrun --nproc_per_node=2` 占用 GPU4/6。不得用本地工程 checkpoint 冒充远端结果。

## 远端运行条件

远端已部署在 `/home/wbcd/workspace/oyx_ws/wam-v2-20261014/repo`；使用
`/home/wbcd/miniconda3/envs/graspenv/bin/python`，GPU0/2。默认三域输入路径已软链接
到既有NAS data-remote；统计位于本repo的
`outputs/ptv3-shared-wam-v2/horizon-stats-main3-20261014-r1/stats.json`。
只迁移统计的data路径，TRAIN数值、划分与抽样权重保持不变。

该主机默认文件句柄软上限1024不足以容纳每source128项mmap缓存。启动监督器前，
在当前shell使用 `ulimit -n 65536`；这是进程级设置，不修改系统配置或其他任务。
训练与spawn workers会继承它。硬上限1048576无需改变。

远端Torch2.0.1的fused AdamW从CPU映射checkpoint恢复时，会把step标量留在CPU。
版本d0288294在v2的restore_optimizer中显式将fused/capturable step移到参数设备；
恢复必须使用该修复。GPU回归工具为 `tools/audit/audit_optimizer_resume.py`，
`--legacy`复现旧路径，默认路径验证加载后继续更新。
大规模训练前重新确认GPU空闲、固定预算和配置；工程预检不证明长期预测效果。
