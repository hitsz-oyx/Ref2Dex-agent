# PTv3-Shared-WAM v2

用户已确认的新预训练路线：[架构与数值合同](docs/user/架构.md)。复用三域数据与
加载工具；独立联合PTv3、hand双路编码、768D×12 shared和horizon标准化。
旧Task仍承担当前660机器人微调。本Task尚不提供robot/Chunk-R/native执行入口。

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
