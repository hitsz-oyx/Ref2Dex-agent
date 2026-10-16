# 原生连续3D数据准备

沿用用户的[路线输入](docs/user/ref/ref1.md)，从已有手参数、物体世界位姿和CAD构造Shared-WAM几何输入与未来标签。

阅读入口：[架构、空间和审计报告](docs/user/架构.md)、[当前状态](docs/STATE.md)、[官方来源调查](docs/research/2026-10-14-geometry-storage-survey.md)。当前输出是工程数据，`training_allowed=false`；没有启动多源正式训练。

代码位于`src/pose_mesh_data/`，执行入口在`tools/run/`，审计入口在`tools/audit/`。数据、日志和静态审查图留在`outputs/3d-data-preparation/`及项目`tmp/`。

2026-10-15新增[清洗结果与复现说明](docs/research/2026-10-15-native-cleaning-results.md)。当前固定入口是
`outputs/3d-data-preparation/native3d-clean-mixed-20261015-r1/`：37,591段、427,707个TRAIN窗口、
105,952个时钟已核实DEV窗口，另有8,933个HO-Cap待核实时钟DEV窗口不在默认reader中。
已做法线修复、QA/时钟断点及完整未来窗口过滤、HOI4D显式15→30Hz插值和逐行provenance冻结。
使用`CleanPoseMeshWindows(..., engineering=True)`与`clean_collate`，模型字段和来源元数据分开保留；
旧`PoseMeshWindows`会拒绝清洗快照，避免重新造出已排除的窗口。工程通过与正式训练准入分别记录。

读取时将本Task和`src/task/ptv3-shared-wam/src`加入`PYTHONPATH`。复查入口：

```bash
/home2/wyy/miniconda3/envs/graspenv/bin/python src/task/3d-data-preparation/tools/audit/audit_clean_corpus.py --help
/home2/wyy/miniconda3/envs/graspenv/bin/python src/task/3d-data-preparation/tools/audit/audit_clean_mixed_batch.py --help
```

每源24步hand/scene的TRAIN count/mean/M2由`tools/run/accumulate_clean_moments.py`单独处理，
以后使用`pose_mesh_data.statistics.mix_moments`按指定source概率组合共同统计；不把DEV或TEST用于拟合。

适配当前v2 encoder时使用`outputs/3d-data-preparation/native3d-clean-encoder-20261015-r1/`：
427,666 TRAIN/105,946 verified DEV，另8,933 HO-Cap nominal DEV。当前观测输入域额外排除UniDex的41 TRAIN/6 DEV窗口，
原完整清洗版保留；对应TRAIN充分统计在同目录`source-moments.json`，已同步扣除排除窗口贡献。
35项测试、96窗端点复算、七源正反混合batch通过，数据处理阶段已完成。

新增[六源到完整v2模型的接入记录](docs/research/2026-10-15-native-v2-integration.md)：
`native-training-recipe-20261015-r1`保存显式source采样草案和共同TRAIN标准化参数，
`native-v2-smoke-20261015-r2`完成GPU3上完整98.6M参数模型的四步工程验证；41项Task测试通过。
相关入口为`tools/run/prepare_native_recipe.py`与`tools/audit/smoke_native_v2.py`。
代码仅在本Task中增加native adapter，原v2构造器和三源统计门禁保持不变。
当前允许用途仍是工程检查，长训配方和parent输出标准化变换另行确定。

全量扩产已完成当前授权范围，用户已取消400GB产物上限；`python3 src/task/3d-data-preparation/tools/audit/status_geometry.py`查看实际进度。HOT3D全量转换、UniDex批采集、TACO全量下载/转换与已授权HO-Cap/DexYCB复用的产物自动汇入`outputs/3d-data-preparation/native3d-live-corpus-20261014-r1/`，窗口不展开存储。新来源继续是工程语料，尚未进入模型训练。

授权HOT3D下载清单在仓库根目录，含临时签名URL，不提交。下载器仅选择`ground_truth/hand_data/assets`。

```bash
# 项目根目录；使用已有环境与项目临时目录
export TMPDIR="$PWD/tmp/3d-data-switch-20261014-r1"
export PYTHONPATH="$PWD/tmp/video-data-preparation-deps"
/home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest src/task/3d-data-preparation/tests -q
/home2/wyy/miniconda3/envs/graspenv/bin/python src/task/3d-data-preparation/tools/audit/audit_posemesh.py \
  --root outputs/3d-data-preparation/hot3d-posemesh-pilot-20261014-r1
```
