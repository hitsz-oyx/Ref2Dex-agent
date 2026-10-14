# 原生连续3D数据准备

沿用用户的[路线输入](docs/user/ref/ref1.md)，从已有手参数、物体世界位姿和CAD构造Shared-WAM几何输入与未来标签。

阅读入口：[架构、空间和审计报告](docs/user/架构.md)、[当前状态](docs/STATE.md)、[官方来源调查](docs/research/2026-10-14-geometry-storage-survey.md)。当前输出是工程数据，`training_allowed=false`；没有启动多源正式训练。

代码位于`src/pose_mesh_data/`，执行入口在`tools/run/`，审计入口在`tools/audit/`。数据、日志和静态审查图留在`outputs/3d-data-preparation/`及项目`tmp/`。

授权HOT3D下载清单在仓库根目录，含临时签名URL，不提交。下载器仅选择`ground_truth/hand_data/assets`。

```bash
# 项目根目录；使用已有环境与项目临时目录
export TMPDIR="$PWD/tmp/3d-data-switch-20261014-r1"
export PYTHONPATH="$PWD/tmp/video-data-preparation-deps"
/home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest src/task/3d-data-preparation/tests -q
/home2/wyy/miniconda3/envs/graspenv/bin/python src/task/3d-data-preparation/tools/audit/audit_posemesh.py \
  --root outputs/3d-data-preparation/hot3d-posemesh-pilot-20261014-r1
```
