# P-20261022-s2e-geometry-tau

## Decision / preregistration

区分：同一已就绪v7抓姿消费器下，WAM未来知识是否优于无需网络的物体相对参考重定向。
若WAM成功数领先至少2/5，描述性支持此接口上的预测优势；若同量级，近期优先考虑
RL样本效率/泛化价值（仍须先获执行资格）；若共同失败，保留消费器/条件失配解释，不能
据此证明WAM无价值。最便宜的可判别设计为单批两链各5次，零训练，不追加筛选批次。

- task: `wam-rl-policy-v3`
- experiment_id: `P-20261022-s2e-geometry-tau`
- run_id: `s2e-geometry-20261022-r1`
- branch: `detached` worktree `tmp/s2e-geometry-20261022`，不新建分支。
- git_commit: actual execution commit recorded in output manifest; implementation committed before run.
- work_version: `s2e-geometry-v1`
- input: S2b完整205状态reference；原同已见`s1_cylindersmall_pass_1`初态，native/model seed195/196。
- interface: `MeasuredGraspConsumer` / `GRASP_INTERFACE` v7，两个τ来源走相同R、腕姿约简、
  grasp intent、measured consumer、PD；原R及WAM11000冻结。不改共享接口文件。
- budget: 每链≤1500、总≤3000 environment controls，wall≤1800s，0 learning updates，
  产物≤2GiB；成功/原安全abort/300controls即停。GPU5模型、GPU6物理，仅空闲时用，4空闲。
- order: W/G、G/W、W/G、G/W、W/G；一个seen初态五次reset，不是五独立任务或seed。
- same initial state: 9字段对旧source初态误差≤1e-6，actualPD误差≤2e-5。

## A: Geometry definition and anti-alignment boundary

对每个参考未来帧j：`L[j] = inv(T_ref_object[j]) @ H_ref[j]`；按30Hz当前时钟t取
j=t+1..t+24，`G[k] = T_live_object @ L[j]`。无网络、无手相位匹配、无手平移/旋转
贴合。API不接受live hand。R只需8帧差分编码，`tau=G[:8]-H_live`，重建目标恒为G。
完整24帧保存并供共同腕姿约简消费，超过完整未来范围则invalid，不复制终帧。
H4/encoder availability gate和预测触发phase对两臂相同；geometry不需模型，但为同接口保留gate。
逐帧物体系定义会消去参考物体刚体抬升；抬升任务由同一v7实测postgrasp servo执行。
这与S2d以参考当前帧锚定整个未来（保留source lift）不同，不能把两者当同一对照。

## B/C: Fixed evaluation and metrics

WAM：`WAM(H_live,G,E)`，完整24未来，冻结flow seed；geometry如A。两链使用同R checkpoint、
shape/task输入、夹紧、机械安全、30Hz/1env；运行中不改参数、不拟合R/v6、不做RL。
主指标为原窄witness：≥5cm、真实≥2指0.1N、耦合、离桌连续elapsed1s；ever/final各报。
全5次最长持握elapsed秒、输出clip controls、内部R clip controls、joint-limit abort必须分别报告。
裁剪定义为任一canonical轴raw→clip差>1e-6；完整RELEASE与精确8cm目标不被此指标替代。
环境：client graspenv Python3.8/Torch2.4.1+cu121；native固定dexplore_v120_train/Torch2.0.1+cu118。

## Diff / reproduction / expected values

新增自己的`geometry_tau.py`、`run_geometry_tau_comparison.py`与`test_geometry_tau.py`，不修改共享实现。

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
 /home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q -p no:cacheprovider \
 --basetemp=tmp/s2e-tests src/task/wam-rl-policy-v3/tests/test_geometry_tau.py \
 src/task/wam-rl-policy-v3/tests/test_grasp_intent.py
# 13 passed；SE3重建/等变误差≤2e-15，无手参数，缺完整未来invalid。
REF2DEX_ASSET_ROOT=/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent \
 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
 /home2/wyy/miniconda3/envs/graspenv/bin/python -u \
 src/task/wam-rl-policy-v3/tools/run/run_geometry_tau_comparison.py \
 --output outputs/wam-rl-policy-v3/s2e-geometry-20261022-r1 --native-gpu 6 --model-gpu 5
# 工程期望COMPLETED/10回合/0updates/≤3000controls；效果不预先承诺成功数。
```

隔离worktree需在其`data/processed_data/inspire_rl_object_dexplore`软链接既有只读输入。
资产、checkpoint均从REF2DEX_ASSET_ROOT读取；输出和cache仅在隔离worktree内。

## Limitations / future evidence

本批是Decision probe，不能推导统计显著性、跨物体泛化或RL效率。参考相位按时钟，
不依赖实测接触；WAM则使用live H4与task goal，比较的是这两个具体τ来源。相位错误仍
可能损害geometry；消费器未资格时，共同失败不能证明两种未来具有同等信息量。
历史WAM2/5不与本批合并；同期匹配WAM为比较基准。完整任务、独立初态与RL在future evidence。
