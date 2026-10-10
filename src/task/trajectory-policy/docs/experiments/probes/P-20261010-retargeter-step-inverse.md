---
schema: ref2dex.probe.v2
probe_id: P-20261010-retargeter-step-inverse
experiment_id: P-20261010-retargeter-step-inverse
date: 2026-10-10
task: trajectory-policy
branch: main
git_commit: pending
claim_id: C3
hypothesis_family: HF-trajectory-policy-retargeter
probe_index_in_family: 2
seed_pool: probe
seeds: [293]
decision_changed_if_positive: expose local wrist motion and velocity explicitly in the learned native retargeter
decision_changed_if_negative: separate hand geometry errors from insufficient local inverse conditioning
status: UNCLEAR
run_id: retargeter-step-inverse-20261010-r1
---

# Does an observable local wrist inverse explain the native control precision gap?

## Motivation / Decision Note

服务Mission A/ref1_1 learned R。前一双臂fit τ有信息但PD精度不够，独立审计通过。
冻结train L1 .28468、val .30446，差6.95%；train XYZ p95也38.78mm。
这是训练精度问题的线索，不能仅把失败归因于held-out coverage。
分类Decision：先区分腕部手点到motion提取问题与motion到native PD反向映射问题。
最便宜方法是现有数据6腕部DOF的局部线性标定，不重训Transformer、不仿真。

Root选择同两源/整wave split一次标定。若hand路径过5mm/.05rad screen，优先为
learned R显式提供局部运动/速度特征；这是后续结构候选，不是用固定几何替代新路线。
若privileged q过而hand未过，核对几何；若两者均不过，说明简单affine inverse不足，
不据此关闭τ接口或声称不可辨识。无Mission/claim/权限改变，无新分支。

## Fixed protocol

只读 `learned-retargeter-inverse-20261010-r1/manifest.json` 绑定的两个PPO low。
各源wave0--3 train，wave4 validation，partial排除；66304/16576同上一Probe。
只比较first native control的6腕部DOF，不监督手指或选择新神经checkpoint。

每DOF固定3项输入：(next wrist q-current q)、current dq、intercept。
目标actual applied native A乘实际scale，单位为腕PD位移/旋转。
Torch CUDA FP64 least squares、train-only coefficients，validation仅一次评价。
privileged比较臂使用actual next q，**只作诊断，不能加入策略输入**。
hand臂的next wrist只来自actual下一帧11点手几何，经当前q/源reset固定rigid-root
template恢复；复用旧Task geometry源码只读，不启动旧Task实验或读取teacher packet。
报告恢复wrist与actual next q误差、两臂train/val all/startup PD p95及分DOF MAE。
预测按原生[-1,1]clip后评价，6腕部first-control screen沿用XYZ5mm/rotation.05rad。
hand臂all/startup均过为机制PROMISING，否则UNCLEAR；不是抓取controller验证。
不改变上一Probe的固定门槛、结果或checkpoint。

## Resources / stop

一张启动前idle GPU5，新增2GPUmin/16MiB，单次<=120s；GPU6个小lstsq，CPU
文件/刚体几何/误差统计。GPU preflight<=512MiB/util<=10；foreigncompute>512MiB
停止。input/source drift、nonfinite、split/unit错误保留FAILED，不调参或扩预算。
输出`outputs/trajectory-policy/retargeter-step-inverse-20261010-r1/`。
入口`tools/audit/probe_retargeter_step_inverse.py`。

## Results

Not run yet; protocol fixed before calibration.

## Limitations / future evidence

单motion/同initial/seed且失败多，3项affine形式不覆盖接触非线性/隐藏solver。
next actual手轨迹仍privileged GT intention；指令鲁棒性、手指preload、完整抓持，
predicted τ和joint forecaster/action训练、matched Cm-on/off收益仍未证明。
