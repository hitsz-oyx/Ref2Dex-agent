# 尺寸几何、抓后 SE3 与初态覆盖（2026-10-22）

## 结果解读表

| 结果 | 结论 | 下一步决定 | 要求来源 |
|---|---|---|---|
| 同初态 A0 source-PD→同伺服成功，A1/A2 失败 | 实际任务有可执行见证，意图/接近接口是瓶颈 | 保留 R，优先训练漂移恢复残差；不重训 v6 | Hermes 修接口；外部审计要求扩覆盖 |
| A1、A2 均成功 | 共享控制有效；该初态不能证明 τ 独有贡献 | 保留简单可行链，τ 归因仍按冻结 n40 结果 | Hermes 干净消融 |
| 全臂失败，或 A0 source 准入失败 | 当前正控/伺服/任务资格未解决；不能断言任务不可能 | 分开记录原生限位、接触与末态误差；三臂 HOLD | 外部审计区分接口与任务难度 |
| A1 成功而 A2 失败 | 该固定初态与 τ 贡献相容，单 seed 不是显著性证据 | 保留候选方向，不扩大成泛化或 Cm claim | 尚未与 Hermes/外部独立复跑一致 |

本报告预注册后冻结；结果待运行。三处修复和扩覆盖均为 Decision Probe，0 次 v6/R 更新。
它区分“测量判据失真”“抓后控制不到位”“候选目标越界”；最便宜的证据是原始轨迹重放、
合同回归与每初态三链各一次。不是完整 RELEASE 资格，不恢复三臂 RL。

## 几何合同与诊断

原始 S2b205 参考：腕原点最近 **117.565869mm**，实际掌碰撞网格最近 **0.109218mm**。
旧 100mm 原点判据不可达；这不是“物体离目标太远”的解释。原生物体 XYZ 的末态误差另报。

`GraspGeometry` 从 canonical 表面点的凸包最小宽度 `w` 与最大点对直径 `d` 推导：
位置容差 `.10w`，掌面 gap `.50d`，两指 gap 与腕到候选距离 `.25w`；姿态容差
`2 asin(.10w/d)`，对应旋转造成的表面位移不超过位置容差。所有对象使用同一比例，
没有对象名/初态 ID 分支。当前 cylindersmall 为 `w=39.8077mm,d=56.3144mm`，
位置容差 **3.98077mm**；这是新协议，不能倒改 v9/v10 的 15mm 分母或成功率。
稀疏 canonical 点近似真实 CAD，未证明完整碰撞与力闭合。

首版尺寸门直接沿用五指平均距离，在原参考最小12.3544mm仍大于9.95193mm，拒绝了全部
两指抓握；这是测量对象错误，未调大比例。新尺寸阶段使用**第二近指尖 gap**，要求两根
指尖接近；实测原参考最小3.33775mm，保持原 force/coupling/support/dwell 条件后，
连续持握 **2.000s**。旧协议的全部固定阈值分支保持默认行为，v10 续跑使用修改前快照。

## 三处 diff、命令与期望

1. `grasp_geometry.py:GraspGeometry`、`pose_consumer.py:MeasuredGraspConsumer.step` 与
   `ptv3_shared_wam_v6/staged_progress.py`：到位、接近与阶段评分共用尺寸合同；旧调用默认值不变。
   尺度×2则位置门×2；刚体变换后尺寸不变；只有一指接近不能通过两指 gap。
2. `grasp_geometry.py:RelativePoseServo`：持握交接时捕获抓取朝向，并每tick由**当前实测** `T_object^-1 T_wrist`
   刷新目标腕姿 `T_object_goal T_object_wrist`（捕获的初始抓姿只作滑移诊断，避免腕已到旧目标后忽略物体滑移），同时含旋转杠杆臂与姿态误差。position-only 保持
   捕获时物体朝向；SE3 任务追踪显式朝向。平移/旋转增量范数≤`.75m/s÷30` / `2.4rad/s÷30`。
   这些限制约束**命令增量**，不把它们冒充实测物理速度保证。失握重置积分/相对抓姿。
3. `grasp_geometry.py:grasp_candidates/JointAdmission`：物体 PCA 轴、正交轴排列/正负方向与
   实测拇指—对指框架产生候选；排序**之前**检查 18 个原生目标（包含六个被动耦合关节）。
   单独越界一个被动关节必须拒绝候选；零可行候选显式记录并有界撤离，不能删回合。
   最终命令也受原生目标边界约束，但接触动力学仍可能使实际关节越界；原安全 gate 不变。

```bash
PY=/home2/wyy/miniconda3/envs/graspenv/bin/python
PYTHONPATH=src/task/wam-rl-policy-v3/src:src/task/ptv3-shared-wam-v6/src \
 $PY -m pytest -q -p no:cacheprovider --basetemp=tmp/grasp-geometry-review \
 src/task/wam-rl-policy-v3/tests/test_grasp_geometry_se3.py
# 期望全部通过，含比例、两指反例、姿态/杠杆臂、范数限速、被动关节拒绝和残差上界。
$PY src/task/wam-rl-policy-v3/tools/audit/audit_grasp_geometry.py \
 --run outputs/wam-rl-policy-v3/grasp-v10-stat40-20261022-r1 \
 --output outputs/wam-rl-policy-v3/grasp-geometry-review-origin.json
$PY src/task/wam-rl-policy-v3/tools/audit/check_grasp_reference_contract.py \
 --output outputs/wam-rl-policy-v3/grasp-geometry-review-reference.json
# 期望 origin117.565869mm / surface0.109218mm，参考 hold2.000s。
git diff c93164b5 HEAD -- src/task/wam-rl-policy-v3 src/task/ptv3-shared-wam-v6/src/ptv3_shared_wam_v6/staged_progress.py
```

## 扩覆盖冻结协议与可直接复跑命令

12 个初态，各 A1 WAM+R、A2 同 R 零 τ、A0 source-PD 接近→同 SE3 伺服；每臂600 controls、
20s；原安全中止计失败，成功不提前停。36回合≤21600controls、每worker≤2h、0updates。
A0 source 目标不能被生产 PD 精确表达时计 `source_admission_failures`，回退同 R 并单报；
这类 A0 失败不能作任务难度或不可能性证据。

选择依据为 lexical、已有工程 PASS 与几何缓存，无 policy 成绩筛选；8 个对象是**相对
v9/v10 面板新增**，其中 airplane/apple/cubelarge/duck 以前接口开发已见，不能称全新
holdout，更不是 WAM 预训练未见。另4条是已有面板对象的新序列。无对象专用常数。

```bash
PY=/home2/wyy/miniconda3/envs/graspenv/bin/python
TOOLS=src/task/wam-rl-policy-v3/tools/run
RUN=outputs/wam-rl-policy-v3/grasp-v11-coverage-review
$PY $TOOLS/prepare_grasp_coverage.py --output "$RUN"
# 先确认 GPU4/5/6 空闲；三个独立进程可并行，输出必须fresh。
$PY -u $TOOLS/run_grasp_geometry_worker.py --registration "$RUN/registration.json" --worker 0 --gpu 4 --output "$RUN/g4"
$PY -u $TOOLS/run_grasp_geometry_worker.py --registration "$RUN/registration.json" --worker 1 --gpu 5 --output "$RUN/g5"
$PY -u $TOOLS/run_grasp_geometry_worker.py --registration "$RUN/registration.json" --worker 2 --gpu 6 --output "$RUN/g6"
```

实际输出：`outputs/wam-rl-policy-v3/grasp-v11-coverage-20261022-r1/`，
预检：`outputs/wam-rl-policy-v3/grasp-v11-preflight-20261022-r1/`。
环境沿用 graspenv Python3.8.20/Torch2.4.1+cu121，native
`dexplore_v120_train` Python3.8.10/Torch2.0.1+cu118；模型/native 同卡分进程，全部保存manifest。

方法参考：[MIT 物体与夹爪相对变换](https://manipulation.mit.edu/pick.html)、
[Drake 微分 IK 的位置/速度约束](https://drake.mit.edu/doxygen_cxx/group__planning__kinematics.html)。
本实现为 Inspire 自由腕 PD 的有界近似，没有引入 Drake 求解器，不把命令准入当全动力学证明。

具名待裁决：Hermes 给出修复方向；外部审计提出扩覆盖与任务/接口区分；本线程只有实现与
原始轨迹合同验收，尚无两方独立复跑数据，不能声称三方一致。32点凸包对真实几何的充分性、
两指力与接近指尖是否对应同一接触体、动态关节超限仍未由当前模型完全解决。
