# v6 → 官方 measured PPO 接线可行性（2026-10-22）

**12:4x新增真实checkpoint smoke**：已获GPU456工程验证授权，在使用前检查GPU5空闲后，
以真实`flowmatching-overnight-20261022-r2/training/step011000.pt`做真实PTv3 GPU forward
→稀疏21/11→原Scheduled686 BC→native PD，1个真实H4 PASS，WAM前向0.490903s、18目标finite。
没有替换官方PPO的V4Runtime，没有PhysX/optimizer，也不是执行成功资格。
复现命令与输出见[共同复核包](fix-review-20261022.md)；下文CPU seam结果为之前阶段，单独保留。

**当前官方PPO链只覆盖v4；v6在该入口未接线（UNWIRED）。** 本轮没有把V4Runtime
替换成v6。已有P0推理/物理executor入口和P1独立入口使用过v6，不等于官方PPO完成迁移。

## 已有接口、点序与时间

- 官方入口`src/task/wam-rl-policy-v3/tools/run/inspire_measured_official_ppo.py`仍导入并实例化
  V4Runtime，其loader拒绝v6 schema。manifest现显式写wam_version=v4、trajectory_shape=
  [24,2,11,3]、v6_connection=UNWIRED，防止Task名字被误读为WAM版本。
- v6为[24,2,21,3]，右/左slot固定0/1。保留canonical11：wrist、五finger base、五tip。
  `ptv3_shared_wam_v5/contracts.py`的21→11索引是
  **[0,1,5,9,13,17,4,8,12,16,20]**，不可直接取前11点。
- 原生Inspire11先按NATIVE_TO_WAM=[0,1,3,5,7,9,2,4,6,8,10]进入canonical11，再嵌入21。
  缺失10关节、另一手slot与778 skin全部invalid，不能补零后当有效观测或凭空重建。
- 输出选同11点，anchor displacement乘当前object rotation转world，再按
  WAM_TO_NATIVE=[0,1,6,2,7,3,8,4,9,5,10]交给旧Scheduled686。它吃前8帧、30Hz：
  τ[0]是第一未来帧t+1/30，不是当前帧；8步=8/30s，不套Shadow60Hz插值。
- v4训练的执行器直接消费v6的11点投影可能发生分布迁移。shape吻合不代表执行资格；
  P0标准GT都未通过，不能据预测失败断言生成器是瓶颈。

## event / goal 如何进入，现有什么缺口

已有`src/ptv3_shared_wam_v6/online_runtime.py:online`只读真实H4；将当前pose逆变换作anchor，
物体CAD固定2048点，goal_object_delta=当前32个query经过local_goal_pose的SE3位移。
传入`decision.next_event.wam_condition()`：event_type NONE/CONTACT/RELEASE=0/1/2、
event_delta_steps=delta_seconds*30、event_valid、event_hz=30。不读取未来实测标签。
TaskScheduler的event是已知计划/恢复指令，不是预测的接触真值。

当前MeasuredOfficialEnv只收集v4 samples并调用predict_hand；而OnlineV6Runtime.predict要求
(objects_H4,canonical_hands_H4,goal_poses,event_packets,object_codes)，还要求dense2048
cache及v6 stats。不能仅改import或把21 reshape成11。完整迁移还需要：

1. 显式选择runtime、验证模型/normalizer/schema/几何身份，并给resume身份增加版本与布局；
2. 每环境收集真实H4、当前同一TaskController的goal/event，逐环境处理workspace及reset mask；
3. 对齐30Hz/point slots/起步3帧invalid、固定共同初态与随机流，并重验动作/PD及原生reset；
4. 让可执行机器人参考成为**同指标**正控制，确认v6 τ执行器资格后才比较预测与GT。

此外，v6预训练goal多是实际t+24终点，而部署传入的是没有到达时间保证的spatial waypoint；
这仍是已核语义差异，不能以packet shape通过就说goal条件已经适配。

## 本轮最小smoke：1个真实H4，端到端只到PD边界

命令：

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
/home2/wyy/miniconda3/envs/graspenv/bin/python \
  src/task/ptv3-shared-wam-v6/tools/audit/smoke_v6_ppo_feasibility_cpu.py \
  --output outputs/ptv3-shared-wam-v6/negative-result-fixes-20261022-r1/v6-boundary-smoke.json
```

**PASS**：真实`s7_airplane_lift_Retake`4帧→真实online builder的2048 CAD+稀疏21+goal/event→
SharedWAMV6 forward [1,24,2,21,3]→[8,11,3] native/world τ→真实已保存Scheduled686
step50000 actor→12维clamp→18维native expansion→原生Inspire PD方法；最终18目标均finite。
左slot与未观察点mask正确。回归测试：v4 loader拒绝v6 checkpoint、语义点映射、goal/event
及mask往返，共2个新增测试，见`tests/test_ppo_model_identity.py`。

**scope限制**：CPU smoke使用小型随机v6和geometry测试seam，不是训练checkpoint/GPU PTv3
前向，不含PPO optimizer或PhysX。因此只证明这些边界可组合，**不证明v6已接官方PPO、
物理端到端通过或策略成功**。本轮GPU0–3均被其他作业占用，4–7未用；没有为报告另开仿真。

## 可读到的既有真实checkpoint执行证据（本轮未重跑）

P0 `outputs/ptv3-shared-wam-v6/executor-qualification-20261022-r1/`已有healthy4371与FM8500
各16环境×300controls的真实v6推理→BC→物理轨迹，以及全头/仅手头parity：相同encoder
激活下decoder maxdiff均0，独立encoder重算漂移healthy0.03815mm、FM0.000536mm。
这是独立executor入口，不是官方PPO训练入口。按elapsed≥1s，动作回放15/16，GT/healthy/FM
均0/16；完整任务也无合格正控制。所以本轮结论仍是**迁移可行但官方PPO未接线且执行资格
未通过**，不是“v6 PPO失败”。P1继续HOLD。
