# V8 冻结消费接口：τ 贡献与新初态（预注册）

状态：REGISTERED，尚无本轮物理结果。task=`wam-rl-policy-v3`，branch=`main`；experiment_id=`P-20261022-single-tau-calibration` 的工程资格后续；本轮不训练，RL HOLD。

决策：新接口原 WAM 5/5 是否也能由无未来输入实现？最便宜的方法是在完全相同的消费器/初态/预算下替换输入，先跑消融，再跑新初态，最后同接口 GT。若无 τ 也是 5/5，撤回对 τ 贡献的归因。若明显较差，仅支持此局部条件下的贡献；不凭 n=5 宣称普遍收益。

## 冻结协议与预期数字

- 消费器 `grasp-intent.pose-preshape-closure-objectgoal.measured.v2` 不变；原 R、step11000 WAM、30Hz、1 env、native195 / model196，原版与未使用的校准 R 构造顺序保持 v7 一致。生产初态同步 setter、实际 PD 回执验证。
- 主成功谓词不变：实测抬升≥5cm、耦合/有力接触且无桌面支撑，**连续经过1秒**；不是8cm精确到位，也不是完整 RELEASE。独立重读保存物理状态与重算连续区间，要求在线/离线 ever、final、longest 全部相等。分离短持握累计>1秒的反例必须失败；31个30Hz连续状态才有1秒。
- 消融 4×5，逐 block 循环换顺序，每回合≤300 controls/10s，总≤6000 controls、≤1800执行秒。`wam` 期望复现≥4/5，结果差于此仍全部报告；`zero-valid` 保留有效位但未来全零；`no-tau` 全零且有效位 false；`template` 当前物体位姿+固定抓姿/预形/闭合（不调用 R/WAM），其余阶段机/实测闭合/目标闭环相同。三个对照不预设成功率。记录 R 原始输出、最终输出各自裁剪和 WAM/R 调用数；template WAM/R调用期望0/0，零输入两臂WAM调用期望0。
- 固定几何模板来源是 S2b 一条**真实窄任务成功**的机器人示教：首个独立 held 状态的物体相对腕位姿、首次两指有力接触前3帧的手指预形、held 手指闭合值。前抓姿沿物体原点向腕部径向退4cm。冻结后不调参；**使用一条明示示教先验，不宣称示教量对齐的学习收益**。它不使用录制逐帧轨迹或 WAM。零输入两臂保留原 R，是更严格的同执行器信息消融。
- 新初态5×1：`s10_airplane_pass_1`、`s10_apple_pass_1`、`s10_cubelarge_pass_1`、`s10_duck_pass_1`、`s10_mug_pass_1`。选择规则：旧 BC val、engineering PASS、surface-cache 可用、按序列名排序的不同物体，排除 v7 校准序列与此前 source-PD 六条面板。冻结消费器/WAM/R，不重试不筛成功。每条≤300 controls，合计≤1500，执行≤5×1800秒。期望≥4/5作为继续扩展目标，失败照报。它们是**未用于 v7 接口调参的初态**；物体可能进入旧训练，不能称未见物体/终身未见序列泛化。
- GT 同协议2×5（WAM contemporaneous control / object-anchored GT），总≤3000 controls、≤1800执行秒。复用 S2d `object_anchored_targets`，一整个24帧未来使用 `T_live_obj @ inv(T_ref_obj[t])`，编码为未来减当前实测手，t+1是第一帧未来、30Hz、前8帧送 R，与 WAM 同消费器/阶段机/PD。固定源 `src/task/ptv3-shared-wam-v6/outputs/s2b-reference-r2/reference.npz`，205状态；源物理连续 held≥1秒是前置门；source-PD 命令**不执行**，也不冒充 GT→R 正控制。t+24超出记录时 invalid，不伪造参考。期望 GT≥4/5是待检验工程目标，未通过不能称公平上界已建立。

## 可独立复现

环境：`/home2/wyy/miniconda3/envs/graspenv/bin/python`（Python3.8/Torch2.4.1），native `/home2/wyy/oyx_ws/.runtime_envs/dexplore_v120_train/bin/python`（Torch2.0.1/cu118）。启动入口会再查GPU，4/5/6被占即拒绝，无抢卡/无重启他人进程。

```bash
PY=/home2/wyy/miniconda3/envs/graspenv/bin/python
RUN=src/task/wam-rl-policy-v3/tools/run/run_grasp_interface_controls.py
$PY -u $RUN --suite ablation --output outputs/wam-rl-policy-v3/grasp-v8-ablation-20261022-r1 --native-gpu 6 --model-gpu 5
for SEQ in s10_airplane_pass_1 s10_apple_pass_1 s10_cubelarge_pass_1 s10_duck_pass_1 s10_mug_pass_1; do
  $PY -u $RUN --suite robust --sequence "$SEQ" --repeats 1 --output "outputs/wam-rl-policy-v3/grasp-v8-robust-20261022-${SEQ}" --native-gpu 6 --model-gpu 5
done
$PY -u $RUN --suite gt --output outputs/wam-rl-policy-v3/grasp-v8-gt-20261022-r1 --native-gpu 6 --model-gpu 5
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 $PY -m pytest -q -p no:cacheprovider --basetemp=tmp/v8-source-tests src/task/wam-rl-policy-v3/tests/test_evaluation_sources.py
```

所有输出目录必须不存在；改 run_id 可复跑，不能覆盖原产物。runner 保存执行源快照、原始状态/τ/R动作/实际PD、每回合完整结果、独立连续评分和退出回执；manifest 记录实际代码提交。无硬件确定性保证，只提供相同 seed/初态/构造顺序与数值证据。评分重算独立于阶段机累计奖励，仍共享接触观测定义，不是外部物理真值验证。
