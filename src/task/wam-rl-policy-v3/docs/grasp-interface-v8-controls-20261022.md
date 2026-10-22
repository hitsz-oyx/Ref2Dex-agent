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

### 中断与补救（不改变实验臂）

r1 在7个完整回合后，block1/WAM tick3 因 catalog `record` 被上一回合结果同名覆盖发生 `KeyError('object')`。已用 `catalog_record` 独立命名修复；物理控制与消费器不变。r1 保留1806 controls（完整1803+partial3）、91.767执行秒。r2 `--carry-run ...r1` 携带全部7个结果和中断预算，仅补剩余13个预注册回合；新进程可能有隐藏physics状态差异，不能声称跨进程严格确定性。

```bash
$PY -u $RUN --suite ablation --carry-run outputs/wam-rl-policy-v3/grasp-v8-ablation-20261022-r1 --output outputs/wam-rl-policy-v3/grasp-v8-ablation-20261022-r2 --native-gpu 6 --model-gpu 5
```

## 消融实测（r2包含r1已完成回合；不是挑选最好单例）

执行实现：r1=`a1302e639fc58c728808282305899ab6475140d4`，r2=`f307b0b06b2a6ef8fa1af509b468ec205b68afc9`。消费器未改。r2 正常退出，4483/6000 controls（含partial3）、292.542/1800执行秒，0updates。

| 臂，n=5，block0→4 | 最长连续持握秒 | ever/final | controls分布 | R上游clip分布 | 最终clip | 关节中止数 |
|---|---|---|---|---|---:|---:|
| WAM |[1,1,0,0,0]|2/5；2/5|[100,101,300,227,300]|[0,0,2,0,0]|0|1/5|
| 零τ/valid按域可用 |[0,0,0,0,0]|0/5；0/5|[300,300,66,125,66]|[0,1,2,0,1]|0|3/5|
| 无τ/valid=false |[0,0,0,0,0]|0/5；0/5|[300,203,223,300,69]|[0,0,0,1,0]|0|3/5|
| 固定几何模板 |[0,0,0,0,0]|0/5；0/5|[300,300,300,300,300]|[0,0,0,0,0]|0|0/5|

上游clip=R原始12维输出任一维超过±1+1e-6；最终clip=消费器原始输出到实际clip后相差>1e-6。两者不混报。所有20回合均计入；物理安全中止按失败保留。9可见初态字段最大差0，actual-PD独立重构最大误差3.371e-7。WAM调用分布[68,69,298,225,298]，两个零输入臂均0次；模板WAM/R均0/0。

**判断：新5/5不能视作稳定保证，本轮WAM只有2/5，未达到预期≥4/5；不能宣称窄任务稳定资格已保住。** 在冻结R/消费器下，移除τ的两种方式均0/5，提供局部输入贡献证据；未证明普遍效益，n小、依赖已训练输入分布，且不是公平重训无τ策略的样本效率对比。固定模板0/5仅否定这一个冻结模板，不能否定经优化几何控制器。此前5/5作为真实旧run观察保留，撤回把它直接归因为τ或稳定成功率的解读。

20/20轨迹的独立连续held重算与online ever/final/longest完全一致。故意把区间改为累计的mutant被拒绝：生产最长0.7秒/ever=false，mutant1.4秒/ever=true；不把测试通过当作外部接触真值认证。v7/v8实际使用的是原LiftHoldWitness；S2c stage另行只读，不能拿stage累计achieved冒充success。

产物：`outputs/wam-rl-policy-v3/grasp-v8-ablation-20261022-r{1,2}/`；全量向量与每回合路径：`outputs/wam-rl-policy-v3/grasp-v8-summary-20261022/ablation.json`。

```bash
$PY src/task/wam-rl-policy-v3/tools/run/summarize_grasp_controls.py --run outputs/wam-rl-policy-v3/grasp-v8-ablation-20261022-r2 --output outputs/wam-rl-policy-v3/grasp-v8-summary-20261022/ablation.json
```

## 五个新机器人初态实测

冻结面板5/5已跑完，**ever/final=0/5；最长持握[0,0,0,0,0]s**，未达预期≥4/5。没有替换失败样本。实际实现`eb6e562e`。

| 新初态（各n=1） | controls | R上游clip | 最终clip | 关节安全中止 | 初始腕位置距v7 / mm |
|---|---:|---:|---:|---|---:|
| airplane |67|0|0|是|72.681|
| apple |300|见全量summary|0|否|87.510|
| cubelarge |300|见全量summary|0|否|89.992|
| duck |53|0|0|是|48.955|
| mug |300|见全量summary|0|否|104.418|

原始q/dq与同步reset一致，hand FK仅浮点舍入误差（完整误差在每回合`result.json`）。面板初态与v7的手点RMS差72.7..105.0mm，初始腕位差49.0..104.4mm，确实换了机器人初态，不再是同一初态重复。

**泛化边界**：旧机器人BC catalog五例均为val，未用于v7调参；但核对WAM当前recipe实际TRAIN索引，人类源序列airplane/cubelarge/duck分别有77/102/92个TRAIN windows；apple为DEV32 windows/TRAIN0；mug未找到精确源序列，更早谱系未核。故本轮只回答**新机器人初态稳健性失败**，不声称严格WAM未见源序列或未见物体泛化。产物`registered-initials.json`、`wam-sequence-membership.json`与五个run的manifest可核对，不将BC val冒充所有预训练阶段未见。

完整产物`outputs/wam-rl-policy-v3/grasp-v8-robust-20261022-s10_{airplane,apple,cubelarge,duck,mug}_pass_1/`。下一项继续预注册GT同协议检验，不依据这些失败调整消费器/模板/预算。

## 原始物理状态独立复算命令

```bash
CUDA_VISIBLE_DEVICES='' $PY src/task/wam-rl-policy-v3/tools/run/rescore_grasp_controls.py --run outputs/wam-rl-policy-v3/grasp-v8-ablation-20261022-r2 --output outputs/wam-rl-policy-v3/grasp-v8-summary-20261022/ablation-physical-rescore.json
```

期望`PASS n=20 matched=20`；此入口从每个run冻结的observer与原始物理状态重算，不能由stage/reward累计分数冒充。新测试文件6 passed/0 failed，含故意累计持握mutant。GT参考预检查还要求205个consecutive state ticks对应204个actions，τ/valid各205份；第一帧未来仍为t+1，不用当前时刻冒充。
