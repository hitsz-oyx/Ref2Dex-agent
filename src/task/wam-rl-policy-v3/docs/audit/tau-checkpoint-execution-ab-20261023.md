# 结果解读表

| 本轮实测 | 解释 | 决定 |
|---|---|---|
| 旧 2/20 → 新 2/20；净 +0 | 新增成功 1、丢失成功 1 | 新权重未带来净成功收益；改善集中在关节/桌面安全中止减少，接近→闭合推进更少，接触/抬升/持握达成未增。不提升为主线，定位前8帧经R输出的腕指联合与闭合准入。 |
| 安全中止 14 → 9（-5） | 成功与安全分别验收 | 不因接近或误差改善放宽安全门 |

# τ checkpoint 配对执行 A/B — 2026-10-23

新权重未带来净成功收益；改善集中在关节/桌面安全中止减少，接近→闭合推进更少，接触/抬升/持握达成未增。不提升为主线，定位前8帧经R输出的腕指联合与闭合准入。

配对成功不一致例的双侧精确检验 p=1；这是已暴露固定20初态的 Decision Probe，不宣称总体显著性或未见泛化。

## 同一执行合同

`P-20261023-tau-execution-ab` / `tau-checkpoint-ab-20261023-r1`，work_version `tau-execution-ab-v1`。原 `grasp-v11-coverage40-20261022-r1` 固定顺序前20个 reset（20物体各第一条序列），保留 native seed100–119、model seed180–199、reference和初始化；每对旧/新先后顺序交替。没有另找20例冻结文件，运行前明确声明使用原40例的固定前缀，没有新采样或按结果选样。

旧 `outputs/ptv3-shared-wam-v6/flowmatching-overnight-20261022-r2/training/step011000.pt`；新 `tmp/t38-r2/training/step001223.pt`。两臂均实跑20回合，均显式传 `goal=None` 和event-unknown/invalid，真实model.encode逐次断言goal三字段及event三字段均零/invalid；两臂共用当前v6 inference包以支持新checkpoint显式policy。两份state_dict键和形状相同，推理超参和stats/几何缓存相同。不同condition policy均在同一unknown输入下工作，未给旧权重额外goal/event。

同R权重、相位机、原生host、安全门、评分器：105个v3 Python文件逐字保持原v11冻结实现，三worker runtime一致；staged_progress.py保留原冻结版本。刷新v6 execution_contract仅多一个未使用的常量，无执行行为改变。原旧runner会传scheduler goal/event，因此**历史v11率不作本轮旧臂**；本轮新旧均按要求固定unknown条件重新执行。

成功仍为600controls末态：有效实测hold≥1s、物体距初态上方8cm目标≤物体尺寸导出的阈值、无物理安全中止。是沿用的8cm HOLD执行任务，不是完整501 RELEASE/placement资格。安全中止按原门：实际受限关节越界余量0.10（平移m、转动rad）；CAD低点低于桌面20mm且中心在footprint内。`contact_not_certified`是时域末失败分类，不是新增即时中止门。

## 成功、安全及死因

| 指标 | 旧11000 | 新1223 |
|---|---:|---:|
| 成功 | 2 | 2 |
| 安全中止 | 14 | 9 |
| 实际controls | 6237 | 8538 |
| WAM调用 | 5126 | 7446 |
| 死因/终局：contact_not_certified | 3 | 8 |
| 死因/终局：contact_without_lift | 1 | 1 |
| 死因/终局：physical_joint_limit | 8 | 6 |
| 死因/终局：physical_table_penetration | 6 | 3 |
| 死因/终局：success | 2 | 2 |

首次触限为最早实际安全门越界状态；同tick多关节并列计，不以索引顺序捏造先后。

| 首次触限关节 | 旧 | 新 |
|---|---:|---:|
| middle_proximal_joint | 1 | 0 |
| pinky_proximal_joint | 3 | 0 |
| ring_proximal_joint | 2 | 0 |
| thumb_intermediate_joint | 3 | 5 |
| thumb_proximal_pitch_joint | 2 | 2 |

## 阶段与末态误差

| stage_achieved累计 | 旧 | 新 |
|---|---:|---:|
| approach | 6 | 4 |
| contact | 4 | 4 |
| lift | 2 | 2 |
| hold | 2 | 2 |
| release | 0 | 0 |

最高阶段分布（互斥）：

| 最高阶段 | 旧 | 新 |
|---|---:|---:|
| none | 14 | 16 |
| approach | 2 | 0 |
| contact | 2 | 2 |
| lift | 0 | 0 |
| hold | 2 | 2 |

`final_goal_error_m`：安全中止例取中止时刻，其余取20s末态，不能把早停误差降低当成功。

| 分位数（m） | 旧 | 新 |
|---|---:|---:|
| min | 0.001215 | 0.000608 |
| p25 | 0.077265 | 0.076353 |
| median | 0.229886 | 0.080864 |
| p75 | 0.882375 | 0.238273 |
| p90 | 1.619204 | 0.949820 |
| max | 2.985536 | 1.782620 |

配对新−旧误差[min,p25,median,p75,max]（m）：`[-2.984927481855743, -0.26401405376028264, -1.0241694992085826e-10, 0.004732132387052501, 0.4413780918970731]`。

控制器曾进入close的回合 **11→4**；阶段认证approach **6→4**、contact/lift/hold **4/2/2→4/2/2**。安全中止少5，同时时域末contact未认证多5，不能把“没有中止”当作完成接近或接触。边际目标误差中位数229.886→80.864mm，而逐reset配对误差差中位数约0；多数未抬升例仍距8cm目标约8cm，不能把更少甩远物体写成任务完成进展。

## 实际消费 τ 的环节

每次完整预测24帧；`run_grasp_geometry_worker.py`的冻结实现仅在approach/close调用τ。前8帧264维作为686维R观测的一部分，R同时输出腕和指的控制意图；并不逐帧跟踪GT轨迹。完整24帧另由`reduce_to_wrist_plan`按两指距CAD≤25mm的首次预测接触截断，未预测到接触时保留全24帧；其末位姿作为grasp候选，在几何可达/压力过滤中使用。真实接触后lift/hold由固定servo和实测反馈驱动。故前8 EPE下降不能直接担保接近、闭合、接触保留和末态到位。

| 实际消费记录（controls） | 旧 | 新 |
|---|---:|---:|
| consumer phase approach | 3015 | 6664 |
| consumer phase close | 2151 | 822 |
| consumer phase lift | 191 | 268 |
| consumer phase hold | 880 | 784 |
| valid τ plan mode approach | 3872 | 6156 |
| valid τ plan mode contact_window | 1236 | 1272 |

这些是新旧各自闭环经过的控制次数，受早停与状态分歧影响，不是同状态预测质量的独立对照。

成功互换的两个例子：

- lightbulb（p008）：旧成功，NEW在close停留561 controls，接触后未认证lift；末态误差1.215→215.337mm。两臂tick3的q/dq/hand/objects逐位相同，此时前8帧τ最大差67.987mm，R动作已经随τ改变；链路确实消费了新预测。
- spheremedium（p017）：旧approach阶段joint abort，新完整成功；误差2985.536→0.608mm。它抵消了lightbulb丢失的成功。两例不足以证明统一的手指或时间机制，当前只定位执行环节，不新增修复/消融。

## 逐reset配对

| reset / object | 旧→新成功 | 旧→新最高阶段 | 旧→新死因 | 旧→新目标误差m | 首次触限tick:DOF（旧 / 新） |
|---|---|---|---|---|---|
| p000 / bowl | 0→0 | contact→none | contact_without_lift→contact_not_certified | 0.191398→0.081279 | — / — |
| p001 / camera | 1→1 | hold→hold | success→success | 0.002020→0.001151 | — / — |
| p002 / cylinderlarge | 0→0 | none→none | physical_table_penetration→physical_table_penetration | 0.864772→0.869393 | — / — |
| p003 / eyeglasses | 0→0 | none→none | physical_table_penetration→physical_table_penetration | 0.268375→0.307080 | — / — |
| p004 / fryingpan | 0→0 | none→none | contact_not_certified→physical_joint_limit | 0.081506→0.081503 | — / 396:16 |
| p005 / hand | 0→0 | none→none | physical_joint_limit→physical_joint_limit | 0.075532→0.075532 | 65:16 / 443:16 |
| p006 / headphones | 0→0 | none→none | contact_not_certified→contact_not_certified | 0.076462→0.076462 | — / — |
| p007 / knife | 0→0 | none→none | physical_joint_limit→contact_not_certified | 0.701434→1.142812 | 118:15 / — |
| p008 / lightbulb | 1→0 | hold→contact | success→contact_without_lift | 0.001215→0.215337 | — / — |
| p009 / mouse | 0→0 | approach→none | physical_joint_limit→contact_not_certified | 1.927887→0.080448 | 148:15 / — |
| p010 / phone | 0→0 | none→none | contact_not_certified→contact_not_certified | 0.076546→0.076546 | — / — |
| p011 / piggybank | 0→0 | none→none | physical_joint_limit→physical_joint_limit | 0.081292→0.081292 | 85:16 / 59:15 |
| p012 / pyramidlarge | 0→0 | approach→contact | physical_joint_limit→physical_joint_limit | 1.584906→1.782620 | 345:8,10,12 / 382:15,16 |
| p013 / pyramidmedium | 0→0 | none→none | physical_table_penetration→contact_not_certified | 1.160076→0.076025 | — / — |
| p014 / pyramidsmall | 0→0 | none→none | physical_joint_limit→contact_not_certified | 0.077504→0.077504 | 119:10,12 / — |
| p015 / scissors | 0→0 | contact→none | physical_joint_limit→contact_not_certified | 0.189575→0.074422 | 193:10 / — |
| p016 / spherelarge | 0→0 | none→none | physical_table_penetration→physical_joint_limit | 0.800838→0.090241 | — / 75:16 |
| p017 / spheremedium | 0→1 | none→hold | physical_joint_limit→success | 2.985536→0.000608 | 358:16 / — |
| p018 / spheresmall | 0→0 | none→none | physical_table_penetration→physical_joint_limit | 0.868730→0.076682 | — / 159:16 |
| p019 / stamp | 0→0 | none→none | physical_table_penetration→physical_table_penetration | 0.923310→0.928377 | — / — |

成功不同的配对，其首次τ差异、同tick实测状态差和R动作变化已保存于summary.json的`first_tau_difference`；这证明预测进入控制路径，不能把它单独当全回合失败的唯一因果解释。

## 验证、环境、资源与复现

20/20对九字段初态逐位相同，同时20/20与原面板保存初态逐位相同；40/40原始安全门逐tick重算、首次中止时刻与control clock核对一致，末态目标误差由原始物体位置重算。实际goal/event valid均0，共OLD5126+NEW7446次WAM调用。独立read-only汇总`independent-pair-audit.json`在成功、安全、阶段、首触限关节和20组原初态上与本报告一致。105个v3源文件与评分器冻结，3份runtime一致；没有覆盖checkpoint、复制geometry/cache或生成视频。

模型：graspenv Python3.8.20 / Torch2.4.1+cu121 / CUDA12.1；native固定venv Python3.8.10 / Torch2.0.1+cu118，实际physics_backend=gpu，30Hz。GPU4/5/6各worker，7未用；0–3未使用。真实代码启动commit98d115327215a5922028861c25174b579ac4b55b；各manifest保留当时HEAD，最终交付commit见Git。

goal合同6项、冻结几何执行/运行快照11项，共17 passed。整库`tools/verify.py --changed`仍被既有P-20261023-tau-fixed-panel-metric卡缺claim_id/hypothesis_family/probe_index_in_family阻断，未为本实验修改无关卡。

运行产物：`outputs/wam-rl-policy-v3/tau-checkpoint-ab-20261023-r1/`。`registration.json`保留原20例路径、seed、交替顺序和预算；`worker-*.diff`给出相对冻结原worker的全部改动；`summary.json`为本报告机器表；`runtime-parity.json`、`checkpoint-environment.json`、`resource-monitor.jsonl`及各worker raw/initial/native-audit可追溯。

```bash
/home2/wyy/miniconda3/envs/graspenv/bin/python src/task/wam-rl-policy-v3/tools/run/run_tau_checkpoint_ab.py \
  --registration outputs/wam-rl-policy-v3/tau-checkpoint-ab-20261023-r1/registration.json \
  --worker 0 --gpu 4 --output outputs/wam-rl-policy-v3/<fresh-run>/g4
# worker1/GPU5 与 worker2/GPU6 同理；使用新输出目录
/home2/wyy/miniconda3/envs/graspenv/bin/python src/task/wam-rl-policy-v3/tools/audit/summarize_tau_checkpoint_ab.py \
  --run outputs/wam-rl-policy-v3/tau-checkpoint-ab-20261023-r1
```

局限：固定小样本、已暴露物体、无seed重复；所存九字段初态不能认证PhysX隐藏求解器缓存。死亡标签是原认证/几何安全口径，不是新接触对真值。本轮无训练、无第三臂、无新成分分解；不把局部结果升级为τ路线或所有可能执行器的否决。

实际新增普通文件约150MiB，14,775 controls、0updates，最长worker1458.819s（24.31min，整批约25min）；资源采样最小可用空间127.67GiB，显存峰值GPU4/5/6分别5322/5320/5314MiB。全部本轮worker与40个native自有PID已退出；未停止或改动其他作业。资源原始记录见completion-resources.json与resource-monitor.jsonl。
