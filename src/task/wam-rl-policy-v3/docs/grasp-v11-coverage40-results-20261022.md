# v10完成、v11三处修复与每链n40新物体覆盖

## 结果解读表

| 已完成结果 | 结论 | 下一步决定 |
|---|---|---|
| v10旧seen：同R WAM35/40 vs零τ0/40，p=2.27289e-17 | 旧15mm窄任务达到观察目标≥80%，同链τ内容有局部贡献 | 保留v10成果，不称预训练收益或新控制器泛化 |
| v11接口未见：A1 3/40 vs A2 0/40，p=.240506，CI重叠 | 新几何/SE3接口未通过预注册贡献判据，也未达到工程成功率目标 | 不推广为已合格executor；三臂RL继续HOLD，不重训v6 |
| 同后段伺服A0 14/40，12个同初态A0成功而A1失败 | 至少这些初态有另一执行链的物理成功见证，不能把全体失败归为任务不可能 | 优先修接近/闭合消费与受压卸载；A0不是GTτ→R上界 |
| 最终PD目标准入非法0，但A1实测限位14/40，13次在接近/闭合 | 静态目标准入没有解决动态接触压力 | 不放宽安全gate，不把软件PASS当物理正确 |

**本轮预注册120回合全部结束并通过raw复核**，2026-10-22注册、10-23完成。
20个此前未在该接口调过的对象×2不同原始序列；不是WAM/BC预训练未见。
没有按source成功筛选、没有失败换样本、没有训练更新、没有完整RELEASE资格。
当前main已有并行v13机制/GT验证；本线程不重复启动其工作。

## 成功、阶段与统计

| 新接口未见，均n40 | A1 原R+WAMτ | A2 同R，仅零τ | A0 source-PD接近→同SE3 |
|---|---:|---:|---:|
| 严格末态成功 |3/40 (7.5%)|0/40|14/40 (35%)|
| Wilson95 |[2.584%,19.864%]|[0%,8.762%]|[22.135%,50.494%]|
| ever精确持握 / final精确持握 |3 / 3|0 / 0|15 / 14|
| 末态持握，不要求精确到位 |3|0|15|
| 曾达approach→contact→lift→hold |14→12→3→3|8→4→1→0|31→31→19→17|
| 末态approach→contact→lift→hold |3→3→3→3|0→0→0→0|16→16→16→15|
| 新几何阶段 + ≤15mm终点的完整成功（次要） |3/40|0/40|14/40|
| 仅≤15mm终点、不要求持握（不是成功率） |5/40|0/40|17/40|
| 完整20s且无安全中止的生存者 |20|31|26|

双侧Fisher：A1/A2 **.2405063291**，A1/A0 **.0051879720**，A2/A0 **3.0775315e-5**。
配对补充McNemar依次 **.25 / .00341796875 / .0001220703125**。
A1/A2预注册规则要求p<.05且A1 Wilson下界>A2上界，本次两项都未通过。
A0对A1有更高观察成功率，但A0包含source控制信息、执行链不同，不能据此评价WAM参考质量。
20对象内两序列相关，trial级Wilson/Fisher按机主要求报告，仅描述固定面板，不冒充40独立对象总体推断。
HOLD不要求release，本卡release均N/A。旧seen不并入新物体子集。

判据冻结：env1、30Hz、600controls/20s，不成功即停；真实接触耦合、离桌lift≥5cm、连续hold≥1s，
终点位置误差≤最小凸包宽度w的.10且无物理中止。该批位置容差范围**1.485–12.000mm**。
此新协议不能回改v9/v10的15mm分母；次要15mm列仍使用新几何阶段，不能称旧协议重评分。

## 互斥死因与误差分布

| 死因，均n40且各列合计40 | A1 | A2 | A0 |
|---|---:|---:|---:|
| 成功 |3|0|14|
| 实测关节限位中止 |14|6|12|
| 原物理桌面gate中止 |6|3|2|
| 未接触 |15|30|4|
| 接触未抬升 |2|1|4|
| 抬升但连续持握不足 |0|0|2|
| 曾持握后丢失 |0|0|1|
| 末态持握但未到位 |0|0|1|

A1限位相位为末次施加控制时approach8、close5、lift1；关节事件为
`thumb_intermediate_joint`11、`thumb_proximal_pitch_joint`5，共16关节事件/14回合。
A2相位approach3/close2/lift1；A0 approach1/close11。多个关节可以在同一回合触发。
这些是原gate标签，桌面标签仍是CAD/有限桌面几何代理，不把它等同经独立碰撞求解器确认的穿透。

| 分布（每臂40，含提前中止终点） | A1 | A2 | A0 |
|---|---:|---:|---:|
| 最长持握秒 min/median/p90/max |0 / 0 / 0 / 17.300|0 / 0 / 0 / .133|0 / 0 / 18.283 / 18.800|
| 终点XYZ误差mm median/p90 |157.261 / 1166.783|78.448 / 1020.298|68.776 / 302.269|
| 仅完整20s生存者误差mm median/p90 |82.156 / 1039.190 (n20)|76.837 / 99.203 (n31)|3.239 / 715.966 (n26)|

全40值与六分位数、误差/w、每例raw路径见`summary.json`和`details.json`。
A1末态持握3例全部精确到位；把终点门放到15mm也仍只有3例完整成功，不能只归因于位置门太严。
A0的14个成功回合均没有source动作表达失败；12个A0成功/A1失败的配对初态可作为后续修复的物理见证。
它们在本仿真协议下成功的后执行机器人状态轨迹可供同τ通道参考构建，**不等于已验证GTτ→R上界**。

## 控制、预算与无法自证处

| 控制统计 | A1 | A2 | A0 |
|---|---:|---:|---:|
| 环境controls / R调用 |15196 / 15196|20898 / 20898|16745 / 16745|
| R/source上游归一化超界 |439 (2.89%)|1854 (8.87%)|0|
| 最终np.clip改变命令 |0|0|0|
| 几何候选/过滤路径活跃controls |8057|16413|557|
| 最终18个PD目标非法 |0|0|0|
| SE3平移增量在速率上限边界 |177|10|1022|
| SE3转动增量在速率上限边界 |0|0|3|
| 命令增量速率越界 |0|0|0|

上游超界定义为意图提取之前的R/source规范动作超出[-1,1]；最终裁剪按raw final action与clip差>1e-6计。
R调用是前向计数：A0接近阶段以source替换其输出，不能用该计数把A0说成同R消费τ。
**边界次数不是精确裁剪次数**；最终0裁剪不代表没有候选替换、限速或原生目标投影。
没有保存投影前完整命令，精确原生投影次数读不到，`details.json`显式null；不填0。
旧`goal_servo_saturated_controls`的分量阈值不用于度量新SE3范数限速。
A0的source表达被拒43controls、涉及3回合，回退同R并完整保留；14个A0成功均不属于这些回合。
即使source被接纳，它仍经过共同控制器，A0不是无改动source命令回放。

实际**52839controls**，相对上限72000为73.4%；0中断补跑、0训练更新。
三worker实际墙时3904.3/3680.7/3535.5秒，合计约3.09 GPU·h；只用4/5/6。
为避免空等，在已有handoff转为只用4后先用5/6、再加入4；原排队记录保留，算法/种子/顺序不变。
完成后没有停止任何并行v13进程；GPU空闲与否请重新查nvidia-smi。

独立核到：40/40组三链九字段初态一致；40/40 A2保留同R、τ全零、24帧未来内容被移除；
120/120 raw逐帧重评分一致，三个worker的标签翻转负控制均被拒绝；details限速/原生目标准入检查PASS。
全批实际PD重建最大误差**3.54983e-7**，不把PD目标一致冒充真实关节状态无接触偏差。
当前完整v3测试**269 passed /0 failed /0 skipped，20.36s**，日志`unit-tests.log`；
此前257项日志对应较早源码，后续并行会话增加测试，不能把不同清单报成同一测试总数。
注册器错误注入killed2/2；快照故意漏source_pd被拒绝；限速检查能抓住40mm/.10rad故意越界。
软件PASS没有推翻A1物理限位14/40这一失败。

不能自证：canonical512点对真实接触几何的充分性、力来自物体还是桌面、完整穿透/力闭合真值、
GTτ同通道上界、WAM预训练独有收益。原始轨迹重评分仍共用冻结S2c/observer，非独立接触真值。
Hermes与GPT独立物理复跑结果仍待裁决，不能称三方一致。

## 三处diff、来源与下一步

代码位置、回归与数值期望见[本线程独立复核](grasp-v11-independent-findings-20261022.md)末节；
`grasp_geometry.py:20`尺寸合同，`:137`live SE3/限速/抗饱和，`:54/:96/:170`18关节候选准入；
快照依赖为`tools/audit/freeze_grasp_geometry.py:8`。算法实现包含并行会话工作，本线程独立发现并复核
20mm滑移反例、快照缺依赖，以及本批n40物理结果，不把共同作者实现算成独立物理复跑。

Hermes提出三处修复；GPT反对把尺寸化/姿态项/seen80%当统一解法，主张扩覆盖和有界残差。
本结果支持“不把静态准入当动态正确”，不支持“这三处已经解决泛化”。
v6当前物体坐标与goal_valid全True已核正确；训练t+24实际终点 vs 在线调度waypoint的语义/时间差仍保留，未靠GT未来补条件。
BC应学习漂移状态意图恢复；并行小试验BC DEV MSE .3864→.4604，PPO前后0/2只算工程运行，不作方法负结论。
PPO最新准入为干净消融、阶段reward、预注册预算/对照，**不恢复seen80%通用门槛**；三臂样本效率RL仍HOLD。

我的下一修复优先级：接近/闭合下的实测受压卸载。冻结`pose_consumer.py:234`仍直接用R preshape，
force调节在`:237`后只用于lift/hold；A1有13次限位发生在进入该调节分支之前。
应把q/dq/力驱动的逐指卸载/重试接进接近与闭合，保留已能精确到位的后段，不先加训练步数。
这是假设明确的修复入口，不宣称它是唯一根因；用12个有物理见证的配对失败开发，再按n40冻结复验。
下一修复的预期验收为A1实测限位14/40→≤4/40，同时不能把中止转成更多桌面gate失败，原3成功不得丢失；
这是后续目标，尚未实施/取得，不承诺一次修掉另15例未接触，不超本卡预算启动新批次。
并行[v13](grasp-v13-mechanism-20261022.md)已在做组件/GT链验证，应复用本批成功轨迹，不另起重复归因实验。
本卡在预算完成处收口，**未见物体高成功率仍未解决**。

## 精确复现

前置：同服务器仓库、原数据/几何缓存/原R、step011000 WAM、该run的只读冻结源码与references。
不需要新训练。第一次运行的source快照为`0e8c8214`，legacy基线/补丁另见各worker snapshot.json。
不要用当前main新写法替代冻结consumer重现本批。

```sh
PY=/home2/wyy/miniconda3/envs/graspenv/bin/python
TOOLS=src/task/wam-rl-policy-v3/tools/run
# 全新输出，复用只读注册/冻结worker/参考；不重做选样，不改模型。
$PY - <<'PY'
from pathlib import Path
import shutil
src=Path('outputs/wam-rl-policy-v3/grasp-v11-coverage40-20261022-r1')
dst=Path('outputs/wam-rl-policy-v3/grasp-v11-coverage40-hermes-repeat1')
dst.mkdir()  # 存在则报错，不覆盖
shutil.copyfile(src/'registration.json',dst/'registration.json')
PY
nvidia-smi
$PY $TOOLS/run_grasp_coverage40.py --run outputs/wam-rl-policy-v3/grasp-v11-coverage40-hermes-repeat1 --detached
# 期望120回合各臂40，40初态/40A2输入检查、120raw replay通过；成功数由真实运行给出，不预填。

# 原产物只读统计复核：
$PY $TOOLS/summarize_coverage40_details.py --run outputs/wam-rl-policy-v3/grasp-v11-coverage40-20261022-r1 --output outputs/wam-rl-policy-v3/grasp-v11-coverage40-20261022-r1/details-hermes.json
# 期望PASS；A1/A2/A0=3/0/14，controls=15196/20898/16745，native target/rate violation均0。
$PY -m pytest -q src/task/wam-rl-policy-v3/tests
# 当前完整清单269passed/0failed；清单继续变动时须报实际全量。
```

环境：graspenv Python3.8.20/Torch2.4.1+cu121；native dexplore_v120_train Python3.8.10/Torch2.0.1+cu118/IsaacGym。
全部产物：`outputs/wam-rl-policy-v3/grasp-v11-coverage40-20261022-r1/`，核心为
`registration.json`、`supervisor.json`、`summary.json`、`details.json`、`initial-parity.json`、`ablation.json`、
`g4-rescore.json`/`g5-rescore.json`/`g6-rescore.json`及每回合raw目录。
