# 已确认缺陷修复与危害量化（2026-10-22）

本轮依据机主12:3x指令修复；没有重启flowmatching、P1或新增方法实验。旧结果/冻结源码/
checkpoint保持原样。上一轮未提交的审计、范围修订及本轮修复一并交付，保留无关用户修改。

## A1. UNKNOWN：选择显式拒绝，不重标终态

`wam_rl_policy_v3/task_admission.py`定义唯一`reject-unknown.v1`准入策略。实际打开task.json，
记录每个被拒绝sequence、terminal_mode、reason、task_path和准入前后分母；不改task.json。
MeasuredOfficialEnv与native host使用同一策略；native固定motion选项也不能绕过它。
`TaskSpec.require_success_path()`由TaskScheduler构造函数在任何transition之前调用，
UNKNOWN立即报错；仍可解析/检查UNKNOWN标注，不修改原始语义。

实际501 eligible → **437 admitted /64 rejected**，策略与native选集完全一致，清单在
`outputs/ptv3-shared-wam-v6/negative-result-fixes-20261022-r1/task-admission.json`。
两端未来运行均写task-admission.json。所有任务都UNKNOWN则显式setup failure，不开空训练。
resume identity新增准入policy，旧501分布checkpoint不能静默冒充437分布的同一次实验恢复。
继承BC权重依旧是旧训练所得，本轮没有重训或宣称洗掉历史BC数据。

回归：`test_executable_task_admission.py`6项，覆盖真实Env构造的拒绝与计数、原标签bytes不变、
全UNKNOWN失败、native/policy同集及握手拒绝旧采样集合、旧resume identity拒绝；加现有HOLD/RELEASE正控制。
修前3个最小真实入口测试均失败，修后通过。fixture已排除几何自身报错造成的假red。

## A2. v6/v4：报告先行，官方入口未替换

官方PPO继续V4Runtime，显式记录v4/[24,2,11,3]/v6 UNWIRED。v6→11映射、native排列、
30Hz、goal/event注入及缺失观测mask见[接线可行性报告](v6-ppo-wiring-feasibility-20261022.md)。
本轮1个真实H4的CPU接口smoke到PD边界PASS，使用随机小v6+geometry seam与真实BC；
没有把它伪报为GPU/物理/PPO完整迁移。既有P0真实checkpoint物理轨迹另列，不倒灌到官方入口。
回归2项：v4 loader拒绝v6 checkpoint、真实online builder映射/goal/event/mask往返。

## A3. voxel：诊断加强，原gate与网格完全保留

`negative_result_invariants.voxel_first_failure`新增is-hand对应identity、slot/joint语义、
world/anchor位置、与物体中心/最近目标surface/同slot wrist及指尖的关系、越界轴/余量/index。
回放4768个既有P0状态：两首点均right wrist，距中心2.26087/2.20059m，最近surface
2.21377/2.09807m，支持“已经远离目标”的解释，不能称这两例健康抓取误杀。
历史27792条终止点未保存，不拿P0两例代替其分布。

同轨迹“首次H4准入检查”与“每tick终止”的观察性比较：4800 vs4744 controls（差56），
主安全prefix2316 vs2316（差0），完整任务均0/16。未新跑仿真；原物理成本仍4800，
未来动作/采样/reset在去gate后会变化，不能声称省56步或增加成功率。9项相关回归通过。
详细数字、反事实边界及命令见[危害表](negative-result-harm-20261022.md)。

## A4. scale：不存在的请求必须明确拒绝

`wam_v4_runtime.load_geometry`找不到请求scale或请求非有限时ValueError，不能取rows[0]
再用请求scale命名key。最小回归使用一个有效512点/32query几何缓存，请求1.234567而仅有1，
必须在读取错误scale几何前拒绝；已有合法scale正控制仍通过。测试在
`test_negative_result_invariants.py`前两项。保存的历史r3请求固定scale1，50个物体当前保留
cache均有匹配（0/50缺失），没有证据它解释了该历史运行零成功；全仓错误请求日志缺失。

## 日志缺项也已修：保留reward，不改变奖励函数

MeasuredOfficialEnv累计记录实际transition数、六个reward分项（含外部terminal修正）、
三个阶段的controls、结束回合总controls/return；每个episode在reset前携带sequence、
terminal_mode、failure_reason、controls、return与分项。runner按rank写episodes.jsonl，
**8MiB/rank硬上限**；超限显式计dropped，完整aggregate继续累计。每次log对统计做DDP sum，
与旧全局计数一样是三个rank重复一个global值，不相加；finally保留各rank最终local成本。

`test_measured_episode_logging.py`用真实Env step/reset CPU replay验证2env×5步=10交互，
phase/交互总数一致、reward分项sum等于实际返回reward、结束controls6、局部reset清理与
未reset状态保留、文件cap/拒绝覆盖。该回归修前因缺training_statistics失败，修后通过。
这是观测性修复，**没有改reward数值、加奖励或改变gate**，旧return无法回补。

## B. 影响范围与预算

详细集合界/证明/假设见[量化危害](negative-result-harm-20261022.md)。这里统一分母：
iteration1570是27792/53174=**52.2661%**；最终1575是27869/53347=**52.2410%**。
旧r3这段实际完成1344次更新，预算**16515072**环境control transitions；native实际
**16527360**，另12288采集未更新。53,347是结束训练回合，不是独立测试样本。

| 缺陷 | 已核受影响数量/占比 | 具体被污染结论/卡片 | 应如何处置 |
|---|---|---|---|
| v4被当v6 | 53347/53347结束回合及16515072已更新交互均属于v4链；这是归因污染，不是1651万步全无用 | P-20261020-inspire-measured-official-ppo；今天P1的历史依据 | 撤销“v6 PPO失败”；保留v4训练未观测完整成功；P1仍HOLD |
| UNKNOWN无成功路径 |64/501任务12.774%；9314/56449初始化16.500%；安全prefix候选2150/56418=3.81084%；ended集合界6212–9314 | P-20261020-inspire-measured-official-ppo | 撤销所有任务均有成功机会的假设；降级总体0为描述；不认UNKNOWN解释全部0 |
| voxel每tick终止被误称机械越界 |最终27869/53347=52.241%；1570数字另报；缺历史point/时长 | 同PPO卡；P0 executor-qualification仅其自身2/16另列 | 撤销机械越界解释和自动误杀说；保留真实表示域终止频率 |
| return/phase未保存 |3rank×1344=4032条curve记录均无reward/return/phase字段，53347结束episode缺逐例日志 | 同PPO卡 | 撤销“PPO没收到reward”；旧精确预算/分项无法恢复，今后增加有界日志 |
| scale静默fallback |该历史r3保留cache的50物体均有请求scale1匹配，0/50缺失；全仓错误请求量未知 | 同PPO卡及未来所有load_geometry调用 | 已修潜在bug，但不能归因历史全0；加回归与影响面注释 |
| raw SIGINT未final-save |至少380updates/109440样本曝光、约24.8min尾段未持久化；非环境交互 | P-20261022-wam-v6-flowmatching | 保留11000真实权重与11380日志；不把trial/best指针当latest，不伪报final保存 |

“实际吃掉多少预算”不能拿回合比例相乘：在冻结链及每结束回合3–1800controls的保守界下，
UNKNOWN已结束回合内部控制步**18636–16382973**；voxel **83607–16438638**；
并集 **83607–16466580**。范围很宽就是缺逐episode长度/sequence/重叠日志的实际限制。
它们不是可避免浪费或有用梯度损失的估计；UNKNOWN局部reward仍可能提供学习信号，
voxel也可能保护真实失稳。删除gate/换任务会改变后续分布，因果收益无法从旧产物识别。

若只从已发生记录中排除UNKNOWN与voxel，保留观测是 **0/N，N∈[16164,25478]**；
只排除voxel则0/25478；只排UNKNOWN则0/N，N∈[44033,47135]。UNKNOWN集合界来自
每ended对应唯一初始化的冻结链，不是16.5%频率推算。不能把这些写成“修复后仍零成功”，
更不能给v6或独立成功率置信界。旧未经修复模型身份/实现/协议的18张卡修订也一并纳入本次交付。

## C. 长训停止与实际权重

[独立停止/权重/曲线报告](flowmatching-stop-checkpoint-audit-20261022.md)给出15目录项/
9个唯一CPU payload、11个目标PID均退出、GPU4/5/6各6MiB/0%、信号日志及完整CSV/图。
主训练实际保存**10500/11000**，另有诊断保留copy8500；trial20/128、strict9/10是旧阶段。
**11020有日志没有对应权重；最后确认日志11380，最新实际权重11000**，至少380更新未保存。
training/best仍指trial20，trajectory-best仍指strict9，是selection身份，不是latest。
Hermes停止操作由机主转达；日志独立确认12:29:11 SIGINT，不能证明是谁发信号。

机主以step11020观察点及后期收益趋平决定主动停止。实际9000@8.711h loss.689010→
11000@10.713h loss.661275；各中心±15min窗口总loss均值下降3.96%，contact仅0.73%、
geometry_contact2.48%，固定DEV没有持续改善。因此写“后期接触收益弱”，不伪称严格恒定。
raw SIGINT绕过final-save/终态写入，原process FAILED和progress RUNNING保留原样，
另用stop receipt说明已停。没有重启训练、改选择指标、删除checkpoint或为报告另开GPU任务。

## D. 验证、交付及下一步

最终针对回归 **56 passed in6.39s**；19项不变量正控制和19个故意错误检测PASS，另有voxel
边界/协议/预算集合界回归。1个真实H4的CPU→PD接口smoke PASS。53个owned文件的Python
语法/结构/Markdown链接检查、git diff --check、INDEX一致检查PASS；P1继续HOLD。
整库verify的6张既有v2卡metadata缺口仍单列，不冒充全部通过。
本轮允许git/network，上一轮只读阻断已解除，将只提交我们拥有的审计/代码/STATE/卡片，
不带入无关docs/PROJECT_MEMORY.md。提交/远端SHA在实际push后核对并回报，不能预填成功。

**第一个应跑的实验**：两环境各一个明确HOLD/RELEASE的原动作正控制，固定初态、有真实
终态≥1秒证据，300controls/环境上限（600交互）；同时记录实际q/dq和第2/4步indexed reset，
独立评分与TaskController逐帧核对。若正控制不能通过，即证伪“当前任务/执行链已具备RL
成功机会”，不进入三臂或生成器比较。原动作正控通过之后，才在同执行器/同初态加入GTτ；
先区分任务/动作/reset问题，再谈预测或样本效率。此为下一步建议，本轮没有执行新物理。

复核入口（均CPU，产物均在项目outputs，临时文件在项目tmp）：

```sh
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$PWD/tmp/negative-result-fixes-20261022-r1"
export PYTHONPATH="$PWD/src/task/wam-rl-policy-v3/src:$PWD/src/task/trajectory-policy/src:$PWD/src/task/ptv3-shared-wam-v6/src"
PY=/home2/wyy/miniconda3/envs/graspenv/bin/python
"$PY" -m pytest -q -p no:cacheprovider --basetemp="$TMPDIR/review" \
  src/task/wam-rl-policy-v3/tests/test_executable_task_admission.py \
  src/task/wam-rl-policy-v3/tests/test_measured_episode_logging.py \
  src/task/ptv3-shared-wam-v6/tests/test_negative_result_invariants.py \
  src/task/ptv3-shared-wam-v6/tests/test_ppo_model_identity.py \
  src/task/ptv3-shared-wam-v6/tests/test_voxel_harm.py
"$PY" src/task/ptv3-shared-wam-v6/tools/audit/audit_harm_budget_cpu.py
```

完整56项回归的路径、结果、smoke和改动清单：
`outputs/ptv3-shared-wam-v6/negative-result-fixes-20261022-r1/validation.json`。
旧日志覆盖与scale绑定补证分别为同目录old-log-coverage.json、old-scale-bindings.json。
