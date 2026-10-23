| 观测 | 含义 | 决定 |
|---|---|---|
| **加载稳态位置 10–17mm → ≤5mm**，且动态 P95 同步改善 | 稳态校正有效 | 固化进执行器主链，重跑 brief48 网格看**可行域外扩多少** |
| 稳态改善但动态 P95 不动 | 只修了稳态项 | 再谈带宽（更高刚度/jerk 受限参考） |
| 两者都不动 | 误差源不在 TRACK 环内 | 回去查参考映射与加载传递（含 T_hand_object 估计） |
| 出现新的安全中止/碰撞 | 校正引入风险 | 立即回滚该改动，保证据 |

# 原 TRACK：实测物体 SE3 稳态校正

预注册：P-20261023-object-se3-steady-correction；main；Decision，区分有界误差积分能否消除接触负载静态地板及改善动态误差。物理结果待测，不能以单元测试代替。本轮只做此修复，不训练、不用τ、不改抓取/相位机/PD/FF/物理门。

引用v53：同端点配对加载位置P95为12.992/16.976/16.662mm，卸载0.0215/0.0222/0.0221mm；旧B矩阵无/有FF P95中位33.267/26.843mm，掌部38.350/31.240mm，漂移约0.615mm。它们不是本轮新物理结果，亦不能外推所有物体。没有证据完全否定所有滑移/坐标问题，只排除了记录覆盖下的主导机制。

## 实施前冻结：同端点、同时间律、单一因素

物体平移误差为世界系 `p_ref-p_meas`；姿态误差为世界系 `Log(R_ref R_meas^T)`（SO(3)最短测地旋转向量）。积分修正对象参考后，仍经同一个 `hand_setpoint` 映射消费，保留原FF及原PD。位置/姿态分别增益，三轴等方性，积分范数上限40mm/0.14rad，更新速率上限15mm/s与0.08rad/s，积分增益1/s与0.7/s；不是按方向或物体调参。每次新TRACK构造器清零；无有效实测接触、有限关节内裕量不足20mm/20mrad、目标/命令约束不允许时停止并清除该修正，不让历史积分持续朝安全边界推。积分候选导致步幅/动作饱和则拒绝该次增长；原参考及FF不被替换。

物理验证按顺序：①原冻结10秒常值参考，B、FF on（导数零），3次，每次原获取+300 TRACK，尾3秒为稳态；对照只读v53配对B三段。②若无新增安全/碰撞，再用旧r4中100mm直线、30/100/250mm/s的原始reference，FF off/on各一次（6格），不改时间、不新采参考，取得同一0/30/100/250速度扫描的截距/斜率/R²。所有新初始q/dq/object/palm/T_HO与各旧对应格逐位核对，否则标不完全配对。总预算≤6000controls、≤1800秒、≤1GiB，空间<20GiB停；每次只用实测空闲的4/5/6/7。出现相对旧对应格新增安全中止/碰撞立即停止后续新物理任务，默认入口仍off，撤回启用，不调门/PD救数。

成功沿用物体P95≤15mm、SO3≤5°、漂移≤10mm/10°、原物理门与名义限位20mm/20mrad；静态≤5mm是本轮额外工程门，不能替换全段成功指标。分报真越界、名义内≤0.10贴边、命令越界实测未到，连续腕角不纳入有限关节余量。保留原自碰撞过滤关闭的缺口。仅静态门与动态改善同时通过才考虑默认启用和brief48扩网格，未经验证不默认启用。

产物：`outputs/wam-rl-policy-v3/object-se3-steady-correction-20261023-r1/`；参考只读v53 `static-reference.npz`、旧r4 `case012/014/016`（FF off）及`case013/015/017`（FF on）的原参考。启动前再次核对索引元数据。物理验证待空闲卡；不抢占已有进程。

## 已实现的 diff 与期望变化

- `src/wam_rl_policy_v3/object_tracking.py:87`：`ObjectSteadyCorrection` 用上述 SE3 误差、等方范数投影和速率限制积累偏置；`:137` 原 TRACK 构造器显式选择校正，默认关闭；`:174` 以实测对象驱动，并检查接触、当前/0.10s外推有限关节边界、参考可达性与增量PD。外推只撤去校正，**不复用上一轮失败的预测制动去改原动作**。期望加载静态位置12.992–16.976mm→≤5mm、姿态2.410–2.987°→≤2°，目前是验收目标。
- `object_tracking.py:189`：FF保留原 `velocity/30`、角导数/30及原PD增益；经修正参考的刚体杠杆重新求导，积分更新本身不作为额外FF。有限差分测试验证旋转偏置后的杠杆速度，防止保留FF时引入新的映射误差。`:221` 后的25mm/0.12rad步幅cap与手指力规则不变。
- `tools/run/run_object_tracking.py:44`：`--object-steady-correction`；构造/step输入实测物体、原物体参考、固定T_HO；原评分仍用未校正参考。新校正臂首次物理限位/碰撞即停止后续case，入口默认off，方便撤回启用；不改抓取段。期望新增安全中止/碰撞=0、实际命令越界=0，均待物理验证。
- `tools/run/run_object_steady_probe.py`：冻结3静态+6动态的顺序、预算及空闲卡等待；只选4/5/6/7中≤64MiB且利用率≤5%的卡，不抢占、不重试失败仿真。`tools/audit/audit_object_steady_probe.py`：独立对原参考评分、重放修正参考映射/PD/FF，汇总静态/动态分解及三类限位口径；未获得新物理数据前不输出收益数字。

## 工程验证与不能替代的物理证据

代码冻结提交 `5c9f01f6d0af8d11397fbf3d1b1dc655edd78dac`。首次完整清单 **49 passed / 0 failed（1.75s）**：旧三文件32项+新17项；5种故意错误（漏反馈、逐轴限幅替代范数、饱和时不回退积分、接触无效仍残留积分、SO3误差用body坐标）各被其见证测试抓住，5/5。关闭校正时，与`01ce376a`冻结旧实现200组FF开/关命令**逐位一致**；手指输出、原评分参考不被修改。该一致性只证明所测输入的关闭路径，没有证明启用后的接触动力学。

CPU旧状态上的候选门重放（不是闭环、不是新仿真）：9段1184个控制输入，积分候选接受1165，饱和拒绝19；没有报告任何误差改善。这只检查候选能否通过防线，不预测物理反馈。3段静态各300步中4拒绝/296接受，候选偏置最终达到40mm/0.14rad上限，是在**旧误差保持不变**的反事实重放下的必然积累，不能解释成修后会触顶或已治好稳态。初始PD馈入、原状态与旧产物全部只读，详见 `frozen-state-guard-replay.json`；contact有效性只取旧记录contact布尔值，不冒充缺失字段的重新测量。

运行环境：CPU工程测试为 `/home2/wyy/miniconda3/envs/graspenv/bin/python`；正式物理host继续 `dexplore_v120_train`，种子195、30Hz，同source-PD预持握。写盘前空闲70,817,034,240B（约65.95GiB），预算≤1GiB、不删除历史产物。物理尚未启动时，训练更新=0、新物理controls=0；这些零不表示失败率。

## 可复现命令与预期输出

```bash
PYTHONPATH=src/task/wam-rl-policy-v3/src /home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/wam-rl-policy-v3/tests/test_object_tracking.py src/task/wam-rl-policy-v3/tests/test_reference_admission.py src/task/wam-rl-policy-v3/tests/test_tracking_error_decomposition.py src/task/wam-rl-policy-v3/tests/test_object_steady_correction.py --basetemp outputs/wam-rl-policy-v3/object-se3-steady-correction-20261023-r1/review-tests
```

预期：最终50 passed、0 failed，其中5个故意错误测试被捕获；不启动GPU。

```bash
/home2/wyy/miniconda3/envs/graspenv/bin/python -u src/task/wam-rl-policy-v3/tools/run/run_object_steady_probe.py --output outputs/wam-rl-policy-v3/object-se3-steady-correction-20261023-r1/probe --wait-seconds 600
```

本轮实际由同一命令将stdout/stderr保存为 `resource-and-probe.log`；同目录执行凭证已存在时明确拒绝覆盖。独立复跑请换新的`outputs/wam-rl-policy-v3/...`目录。空闲卡到位才启动，预期COMPLETED、静态3段+动态6格；否则明确BLOCKED_GPU（0新物理），不自动换到占用卡。完成后离线命令：

```bash
/home2/wyy/miniconda3/envs/graspenv/bin/python src/task/wam-rl-policy-v3/tools/audit/audit_object_steady_probe.py --run outputs/wam-rl-policy-v3/object-se3-steady-correction-20261023-r1/probe
```

预期生成analysis.json：原参考评分一致、目标映射差<1e-10、PD增量差<2e-6m、FF导数差<2e-5m/s；它同时保留各初态差异和所有失败。收益门仍须实测静态≤5mm、动态P95改善、无新增安全/碰撞才可判第一行，不满足不得默认启用或宣称brief48可行域已外扩。

## 本轮收尾：BLOCKED_GPU，不能判定物理收益

实际等待600.170秒，11次约60秒间隔设备查询，4/5/6/7始终不空闲。末次显存21791/22883/22217/9253MiB、利用率均100%；没有创建物理host、没有占用任何GPU、没有干预已有进程。`probe/execution.json` 状态 **BLOCKED_GPU**、`commands=[]`；等待进程正常退出，0训练更新、0新物理controls。没有在后台遗留自动启动任务。

| 指标/样本范围 | 修前（冻结实测，只读引用） | 本轮修后 | 当前结论 |
|---|---|---|---|
| 同端点3段加载稳态物体位置P95 | 12.992/16.976/16.662mm | 未取证，目标每段≤5mm | 不判定改善 |
| 同端点姿态P95 | 2.987/2.410/2.795° | 未取证，目标≤2° | 不判定改善 |
| 100mm直线0/30/100/250mm/s拟合，FF off | 截距9.936mm、斜率196.542mm/(m/s)、R²0.969918（v53同初态敏感性） | 未取证 | 不判定稳态/带宽贡献变化 |
| 同上，FF on | 截距9.477mm、斜率148.129mm/(m/s)、R²0.971039 | 未取证 | 不能把旧FF改善当新积分收益 |
| 原B动态45格/FF状态 | P95中位33.267→26.843mm | 未运行完整网格 | 可行域外扩未知 |
| 新臂有限关节最小余量、真越界/贴边/命令越界实测未到、碰撞 | 各旧trace保留，未改门 | 新臂0段，未取证 | 不把“未跑0次”当安全通过 |

结算：**实现完成，物理验收未完成；未命中预注册四行中的任何实测结论。** 保持显式开关、默认off，未固化为主链默认行为，也未扩brief48矩阵。唯一卡点是没有空闲GPU，不是实验失败；无需修改阈值/增益，也不需要新训练。下一步只需任一4/5/6/7释放后，以同一固定协议的新输出目录运行；例如把上述`--output`改成 `outputs/wam-rl-policy-v3/object-se3-steady-correction-20261023-r1/probe-r2`，保留本轮BLOCKED证据。无需重新申请方法授权。

最后补齐非共轴旋转FF有限差分见证，以及有限关节实测外向速度在进入内边界前清除积分的回归。最终精确清单 **50 passed / 0 failed（1.87s）**，新18项、原32项；故意错误仍5/5。`tests-final.log`保存全量计数。新评分器还在只读旧case012的独立fixture上复核通过：原评分、映射、PD与FF重放一致；fixture只新增“校正关闭”的诊断metadata，不冒充新控制器rollout，`audit-fixture/selfcheck.json`明确标记0新物理。

新增实体产物（不计只读历史软链接、台账自身）：7文件/118,509B；收尾可用盘70,787,371,008B，无历史删除。`closeout.json`保存运行状态和空间账。源码、测试与本文才入库，outputs不入库。首次`git pull --rebase`被共享工作区其它脏文件拒绝；不stash、不覆盖，显式路径提交后普通push；远端SHA核验在最终回复报告。
