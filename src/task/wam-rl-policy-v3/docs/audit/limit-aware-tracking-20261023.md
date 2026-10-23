# 执行器P0：限位准入、不可达停推与速度前馈

| 观测 | 含义 | 决定 |
|---|---|---|
| 限位中止计数显著下降，且 brief48 的可行域**边界外扩** | 准入+检测有效 | 把这两个件固化进执行器主链 |
| 限位下降但成功率不动 | 限位只是症状 | 继续查相位机与接触段（不宣称修好） |
| 参考被大量判为“不可达” | **工作空间/物体几何与任务不匹配** | 把“任务可用几何域”写成任务侧约束并交机主 |
| 前馈带来明显误差下降 | 带宽是主因之一 | 固化前馈；考虑 jerk 受限生成（Ruckig/SciPy 样条） |

## 运行前登记

Decision：检查参考/命令合法性是否能防止持续顶限，以及在同一对象参考下前馈是否扩展数值通过格。不能由“停得更早”推断成功，更不能将准入拒绝从分母删除。

代码事实：当前Inspire是六自由度虚拟浮动腕+耦合手指；已有JointAdmission/MarginAdmission和SourcePDGuard，但不是完整机械臂IK。新实现复用原生PD及全部18关节耦合，针对该浮动腕提供解析位姿准入；不冒充串联机械臂/全指尖碰撞与力闭合求解。

保持brief48的180格、原尺寸/速度/秒时间轴、原15mm/5deg/10mm/10deg/50mm/0.2s/余量/碰撞标准，seed195与同一初态/参考。A仍是控制tick的运动学固定代理，自碰撞缺口保持；本轮不修改碰撞过滤或物理安全门。新保护必须在每次PD命令送出之前验证，包含获取阶段；完整参考在进入TRACK前按实际控制tick及原始knot核验，拒绝写明首个点/关节，不改变参考来通过。

新run `outputs/wam-rl-policy-v3/limit-aware-tracking-20261023-r1/`，最多180格/40,000 controls、7200墙秒、2GiB输出；GPU仅空闲4/5/6/7，当前4/5/6空闲、7占用不碰。先离线错误注入与已保存轨迹复核，再跑物理；不训练、无τ、无模型调用。磁盘<20GiB、OOM、非有限、native故障停止，不自动换配方重试。

预期而非结果：命令端有限关节超出内余量=0；越界参考必显式拒绝；冻结静止状态1秒仍有>15mm/5deg误差时应停止继续施加前进命令。实际q与关节门独立保留，不承诺目标合法能保证接触动力学状态不越界。真越界（超nominal）、贴边（nominal内距边界≤0.10）、命令超界但实测未到逐tick分开报告，并另报原±0.10门中止；“贴边”不等于已越界或物理中止。将新保护停止计作失败，不能改名成功。

brief48原A/B各90格、B0/90和最小关节余量0.047rad仅作冻结基线，本轮新增数值另报，不重跑旧无保护矩阵。预计新保护不必改变这批原本无限位中止的计数；不能在0→0上声称显著下降。

历史状态：PRE_REGISTERED（e6ec2943，先登记后运行）；最终结果见下。

## 补充预注册：有真实限位失败的最小回归（启动前）

brief48基线关节中止0/180，无法证明对已失败链的修复收益。因此另对S2e原几何τ→原R→v7消费单臂做5次相同初态重复，仅插入同一准入/停推保护，原R/原PD/原native与评分器冻结，不重跑旧对照。旧几何臂关节4/5来自brief40引用；旧时域300且成功即停，新回归运行至600或安全/保护停止，前300限位单独配对，旧600结果不得补造。指目标内余量不满足时显式选择耦合可行内区间并落盘；腕目标若不可达则拒绝；保护停止算失败而不是成功。主门槛：前300关节中止<4/5且新增table中止0；另报600末态实际持握/抬高/原15mm目标误差与认证接触，不能拿关节中止减少替代任务成功。

run `limit-aware-tracking-20261023-geometry-r1`，最多3000 controls/1800墙秒/512MiB，仅5和6空闲卡（当前4由本任务矩阵占用，7他人占用不碰），旧模型seed196/native195；不更新模型，不调用WAM预测，不增新基线。两部分合计交互上限43000/新写盘2.5GiB；首次故障停并保留，不自动改配方。此回归用几何τ是复用原失败执行链；对象TRACK矩阵仍完全不使用τ。

## 结果后的工程处置登记（不改既有判据/结果）

r1完整180格已完成，但预测制动使A数值达标8→5、B仍0，不能固化该控制律。交付版保留默认几何/18关节目标准入及ProgressWatchdog，`--predictive-braking`显式选择未获资格的q+0.10s·dq制动（r1旧快照等价为开启；复现r1必须用e6ec2943冻结代码）。对已合法且无需更改的float32动作直接原样返回，避免重复编码扰动接触轨迹；这只保证命令透明，不保证物理求解器跨run确定性。

只追加4格工程smoke：最低参数直线，A/B×FF，原30mm/s与20mm尺度、同参考同native195，不改门限、不重跑180格，不作性能收益统计。run `limit-aware-tracking-20261023-admission-only-smoke`，≤1000controls/300墙秒/64MiB，GPU4用前核空。比较原4格获取与TRACK交接、真实目标准入/PD合法；目标是保留原合法命令与抓取交接，暴露任何意外投影。测试仍须能拒绝超界/卡住参考。两物理run+smoke累计硬上限44000controls/2.6GiB，未启动训练。

## 最终结算

**工程防线已实现，物理P0未解除。** 结果不满足预注册第一行：180格预测制动试验中，A数值通过8→5、B仍0，不存在可行域净外扩；几何/R小回归虽关节中止4/5→0/5，却5/5提前拒绝、成功0/5。不能把“停止得更早”当抓取修好。因此预测制动不进入默认控制律；默认保留目标准入与无进展停推。前馈对应第四行的部分支持，不能解释成全部瓶颈已解决。

第三行只适用于R意图子路径5/5拒绝，不能推断物体或任务本身不可达：通用对象参考180/180通过解析几何准入；被拒的是R局部命令转换出来的腕目标，不是完整任务的几何上界。第二行也只能作“中止计数下降而任务未改善”的描述，早停暴露时长不同，不能据此认定限位只是伴随症状。

Task `wam-rl-policy-v3`，experiment `P-20261023-limit-aware-tracking`，branch `main`。运行r1的核心代码为`e6ec2943`；几何回归wrapper为`13fb97f7`，其中加入的guard仍逐字等于e6ec2943。交付策略处置/动作透明分支为`e6ca53ed`。完成证据中5份核心文件与e6ec2943逐字匹配，几何回归380份既有源码与原S2e逐字一致。交付HEAD不能冒充实际物理run版本。

### 三处落地及适用域

1. `reference_admission.py:32::JointEnvelope`：按本机原生18维limit/offset/scale构建内区间，平移20mm、转角20mrad，窄量程最多10%；包括被动耦合7←6×1.05等及thumb16/17←15×0.6/0.8。`:65::pose_configuration`对虚拟XYZ浮动腕作解析逆映射，连续腕采用最近等价Euler分支；`:84::admit_trajectory`在全部control tick与原knot检查后才准入，失败记录首时间/关节。不是串联臂IK，不认证扫掠碰撞或力闭合；palm与虚拟root原点不同/腕实际有限则显式拒绝。
2. `reference_admission.py:98::protect`在原生PD转换前约束所有有限关节，记录合法替代目标；native返回后独立验同一PD合同。`ProgressWatchdog:134`在1秒实测窗口中检测静止/无改善的超差目标并锁存停止，不继续积分或推目标；每格重建实例。移动参考只把缺乏实测运动判为僵持，不能把保持恒定跟踪滞后误叫静态不可达。此停推是失败类别，不是成功。
3. `object_tracking.py:101/124`为逐tick参考/命令接点；`run_object_tracking.py:94/104`覆盖获取与TRACK的真实step，`:169`为整参考准入、`:201`为停推。速度前馈沿用brief48的一阶导：物体线速度+旋转杠杆臂、世界角速度转连续腕角速度，再乘1/30进入原位姿目标。原PD增益、限幅、物理安全门与评分阈值未改。**前馈本身已有实现，本轮新增的是准入/停推与独立带宽量化。**

源码考据也更正背景的过强说法：仓库已有`grasp_geometry.py::JointAdmission`、`source_ceiling.py::MarginAdmission/SourcePDGuard`，但它们不是全机器人限位感知IK；本轮新TRACK接点此前未使用这些准入合同。不能以grep无IK库断言全仓没有目标余量保护。本轮未宣称所有旧PPO/source入口都已切换新防线；未过第一行的预测制动不得推广到它们。

### 全矩阵：固定原尺寸/速度/时间律，180/180完整执行

各组45个不同参数格，每格1次，不是45个独立任务或seed。对象参考180组与brief48对应文件逐字一致，reset q/object一致；FF开/关90对的交接q/object最大差均0。与旧控制器比较时，新增获取制动改变了交接：对象位置差4.1848mm、姿态差13.9259°、q最大差0.14818rad；不把这种差异隐去或从评分窗口删去。旧/新间不能把变化唯一归为TRACK本体；新run内部FF比较的交接一致。

| 组（每组45格） | 原brief48数值通过→本轮 | 位置P95跨格中位 mm | 姿态P95跨格中位 ° | 完整执行 |
|---|---:|---:|---:|---:|
| A / FF关 | 0→0/45 | 26.552 | 9.163 | 45/45 |
| A / FF开 | 8→5/45 | 20.867 | 9.075 | 45/45 |
| B / FF关 | 0→0/45 | 48.029 | 12.444 | 45/45 |
| B / FF开 | 0→0/45 | 37.306 | 12.254 | 45/45 |

A新增figure8/50mm/30mm/s一格，原先circle20、figure8 20、helix20、random20四格丢失，原8格中保留4；合计5，不构成边界外扩。全部通过格仍平均30mm/s，没有减速改参考；100/250mm/s未建立新可用格。B无新增可用格。

A仍是control tick运动学固定对象的代理，不是严格焊接/负载惯性正控制。B90/90真实接触执行完整路径，最大手物漂移0.3401mm/0.7821°，未触发滑移或脱手；最小有限关节余量0.039764rad在`thumb_intermediate_joint`。A最小0.085755rad在`thumb_proximal_yaw_joint`。连续腕角不纳入有限门。不沿用旧失败总数：本轮A失败85/90、B失败90/90，均为数值跟踪/闭合超阈值，物理limit/slip/记录的collision中止均0。

自碰撞过滤仍关闭，不能认证无自碰撞；A刚体代理与本机缺overlap字段的限制保留。没有因为这些未覆盖项把数值通过当完整资格PASS。

### 真越界、贴边、命令越界分开计数

定义：真越界=实测q超出URDF nominal；贴边=仍在nominal内、距任一边≤0.10（平移m/转角rad分别解释）；旧物理门=超nominal±0.10；“超nominal但尚在门容差内”另列。命令超界但实测未到=实际送出的PD目标超nominal、同一关节执行后实测仍在nominal内。连续腕完全排除。下表统计有任一符合关节的control，不把多关节事件相加充当controls；不同关节可令同一control出现在不同列。

| 新数据 | controls | 真越界 | 内侧贴边≤0.10 | nominal外但未越旧门 | 旧±0.10门外 | 命令越界、实测未到 |
|---|---:|---:|---:|---:|---:|---:|
| 对象矩阵 ACQUIRE | 9000 | 0 | 3240 | 0 | 0 | 0 |
| 对象矩阵 TRACK | 24860 | 0 | 24860 | 0 | 0 | 0 |
| 几何/R回归（提前拒绝） | 77 | 0 | 12 | 0 | 0 | 0 |

矩阵所有33,860个实际PD目标均通过内余量准入（独立违规0）；原生PD独立重建最大差2.11735e-7。原brief48的物理关节中止也是0/180，0→0不能称显著下降。TRACK贴边24,860/24,860不等于危险/中止：thumb yaw等合法持握构型本就在该宽阈值内；逐关节事件表在limit-audit.json。不能把“贴边”与超URDF/超旧门混算。

预测制动q+0.10秒·dq仅是保守风险启发式，不是经过验证的停止距离/接触动力学求解。矩阵获取540/9000、TRACK6495/24860，共7035/33860（20.78%）步介入。去重PD改动7035步；外层第二次检查只看到540次，不可漏掉TRACK内层已经完成的6495次。最终canonical超范围0不等于无servo限幅或无保护干预。

### 前馈与带宽：不把误差下降写成成功

每格用共同时间窗口拟合 y(t)=β·reference(t−δ)+常数偏置，δ按30Hz网格搜索[-min(.25,T/4), min(.5,T/4)]；同一窗口比较残差。β是整条轨迹的描述性标量幅值拟合，不是已辨识的频率响应。每组42/45可拟合，另3条太短单列；位置/姿态P95仍使用全部45格。

| 阶段 | FF关→开位置P95中位 mm | 姿态P95中位 ° | 延迟中位 ms（42格/组） | 幅值β中位 | 幅值衰减1−β |
|---|---:|---:|---:|---:|---:|
| A | 26.552→20.867 | 9.163→9.075 | 233.3→166.7 | 0.7607→0.8052 | 23.93%→19.48% |
| B | 48.029→37.306 | 12.444→12.254 | 166.7→133.3 | 0.5393→0.5847 | 46.07%→41.53% |

B位置P95下降22.33%，但仍37.306mm，远大于原15mm；姿态12.254°仍大于5°。延迟约−33.3ms、幅值衰减改善4.54百分点，支持保留前馈，不能说带宽是唯一主因。已有brief48数据的新离线同口径拟合另存baseline-bandwidth.json（A延迟233.3→166.7ms，B200→150ms），不是新仿真。

### 原失败链的5次回归：0中止不是修好了

原S2e几何τ→原R→原v7消费：5个同一可见初态重复，不是5物体/5seed；native195/model196。没有WAM预测、没有更新R。记录了内部几何τ是复用原失败链，不能把这部分叫“无τ”。对象TRACK矩阵仍无τ、无模型调用。

| block | 旧关节门/旧controls | 新controls | 新物理门 | 新保护终点 | CPU重建被拒R腕目标Z m | 旧600末态 / 新成功 |
|---|---|---:|---|---|---:|---|
| 0 | joint / 106 | 15 | None | REFERENCE_REJECTED | 0.016080 | 未测 / 0 |
| 1 | joint / 156 | 17 | None | REFERENCE_REJECTED | -0.094343 | 未测 / 0 |
| 2 | horizon / 300 | 15 | None | REFERENCE_REJECTED | -0.038740 | 未测 / 0 |
| 3 | joint / 141 | 15 | None | REFERENCE_REJECTED | -0.038187 | 未测 / 0 |
| 4 | joint / 124 | 15 | None | REFERENCE_REJECTED | -0.038738 | 未测 / 0 |

原生root-Z名义[0,3]m，本轮内域[.02,2.98]m；4例R目标低于0，另1例.01608m低于内余量。其是受限位姿伺服**之前**的R意图，不是最终送出的PD；因此不否认brief40“触门关节末次实际PD限外0/21”。这些也不是物体最终任务目标，不能从它们证明任务物理不可达。

交给任务侧的几何域合同：约束的是`T_world_object @ inv(T_hand_object)`得到的腕原点，而不是只看物体中心；本机腕X/Y内域[-4.98,4.98]m、Z[.02,2.98]m，连续腕角分开。独立手指目标还须满足全部耦合区间，例如拇指pitch约[.033333,.53]rad（下界由被动中节0.6比例收紧）、yaw[.02,1.13]rad；这些只是几何/目标必要条件，不能据此认证接触或碰撞可达。没有按物体身份设常数。

新回归77controls、关节0/5、table0/5、拒绝5/5、接触/抬升/持握认证各0/5，最长持握全0秒、最终目标误差约81.987mm、成功0/5；均未走到600，不能升级为600末态失败率或安全性证明。原4/5关节事件的长暴露被提前拒绝截掉，不能作因果改善或显著性检验。所有初始可见字段与原记录最大差0；block0/1公共前缀q/dq/object/action/forecast逐位相同，block2/3有微差、block4闭环前缀q差可达0.512rad，跨run物理确定性未获资格，不隐瞒。

同一guard在旧21个关节失败的保存状态上21/21能在首个nominal越界前报警，领先2/80/262个controls（min/median/max）。这是新增离线预警时机证据，不是21次中止都能被修好的反事实。真实子步驱动误差/成对力仍不足，不能归因成某一个PD参数bug。

### 交付默认策略、回归与未覆盖

默认保留静态参考/18关节目标准入和ProgressWatchdog；预测制动必须显式`--predictive-braking`。准入拒绝保留在分母、记录原因；目标不合法只可显式合法替代，不伪造成功。合法、无需替代的float32命令原样返回。

最终CPU检查：32 passed / 0 failed（4文件，2.19秒）；8/8注入错误被检出，覆盖漏限位、漏被动耦合、漏制动、发送未保护动作、漏stall、混贴边/越界、错误机器人校准、仅检查端点。新增合成轨迹可恢复已知200ms延迟/0.7幅值并剔除常量偏置。旧真实TRACK代码反例：目标X=6m仍输出推进，新代码拒绝；对thumb16外滑的合法目标，预测制动分支将根推进0.009m变为0（CPU证据，不是物理保证）。

**关闭预测制动后的生产准入透明性**：在旧真实9000个获取命令+24860个TRACK命令上调用当前生产protect(predictive_braking=False)，33860/33860逐位不变，最大动作差0。该证明覆盖实际记录状态上的命令语义，不能替代新物理rollout或完整控制链资格。

4格默认策略smoke：**NOT_RUN_RESOURCE**。GPU4预检时22563MiB/100%，GPU5在新目录预检时21651MiB/86%，GPU6同样24211MiB/56%；均未启动仿真、两次0controls，保留各client.log，不覆盖。没有杀/抢他人进程，没有转去0–3或占用7，没有自动轮换重试。最小补证：有一张空闲4–7后，只做已登记4格smoke（≤1000controls），不重跑180格。当前不得宣称默认分支通过物理非回归。

**下一步最小动作**是先补这4格默认准入/停推smoke；R支路下一项是把局部R增量与最终抓姿的语义分开，生成合法替代意图后同5例回归。不能因R局部意图被拒就宣布物体不可达，也不把保守预测制动继续推广到旧PPO/source链。RL保持HOLD，无训练/模型更换/schema/PD/门阈修改。

### 可核对命令、版本和产物

```sh
PY=/home2/wyy/miniconda3/envs/graspenv/bin/python
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
# CPU独立复算，不启动仿真；输出为本run派生文件
$PY src/task/wam-rl-policy-v3/tools/audit/audit_limit_tracking.py --run outputs/wam-rl-policy-v3/limit-aware-tracking-20261023-r1
$PY src/task/wam-rl-policy-v3/tools/audit/audit_limit_geometry.py --run outputs/wam-rl-policy-v3/limit-aware-tracking-20261023-geometry-r1
$PY outputs/wam-rl-policy-v3/limit-aware-tracking-20261023-r1/check_admission_transparency.py
$PY src/task/wam-rl-policy-v3/tools/audit/check_admission_mutants.py
PYTHONPATH=src/task/wam-rl-policy-v3/src $PY -m pytest -q -p no:cacheprovider src/task/wam-rl-policy-v3/tests/test_reference_admission.py src/task/wam-rl-policy-v3/tests/test_object_tracking.py src/task/wam-rl-policy-v3/tests/test_source_pd.py src/task/wam-rl-policy-v3/tests/test_continuous_wrist.py
# 历史实际物理命令（已执行；不可覆盖产物）。r1默认制动开启对应e6ec2943，当前交付默认关闭，不能拿当前代码假称同一次运行。
$PY -u src/task/wam-rl-policy-v3/tools/run/run_object_tracking.py --output outputs/wam-rl-policy-v3/limit-aware-tracking-20261023-r1 --gpu 4 --matrix full --max-controls 40000 --reference-start outputs/wam-rl-policy-v3/object-trajectory-tracking-20261023-r2/reference-start.npy --seconds 7200
$PY -u src/task/wam-rl-policy-v3/tools/run/run_limit_geometry.py --output outputs/wam-rl-policy-v3/limit-aware-tracking-20261023-geometry-r1 --native-gpu 5 --model-gpu 6
# 仅当有空卡时在全新目录补4格；此命令本轮尚未成功执行
$PY -u src/task/wam-rl-policy-v3/tools/run/run_object_tracking.py --output outputs/wam-rl-policy-v3/limit-aware-tracking-20261023-admission-smoke-FRESH --gpu <空闲的4至7> --matrix sentinel --max-controls 1000 --reference-start outputs/wam-rl-policy-v3/object-trajectory-tracking-20261023-r2/reference-start.npy --seconds 300
```

预期CPU复核：180格、reference_admissions180/180、inner_pd_violations=0；几何5拒绝/0成功；透明性33860/33860未变；32 passed/0 failed、mutants8/8。完整逐格在r1/limit-matrix.csv，分类/拟合/PD验证在limit-audit.json，R逐例与被拒坐标在geometry-r1/geometry-audit.json；pre/post state/PD/参考/接触均保留。准确patch：`git show e6ec2943 -- src/task/wam-rl-policy-v3`（初版防线）、`git show 13fb97f7 -- src/task/wam-rl-policy-v3`（冻结回归入口）、`git show e6ca53ed -- src/task/wam-rl-policy-v3`（实验制动隔离与透明默认）。实际geometry runner.patch可审查每个旧runner变动；复现历史必须使用产物runtime-code，不可把当前源码覆盖过去。

运行环境：client graspenv Python3.8.20、numpy1.24.4、SciPy1.10.1；native dexplore_v120_train Torch2.0.1+cu118，GPU PhysX/CPU tensor exchange；R模型使用原Torch2.4.1+cu121。对象矩阵native seed195，几何模型196/native195。主矩阵1546.068墙秒；两run共33937controls、0训练更新。自有4个client/native PID全部退出，两个native均返回0；别人随后占卡不属于本任务显存，未干预。

产物快照：主矩阵664,502,496 bytes，几何回归3,081,443 bytes，两次GPU预检日志各354 bytes（均0controls）；低于2.6GiB上限，outputs不入库。最后共享盘可用70.59GiB，高于20GiB门；相对开工共享盘变化不归因于本任务。未删历史outputs、未碰其它会话脏文件。pull --rebase被脏文件拒绝，未stash，显式路径提交后普通push；最终HEAD/origin/main证明随交付给出。
