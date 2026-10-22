# v5：冻结重复的动作边界口径与完整分布

本轮为已有产物的工程重算，**0新仿真controls、0学习更新**。输入是预注册
`7dc82741`的25回合，而不是挑一个最好结果；执行代码/输入仍由原run身份绑定。
原v4结果提交`49b5ca8d`，本报告diff基线也是该提交。没有改变动作、评分器、阈值或旧成绩。

## 口径更正与适用范围

此前“实质clipping / 超界拒绝=0/0”属于**六例生产source-PD**，不属于神经网络R的四条链。
v4已列四条R链clip计数5/50/24/11，但与此前0/0并列时范围说明不够明确；在此明确撤回
将“无动作饱和”推广到校准R、原R或所有25回合的任何读法。**不是靠两个阈值解释差异**。

| 名称 | 本轮可复核定义 | 范围/分母 |
|---|---|---|
| `clipped_controls` | 一个tick的12维canonical raw action中，任一维`abs(raw-clip(raw,-1,1))>1e-6`即计1 | 原v4真实runner每条链都计；不是被裁的坐标数 |
| 实质clipping | 上述同一阈值；旧生产source逆映射在执行前拒绝`max(abs(raw)-1)>1e-6` | 旧0/0仅六例source-PD；NN执行是先裁剪再发PD，不拒绝 |
| `source_clipped_controls` | 冻结runner只在source分支发现`abs(raw)>1.000001`时递增并抛错，是**拒绝计数**，不是所有链的裁剪计数 | NN分支未检查，旧日志默认0不提供证据；新汇总对NN给null/not applicable |
| 边界/饱和命令 | 任一已执行维`abs(action)>=1-1e-6`，计该tick一次 | 包含合法端点，不必发生裁剪；另报前6维平移/腕角端点 |
| 实测关节越名义限位 | 实际q超出有界DOF的原lower/upper | 包括未超过安全margin的状态；不是动作饱和，JSON另报 |
| 关节安全触发 | 有界DOF实测q超出`lower−0.10`或`upper+0.10`；前三维单位m，其余rad | 原生产physical safety合同，回合在首次触发状态停止；终止状态T也计入，单回合可能多个DOF |

原计数源位置：`tools/audit/repeat_frozen_tau_chains.py:137`（source拒绝）、`:140`（各链裁剪）、
`:149`（原结果字段）；`src/wam_rl_policy_v3/source_pd.py:16`；
`src/wam_rl_policy_v3/physical_safety.py:26`。位置均相对`src/task/wam-rl-policy-v3/`。
原六例runner的`clipped_envcontrols`是初始化0并依赖上游拒绝，不能只看该字段自证零裁剪；
本轮用保存的target、live q、offset/scale和实际action独立反解了全部3230个active controls。
所得实质裁剪**0/3230**、实质逆映射超界**0/3230**、合法手指端点**2/3230**、腕/平移端点
**0/3230**，拒绝日志0。原helper记录的最大roundoff 2.384e-7与本次从已存nearest-angle
目标反解的excess0不是同一中间量；均未越1e-6。其余1402 inactive中性控制仍计总4632预算，
不冒充source控制见证。六例的限位中止2/6仍保留。

## 预注册n=5全量结果

所有向量严格按block0→4排列；这是同一已见序列/相同可见初态的重复，非5个独立任务或
训练种子。最长持握为≥5cm升高、实际force/耦合、离桌等分项的连续elapsed秒数。
成功即停，所以1s是删失下界；不能宣称能无限持握。该窄指标不同于8cm精确goal或完整RELEASE。

| 链 | 每回合最长持握s（n=5） | ever/final | clipping/controls | 边界命令/controls | 关节安全触发回合/状态 |
|---|---|---|---|---|---|
| source-PD | [1,1,1,1,1] | 5/5, 5/5 | 0/400=0% | 0/400 | 0/5, 0 |
| 原R+GT | [0,0,0,0,0] | 0/5, 0/5 | 5/1500=0.333% | 5/1500 | 0/5, 0 |
| 校准R+GT | [.533333,.4,.4,.333333,.566667] | 0/5, 0/5 | 50/1139=4.390% | 50/1139 | 4/5, 4 |
| 原R+WAM | [.966667,1,.8,0,1] | 2/5, 2/5 | 24/931=2.578% | 24/931 | 2/5, 2 |
| 校准R+WAM | [0,0,0,0,0] | 0/5, 0/5 | 11/963=1.142% | 11/963 | 5/5, 5 |

这批恰好所有动作端点均来自实质裁剪；全部是手指，前6维边界计数0。它不能推出测得的
关节没有越限：动作目标与动力学实测q不同。11次安全中止均为原关节gate，无阈值放宽。
总**4933 controls、25回合**；独立score的ever/final和持握分布全部匹配。

### 逐block（完整25项）

`clip`与`endpoint`都是tick计数；q-abort列给状态数/回合数及0-based原生DOF索引。
最后一列只计最长持握中**两端状态均held**的控制；状态t是action[t]之前。
| block / chain | controls | longest hold s | ever/final | clip >1e-6 | endpoint ≥1−1e-6 | q-abort states/episodes; DOFs | first clip / q-abort tick | hold-internal clip/controls |
|---|---:|---:|---|---:|---:|---|---|---|
| block0-baseline-gt | 300 | 0.000000 | 0/0 | 1 | 1 | 0/0; [] | 223 / None | 0/0 |
| block0-baseline-wam | 150 | 0.966667 | 0/0 | 0 | 0 | 1/1; [16] | None / 150 | 0/29 |
| block0-calibrated-gt | 258 | 0.533333 | 0/0 | 13 | 13 | 1/1; [16] | 152 / 258 | 0/16 |
| block0-calibrated-wam | 166 | 0.000000 | 0/0 | 0 | 0 | 1/1; [10] | None / 166 | 0/0 |
| block0-source-pd | 80 | 1.000000 | 1/1 | 0 | 0 | 0/0; [] | None / None | 0/30 |
| block1-baseline-gt | 300 | 0.000000 | 0/0 | 1 | 1 | 0/0; [] | 223 / None | 0/0 |
| block1-baseline-wam | 179 | 1.000000 | 1/1 | 0 | 0 | 0/0; [] | None / None | 0/30 |
| block1-calibrated-gt | 272 | 0.400000 | 0/0 | 10 | 10 | 1/1; [6] | 223 / 272 | 0/12 |
| block1-calibrated-wam | 200 | 0.000000 | 0/0 | 3 | 3 | 1/1; [6] | 152 / 200 | 0/0 |
| block1-source-pd | 80 | 1.000000 | 1/1 | 0 | 0 | 0/0; [] | None / None | 0/30 |
| block2-baseline-gt | 300 | 0.000000 | 0/0 | 1 | 1 | 0/0; [] | 223 / None | 0/0 |
| block2-baseline-wam | 201 | 0.800000 | 0/0 | 0 | 0 | 1/1; [16] | None / 201 | 0/24 |
| block2-calibrated-gt | 115 | 0.400000 | 0/0 | 0 | 0 | 1/1; [6, 8, 10] | None / 115 | 0/12 |
| block2-calibrated-wam | 138 | 0.000000 | 0/0 | 2 | 2 | 1/1; [6] | 80 / 138 | 0/0 |
| block2-source-pd | 80 | 1.000000 | 1/1 | 0 | 0 | 0/0; [] | None / None | 0/30 |
| block3-baseline-gt | 300 | 0.000000 | 0/0 | 1 | 1 | 0/0; [] | 223 / None | 0/0 |
| block3-baseline-wam | 300 | 0.000000 | 0/0 | 24 | 24 | 0/0; [] | 93 / None | 0/0 |
| block3-calibrated-gt | 300 | 0.333333 | 0/0 | 25 | 25 | 0/0; [] | 159 / None | 0/10 |
| block3-calibrated-wam | 245 | 0.000000 | 0/0 | 4 | 4 | 1/1; [15] | 106 / 245 | 0/0 |
| block3-source-pd | 80 | 1.000000 | 1/1 | 0 | 0 | 0/0; [] | None / None | 0/30 |
| block4-baseline-gt | 300 | 0.000000 | 0/0 | 1 | 1 | 0/0; [] | 223 / None | 0/0 |
| block4-baseline-wam | 101 | 1.000000 | 1/1 | 0 | 0 | 0/0; [] | None / None | 0/30 |
| block4-calibrated-gt | 194 | 0.566667 | 0/0 | 2 | 2 | 1/1; [6] | 191 / 194 | 0/17 |
| block4-calibrated-wam | 214 | 0.000000 | 0/0 | 2 | 2 | 1/1; [10] | 66 / 214 | 0/0 |
| block4-source-pd | 80 | 1.000000 | 1/1 | 0 | 0 | 0/0; [] | None / None | 0/30 |

机主点名的block3校准GT：**25/300=8.333%**，最大单维裁剪0.10181737；首次裁剪tick159，
最长持握状态49..59（elapsed10/30=.333333s），首次高度失败tick60。其他4个校准GT首次
裁剪152/223/无/191，首次高度失败64/62/62/68；**所有首次高度失败之前均未裁剪**。
所有25例的最长持握内部裁剪也均0；完整逐tick/逐轴列表在`final/summary.json`。

因此保留“部署的clipped链曾短时满足物理held谓词”的观测，但**不称该链已具备持握资格**：
校准GT五次都未达到1s，4次最终关节中止；8.33%是整回合的真实风险，不能隐去。时间顺序
排除了“该次159帧以后裁剪导致60帧首次高度失败”的解释，却不能证明裁剪与后续失稳无关，
更不能证明去掉裁剪会更好。原R+WAM两个成功回合整回合裁剪0，也不构成WAM统计优势。
最强“无动作饱和”窄任务阳性仅为source-PD5/5；不是GT-τ→R资格。

## 完整RELEASE仍0/6：必要缺口及与τ的关系

这6例是之前**新生产reset/source-PD面板每条一次**，不是把本次25个窄任务都当完整任务；
本轮没有新跑完整RELEASE。旧full ever/final均0/6保持，不用事后评分制造新成功。
证据与逐例数据见[完整任务分析](tau-executor-loop-audit-20261022.md)。

| 必要修复/验收 | 当前数字→期望验收数 | 与τ关系 |
|---|---|---|
| 可信终态/支撑对象声明；保留原录制，显式annotation_invalid或经独立意图证据重建 | 终点8mm/0.12rad容差区域与当前固定桌面支撑不相交2/6→“不相容却仍标有效”0/6 | **与τ无关**；airplane/cubemedium低于桌面约.805/.601m，而允许几何位移仅.0187/.0160m；当前任何控制器都无法同时满足两判据 |
| 明确科学任务是否要求严格经过录制路径；要求则执行器必须实际追踪，不能静默删gate | cubesmall/cup已终态稳定7.167/6.733s仍卡早期75/140帧；若fullpath为目标，未见milestone必须真实通过8mm等原容差，pending到path complete | **评分口径与τ无关**；达路径的控制能力可能与τ有关；原full0不能直接判“没有物理放置能力” |
| source-PD实测DOF跟踪/接触求解；保留原安全边界，不靠改margin解决 | cubemedium106帧/cylindermedium36帧q[16]<−.1rad，安全中止2/6→0/6；actual PD合同误差仍≤2e-5 | source-PD不用τ，故**该失败与τ无关**；不能用神经网络裁剪解释source中止 |
| 合法动作通道与被动指耦合；无裁剪source与NN分开验收 | source已实质clip0/3230；校准GT50/1139、校准WAM11/963并未解决，目标同任务正控实质clip0、关节abort0且实际完成任务 | **source接口问题与τ无关**；NN的分布/动作输出可能受τ条件影响，因果尚未自证；禁止单纯取消clip |
| 完整reference与真实末端观察；保持原任务身份 | 旧GT81状态/70有效controls；S2b f606e2b3已扩展reference及断言，下一新run需实测覆盖终态+至少1s，不把接口PASS算成功 | 完整τ覆盖是GT链必要条件；四例source已观察真实长后缀，故“缺观察”**不能解释全部0/6** |
| GT参考相位/live锚定和闭环条件一致 | ref-clock delta与live-WAM条件差异已确认；先让一个可信任务的source与GT链分别完成，正式面板两链各≥95% | **与τ构造/executor有关**；不是v6轨迹质量结论，补全末端不能单独解决早期高度失败 |

最便宜的下一项仍是一个意图明确、支撑相容、source合法的固定任务；使用S2b新参考另起
协议，先验收source与同τ executor两条执行链。不要先跑RL或以更大WAM训练修评分/PD问题。
本轮不启动该新run，三臂RL门禁继续HOLD。

## diff、复现与错误注入

所有代码路径相对Task：

- `src/wam_rl_policy_v3/action_boundary_audit.py:7`：纯数组重算裁剪/命令端点/真实q安全gate；
  NN的source计数显式不适用。期望“默认source零被误读”由4条链→0条；不改变物理结果。
- `tools/audit/audit_frozen_action_boundaries.py:17`：强制25/25及每链block0..4齐全，原始raw/clip
  与独立score交叉核对，输出全分布和25行表；期望日志与原始数组不一致条数0/25。
- `tests/test_action_boundary_audit.py:16`：包含故意漏报clipping、假未clip action、source超界
  三种相反输入；合法端点/舍入不能被混淆。源码mutants另故意删count验证、把endpoint当clip、
  删安全margin、绕过source拒绝，必须分别触发**断言失败**，不是导入/收集失败。
- `tools/audit/check_hermes_fix_mutants.py:19`：新增上述4个隔离源码mutants，不改live源码。

```bash
# 输入依赖本服务器原始run及scoring-final；输出必须fresh，不幂等、不覆盖旧证据。
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/home2/wyy/miniconda3/envs/graspenv/bin/python \
 src/task/wam-rl-policy-v3/tools/audit/audit_frozen_action_boundaries.py \
 --run outputs/wam-rl-policy-v3/single-tau-repeat-20261022-r1 \
 --scoring outputs/wam-rl-policy-v3/single-tau-repeat-20261022-r1/scoring-final \
 --source-run outputs/wam-rl-policy-v3/hermes-source-pd-20261022-r2 \
 --output outputs/wam-rl-policy-v3/clipping-independent-review/recount
# 期望PASS，n25，总4933，clips按source/baseGT/calGT/baseWAM/calWAM为0/5/50/24/11，
# 关节中止回合0/0/4/2/5，source六例active3230、实质clip0、合法端点2。

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q -p no:cacheprovider \
 --basetemp=tmp/clipping-independent-review-tests src/task/wam-rl-policy-v3/tests

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/home2/wyy/miniconda3/envs/graspenv/bin/python \
 src/task/wam-rl-policy-v3/tools/audit/check_hermes_fix_mutants.py \
 --output outputs/wam-rl-policy-v3/clipping-independent-review/mutants
# 期望13/13源码mutants被实际断言拒绝；不把collection错误算kill。

git diff 49b5ca8d -- src/task/wam-rl-policy-v3 \
 src/task/ptv3-shared-wam-v6/docs/audit/fix-review-20261022.md
```

本次完整测试/错误注入/verify实测结果见下方结算；不把期望输出当已经通过。

**无法自证/待裁决**：物理held仍依赖原FeedbackObserver启发式，不是独立抓握传感器真值；
不从时间先后自证裁剪或τ错位的因果；缺独立任务意图证据，完整RELEASE任务修改须具名裁决；
尚未在新终态reference上采集结果；Hermes/网页审计未复跑本轮diff，不宣称三方一致。

本轮最终实测：全v3 **206 passed /0 failed /0 skipped，17.61s**（`tests.log`）；
隔离源码 **13 killed /0 survived /0 collection errors**（`mutants/summary.json`）。
仅证明这些具体错误会被验收拒绝，不能代替物理资格或证明其余实现无错。
日志根`outputs/wam-rl-policy-v3/clipping-review-20261022-r1/`；`final/summary.json`和
`final/blocks.md`是本轮权威统计。运行环境graspenv Python3.8.20、NumPy/SciPy，CUDA禁用；
只做数组/JSON统计，无模型推理或仿真，不占GPU。

仓库门禁`python3 tools/verify.py --changed`实测 **VERIFY PASS**（`verify.log`）。
