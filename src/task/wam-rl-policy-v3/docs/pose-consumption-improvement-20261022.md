# v6指令：位姿式τ消费改进

本轮只改消费器、闭环GT、approach模式和S2c阶段接口；不新增历史审计、不重训v6、不跑RL。
实验沿用已登记`P-20261022-single-tau-calibration`的probe195/196；本次run单独标识为
`pose-consumer-improvement-20261022-r2`，不是之前校准或重复run，不新增学习更新。

## 预注册的唯一改进批次（运行前冻结）

**Decision：** b明显超过a则优先保留位姿消费；c明显超过b则优先改参考生成/几何重定向；
二者皆失败表示新控制器尚未验收，不能据此否决τ。最便宜有判别力的方法是同一个已见
cylindersmall初态上原a、新b、新c各5回合，共15回合，**最多4500环境controls/1800s**。
每回合300controls/10s，成功或原物理安全gate触发即停；0拟合/0RL更新。只跑这一批，
不根据中途成绩修改参数、补重复或重选checkpoint。b的验收目标是**≥4/5**，实际低于则
如实报告未达标。描述性“明显”预先定为成功数至少相差2/5；不当统计显著性。

- a：原R+live WAM11000，原8步点差分输入；模型实际上只消费24预测中的前8步，保持原路。
- b：同WAM完整24帧只提取**接触前物体坐标手腕SE3序列/预计接触时间**；不把预测指尖
  轨迹交给R追踪。新PD闭环读实测q/dq/palm/forces，实测contact切闭合、垂直抬升保持。
- c：S2b完整205状态参考，各帧hand通过各自object姿态转到object-local；live几何匹配
  参考相位，使用当前object SE3映射，再走与b相同的位姿约简/控制器。不是ref-clock差分。
  这消除一项条件不公平，**仍只是候选参考上界**，物理可执行性须由c结果证明。

相同base R checkpoint、shape、task context、native seed195、model seed196、30Hz/env1和
原物理gate；可见9字段初态误差要求≤1e-6。顺序按abc/bca/cab/abc/bca，完全报告block0..4。
原a条件生成仍冻结旧scheduler v2以保持与原a一致，生产v3没有回滚；物理成功沿原
5cm/force耦合/离桌elapsed1s窄任务，不变成新阶段分数。1s成功即停是下界。五次重复不等于
五个独立任务/训练种子。完整RELEASE不在本次比较中，三臂RL继续HOLD。

## 四项落地与期望数

| 项 | diff（相对本Task） | 期望验收 |
|---|---|---|
| 位姿消费 | `src/wam_rl_policy_v3/pose_consumer.py`的`reduce_to_wrist_plan/PoseConsumer` | b成功≥4/5、相对a至少+2/5；持握分布/限位/clip全报；不承诺实验前已做到 |
| GT闭环 | 同文件`object_relative_reference/live_reference_future` | 物体变换+0.4m时目标同移+0.4m；选择测量匹配phase12而非录制时钟；b/c同一执行入口 |
| approach模式 | `WristPlan.mode/eta_s/approach_direction_object` | 24帧无contact返回approach、contact_index=null；ETA加倍时目标前进量减半；预测contact不能冒充实测contact，丢样本0 |
| S2c对接 | `tools/run/run_pose_consumer_improvement.py`读`StagedProgress` | 使用其current.contact切phase，保存全部stage snapshot；本线程没有另写dense reward或阶段成功判据 |

接触估计：预测24帧中至少2个指尖距CAD≤25mm的首帧；无则mode=approach，输出有限的
ETA估计（非事件标签）及末端预计抓取方向。controller的approach只消费方向/ETA/目标朝向。
contact窗口用接触前四步lookahead wrist pose。手腕由6个刚性base点Kabsch拟合，不跟指尖。
当前测量mount恢复旋转关节EulerXYZ代表，平移PD步≤25mm、角PD步≤.12rad，加实测速度
阻尼；不扩大action范围。实测contact后保持世界xy/朝向，.2s闭合后以.075m/s抬高最多.10m，
指关节只用测量force做有界闭合/松弛。手指接触前保留同base R的显式无τ状态先验（valid=false），
其wrist输出由新控制器替换；b/c不向R伪造压缩τ。无预测保留invalid并保持上次wrist目标。

S2c接口从其共享文件只读拷贝到run冻结源码，独立保存contract与stage snapshots；不修改其
未提交文件。原v6/v5依赖和native host亦冻结到run/runtime-code，所有执行入口可复查。

## 命令与运行环境

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q -p no:cacheprovider \
 --basetemp=tmp/pose-consumer-repro-tests src/task/wam-rl-policy-v3/tests/test_pose_consumer.py
# 期望5 passed/0 failed，含错误相位、错误物体锚定、虚构contact的相反输入。

nvidia-smi
# 仅5/6仍空闲时启动；fresh输出，不覆盖checkpoint或原run。
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TMPDIR=$PWD/tmp \
/home2/wyy/miniconda3/envs/graspenv/bin/python -u \
 src/task/wam-rl-policy-v3/tools/run/run_pose_consumer_improvement.py \
 --output outputs/wam-rl-policy-v3/pose-consumer-improvement-repro --native-gpu 6 --model-gpu 5
# 完成应COMPLETED、runs15、每链n5；目标b>=4/5是效果门，不与工程完成混淆。
```

client：graspenv Python3.8.20/Torch2.4.1+cu121，native：dexplore_v120_train
Torch2.0.1+cu118；GPU5模型、GPU6物理，4保留空闲；不碰0–3/7。
运行必须依赖已有原base checkpoint、WAM11000、geometry/stats、S2b reference-r2及S2c接口。

## 不阻塞本轮：完整任务物理资格两颗钉子的修法

1. **支撑/终点2/6矛盾**：增加任务明确的support surface身份/几何（桌面、地面或已声明其他
   物体），由原意图证据裁决；同一坐标系下先查终点容差与支撑集合是否相交。矛盾项显式
   annotation_invalid且保留序列/原始标签/总数，禁止静默丢弃或改HOLD。只有证据支持时
   重建正确目标/支撑；目标“不相容却仍标有效”2/6→0/6。先修声明，不靠τ训练解矛盾。
2. **source实测关节限位2/6**：检查q[16]被动耦合目标与真实force/惯性冲突，给finger PD加
   有界目标变化和实测dq阻尼；必要时单独版本化solver/关节阻尼参数，保持原安全gate和
   task不变，禁止扩大margin。先在源两条上验证安全abort2/6→0/6、actualPD合同误差≤2e-5、
   合法动作无实质裁剪，并完整观察终态；之后才能回资格面板，不借窄任务阳性过门。
   source链不用τ，故这两类修复都不需要先改WAM。它们是方案，**本轮没有声称已修好**。

运行前r1因冻结旧runtime无未使用的EncodingUnavailable导入而退出，0物理controls；
移除该导入后以r2恢复同一个15回合预算，r1日志与failure.json保留。

## 本轮结果：原型落地，位姿改进未达标，不替换默认原R

实际运行代码`b498d580d30d3676f38bbf5877ebd7312e6b5d92`；S2c依赖来自已提交
`fbcff46b`的stage接口（run中另有冻结副本），不是本线程重复实现阶段reward。
r2完成**15/15、3849环境controls /402.318s、0学习更新**，WAM推理218.768s；在预注册
4500/1800预算内。r1启动导入失败0步，不算一个成功回合或被悄悄丢弃的物理试验。
本轮没有追加第二批、改参数、训练或历史重评分。

| 链（各n=5，block0→4） | 最长持握elapsed秒全分布 | ever/final | controls | final-output clip | 原关节安全abort |
|---|---|---|---:|---:|---:|
| a 原R+WAM | [1,1,.133333,0,1] | 3/5；3/5 |924|9|1/5|
| b 位姿WAM | [0,0,0,0,0] | 0/5；0/5 |1500|0|0/5|
| c 闭环GT | [0,1,0,0,0] | 1/5；1/5 |1425|0|0/5|

**效果门FAIL：b没有达到≥4/5，并差于a；本轮不采纳此位姿消费器替换原R。**
按事前“至少相差2/5”的描述性screen，消费改进分支未触发；c−b只有1/5，参考质量优先
分支也未触发。不能将c的单个阳性认证成公平物理上界，不能从本批次排除τ质量。
原a这批3/5与旧2/5分别保留，未合并成新成功率。新方法没过，不以工程PASS冒充改进。

9可见初态数组对原始初态最大误差0；实际native PD目标对闭环动作合同最大差
**2.4073e-7**，没有扩大网格/动作范围/物理margin。GPU5/6已释放至各2MiB，native正常
退出0，无大训练。final-output clip仅统计最后输出；新链指先验使用了原R的有界无τ输出，
不把0裁剪宣称为所有内部NN输出从未超界。

### 四项交付的实际完成程度

1. **位姿消费已实现、未产生所要求的效果**：b有效模式1485个控制全为approach，预测
   contact-window为0；尚未走到可认证的抓取。不能以没有物理abort代替任务成功。
2. **GT已改成live-object映射与测量相位**：c由旧ref-clock改成同一位姿消费入口，真实
   结果1/5；它有1375个contact-window控制、35个approach控制及15个启动无计划控制，
   仍不具备稳定上界资格。完整参考用S2b205状态，原81状态只作为初态对齐输入，未偷换。
3. **approach表示已接到动作输出**：没有24帧内接触时，保留mode/ETA/方向，未删样本或
   为R塞假τ；但“短窗末端姿态”尚未形成有效的长程抓取approach，b0/5就是效果未过。
4. **S2c接口已接通、闭合保持分支尚无运行见证**：15例其stage contact-ever均0，虽然
   旧窄witness在a有3个、c有1个阳性；新b/c contact phase切换0/10。因此本次不声称
   已验证接触后闭合/保持分支。阶段成功与原窄witness不同，未改任一阈值抹平差异；
   本线程没有重写S2c判据或dense reward。后续改进应先让approach抵达预期抓取姿态，
   并与S2c对齐“驱动phase的测量contact信号”接口，再执行闭合保持；本轮不追加评估。

这是一轮**未达效果目标的具体实现**，不是历史负结果审计或对τ路线的否决。正式三臂RL
继续HOLD；完整RELEASE两类修法在上节，未以本窄任务结果宣称钉子已拔除。

### 精确diff与独立复现

实现diff：

- `src/wam_rl_policy_v3/pose_consumer.py:54` 位姿约简；`:101` 测量PD/闭合保持。
- 同文件`:24`、`:31` 各自object-local参考与live SE3映射；不按ref clock推进。
- 同文件`:46` mode/ETA/方向；`:129` approach动作路径。
- `tools/run/run_pose_consumer_improvement.py:43`、`:107` S2c阶段接口；`:120`、`:131` GT/共享consumer。
- `tests/test_pose_consumer.py:10` 等5个语义单测，期望和实测均5 passed/0 failed。

```bash
git diff 7a00fdca b498d580 -- src/task/wam-rl-policy-v3
# 不启动仿真，完整打印此次所有链的5个结果；期望COMPLETED，3849controls，a/b/c=3/0/1。
PYTHONDONTWRITEBYTECODE=1 /home2/wyy/miniconda3/envs/graspenv/bin/python \
 src/task/wam-rl-policy-v3/tools/run/summarize_pose_improvement.py \
 --run outputs/wam-rl-policy-v3/pose-consumer-improvement-20261022-r2

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q -p no:cacheprovider \
 --basetemp=tmp/pose-consumer-all-repro src/task/wam-rl-policy-v3/tests
# 实测211 passed /0 failed /0 skipped，17.85s；不是物理成功保证。
```

完整产物`outputs/wam-rl-policy-v3/pose-consumer-improvement-20261022-r2/`包含manifest、
summary.json、每回合trajectory/consumer/stages、冻结源码与native exit回执。工程测试及
r1/r2日志在`outputs/wam-rl-policy-v3/pose-consumer-improvement-20261022-preflight/`。
重跑物理命令可用上节fresh路径，但不保证GPU隐藏接触状态确定性；本轮已按预算结束。

本轮结果文件门禁`python3 tools/verify.py --changed`为VERIFY PASS（preflight/verify.log）。
尝试的`--base`参数不受该CLI支持，退出2，未计作验收或覆盖上轮门禁；日志另存。
