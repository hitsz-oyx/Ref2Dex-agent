# v6指令：位姿式τ消费改进

本轮只改消费器、闭环GT、approach模式和S2c阶段接口；不新增历史审计、不重训v6、不跑RL。
实验沿用已登记`P-20261022-single-tau-calibration`的probe195/196；本次run单独标识为
`pose-consumer-improvement-20261022-r1`，不是之前校准或重复run，不新增学习更新。

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
