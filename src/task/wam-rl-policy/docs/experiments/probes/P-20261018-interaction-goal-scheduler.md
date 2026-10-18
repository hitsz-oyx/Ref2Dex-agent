---
schema: ref2dex.probe.v2
probe_id: P-20261018-interaction-goal-scheduler
experiment_id: P-20261018-interaction-goal-scheduler
date: 2026-10-18
task: wam-rl-policy
branch: main
git_commit: recorded-in-run-manifest
claim_id: C1
hypothesis_family: HF-ordered-goal-scheduler
probe_index_in_family: 2
seed_pool: debug
seeds: [0]
decision_changed_if_positive: retain causal hand-object evidence and continuous progress tracking for manual goal review
decision_changed_if_negative: fix progress regressions or interaction proxies before any WAM integration
status: RUNNING
run_id: interaction-goal-scheduler-20261018-r2
---

# 手物历史条件的goal调度与3D审阅

Class: Blocker / engineering Probe。用户指出原几何调度与展示均粗糙；本轮不改WAM。
问题：手物历史、慢速/低频观测、里程碑切换是否导致不合理短目标？结果决定保留哪些
调度机制。最低成本为失败回归测试与同输入的已有轨迹重放，不运行新物理或训练。

## 原方案可复现的不足

新增检查在旧实现上全部失败：持续慢速进展被8秒计时器判停滞；低频观测跨过里程碑
会卡住；切换时短目标瞬跳约20.2mm；接口不读取手部历史。同一物体位姿无法区分
手还远和手已接近。旧13片段回放通过只代表原几何合同，不能支持“调度已合理”。

## 当前修改

- 以局部连续进展和指尖接近的实测改善更新停滞计时，不要求每次改善都跨完整里程碑。
- 短观测间隔与有界步长内，允许线性平移/短弧SO(3)观测插值提供有序穿越证据；
  穿越分数必须单调，不以任意跳跃或全局最近帧确认进度。最终位置仍需实际持续达标。
- 当前局部匹配与已确认里程碑分开，允许投影在已确认位置之前的邻近段；前视跨度
  变化率限制为40mm/s，避免确认容差和硬切换把目标突然推到下一整段。
- 新 `InteractionObserver` 只读当前/过去hand11和物体SE(3)，保留APPROACH、PREPARE、
  CONTACT_CANDIDATE、COUPLED、LOST、UNKNOWN。没有手部观测时显式退化为几何调度。
- 近表面两指不等于耦合；持续移动时物体与手相对运动需足够一致。专门加入
  “静止手旁物体移动”负例。丢失先前随动关系有时间滞回。所有阈值为可调启发式，
  不是接触真值、校准概率或抓稳证书；UNKNOWN/candidate不虚构控制能力。
- 接近/准备状态给较短意图；随动证据稳定后参考实测运动尺度调前视，丢失控制时
  收缩并恢复。用户显式readiness保留；没有学习/认证物理可达性。

## 固定重放

旧几何基线固定为`ee9a915d`，用git只读加载，不切换分支或覆盖工作树。
两者使用同一已知参考、当前物体记录和时钟；新版本额外读同帧实测手及过去历史。
比较是整套新调度与旧版的工程对照，不是单变量因果或机器人任务收益。

保留原4个TRAIN物体，不按新分数重新挑选；新增10Hz观测和合成手接近/丢失情形。
持续未准备、丢失控制和被拒绝的参考进度都保留，不以“所有案例完成”作为通过条件。
回放中观测不受输出goal影响，因此不能评估控制器是否能实现goal或主动恢复。

17项单测通过，包括7项新增回归。接触代理的真实假阳性/假阴性仍需用户视觉审阅。
预算CPU几何<=180秒、单run<=100MiB、空盘>=20GiB；不调用WAM，不占GPU。
实际运行Python/依赖/代码提交与设备写入manifest。

## 3D展示

只改变观测呈现：实测11点手骨架、同scale CAD表面、当前和短目标物体位姿、可旋转
3D视角；展示选择理由、手物证据、前后方案目标及时间曲线。真实记录优先，合成边界
单独标识；不生成虚构机器人手模型或模拟视频。
显示数据单独保留，原始replay.json不替换；浏览器截图检查后再提供服务。

## Limitations

预设阈值未按真实接触标签校准；表面采样密度、指尖定位、手指主动调整会影响proxy。
有界观测插值只是短时局部路径假设，低频下未观测到的绕行无法认证。
原固定里程碑容差/速度上限仍是研究变量；准备和释放等任务意图并不能从物体轨迹唯一确定。
仅验证接口和几何反馈，不声称抓取成功、可达性或WAM收益，不做新的WAM训练安排。

## 首轮反馈与修正

r1在代码c5efed58完成19片段/4967帧，21.68秒。原始输出保留。所有12个真实记录
的确认进度均为0，尽管部分后段已有随动证据：把弱距离/随动代理当成readiness硬门，
阻断了起始手尚未接近时已经实测经过的参考运动，之后无法找回原里程碑。
这不是“需要放宽距离阈值才能得高分”，而是几何进度和交互证据的语义混淆。

r2明确分开：只有显式外部readiness/failure才作硬门；弱proxy只影响goal跨度/模式，
不抹掉独立实测的有序路径证据。新增反例在旧版失败、修正后通过。
r1的10Hz分支还误用了延迟半速时序，r2改为原速度10Hz，同步修正记录；
不将r1/r2当作科学单变量对照。当前仍不能把“经过参考”解释为“操控成功”。
