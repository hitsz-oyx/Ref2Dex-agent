# Scheduler → WAMv4 → Shadow BC/PPO 控制接口

依据用户提供的c0f5907审计，按当前实现复核。该文档的“尚未接入”描述已部分过时，
任务阶段、可观测性和目标时间语义方面的限制仍然成立。接口不代表物理执行已经成功。

## 当前接口

入口 `src/wam_rl_policy/control_pipeline.py:ScheduledShadowControl`。
`control_contracts.py` 定义测量、任务意图、外部执行反馈、goal和tau元数据。

| 边界 | 当前合同 | 必须避免的误用 |
|---|---|---|
| TaskIntent | task_id + phase枚举；调用方给任务意图 | phase目前不进入WAM/actor张量，不能宣称支持release条件 |
| MeasuredShadowState | episode/tick/time；current300；world-from-object SE(3)；world hand11 + mask | Inspire不能直接冒充Shadow；缺失手点不能补零 |
| ExecutionFeedback | readiness/lost_control/task_success + evidence source | observer启发式、predicted C、路径进度不等于外部抓取证明 |
| Scheduler goal | 当前anchor下32个固定query的SE(3)位移；空间路径waypoint | forecast窗口0.8s不表示waypoint就是参考轨迹t+0.8s终点 |
| WAMv4 | H4=当前及过去2/4/6个60Hz测量；512/32/128；24×30Hz | 不复用旧v3 batched_inputs，不用未来状态构造历史 |
| Executor tau | 右手8×11×3；世界系；相对当前手点的位移；米；时间1/60…8/60s | 不是逐步增量、关节角，也不是直接取WAM前8帧 |
| Native action | canonical24；前6腕部、后18手指；clamp[-1,1]后交模拟器独立副本 | 不是q_des；上游坐标变换后的action不能作canonical prefix |
| PPO rollout | raw sampled action + 对应Normal log_prob；执行另存clipped action | 不能用裁剪动作重算旧采样概率 |

Shadow手点顺序来自 `video-data-preparation/tools/run/replay_unidexgrasp.py`：
palm；thumb/index/middle/ring/little bases；同序五个tips。
末端从distal body加既有tip offsets。这个映射不证明Inspire阈值/点位可直接通用。

## 使用顺序

```python
actor = BCWarmStartActor(scheduled_bc_packet).to(device)
controller = ScheduledShadowControl(runtime, scheduler, actor, TaskIntent("lift"))
controller.reset(episode_id)
# 每个60Hz tick从模拟器读取，不从离线teacher未来取值
state = MeasuredShadowState(episode_id, tick, sim_time, current300,
                            object_pose_world, hand11_world, hand_valid)
command = controller.compute(state, feedback=ExecutionFeedback(), sample=True)
controller.execute(command, sim_step)
# PPO保存command.raw_action / action_log_prob / executor_observation及真实reward/done
# simulator完成一步后重新测量，再调用compute
```

前6拍tau为零、history-valid=0，current300仍然提供给原BC；第7拍起使用真实H4。
每条command仅能执行一次，旧episode、重复/跳跃tick、非60Hz测量、过期command会报错。
`execute`只消费内部保存的canonical副本，外部日志不受模拟器原地变换影响。
TIMEOUT要求调用方显式处理恢复/reset；COMPLETE只表示路径完成，不擅自宣布任务成功。
WAM预测C仍作为诊断/后续可选条件返回，当前565维actor不消费C或phase。

## 审计项逐条处置

- Scheduler几何milestone不是任务阶段：增加`progress_semantics=geometric_path_only`、
  `readiness_source`；控制输出分离`path_complete`与外部`task_success`。
  保留已有`arrival_witness`/`milestone_witnesses`，不改几何调度数值。
- horizon语义：显式声明`spatial_path_waypoint`、`reference_endpoint_time_s=None`、
  `reachability_certified=False`；记录训练condition为observed future endpoint，部署存在分布差异。
  当前桥拒绝将另一种goal语义静默代入；未来time-reference goal需独立实现并验证。
- v4接线：已有`V4Runtime`完整512/32/128输入、`ScheduledCurrentExecutorInputs`消费调度goal_delta；
  本次补正式控制入口，旧`fast_bc.FeatureController`不自动升级，也不假装所有旧runner已迁移。
- τ时钟/坐标：已有`tau_contract.py`显式补τ(0)=0、30→60Hz插值、选择右手、旋转到world。
  新增packet的frame/unit/layout/time/mask语义。valid仅说明历史支持，不保证未来正确。
- action prefix：既有`action_contract.py`和native runner已clone隔离；新入口同样隔离，
  不以静态旧版本推翻历史实验，需要各自manifest/vendor证据。
- GT-τ结果：仍只能说明teacher诱导轨迹下的条件执行；不证明任意轨迹执行、跨物体稳定性或WAM收益。

## 验证与下一步

接口测试覆盖：真实scheduler→模拟WAM→actor→原地修改sink、startup、readiness硬门控、
旋转/时钟转换、PPO原始概率回放、手点缺测/robot mismatch/过期动作拒绝。
`tools/audit/check_control_pipeline.py`进一步用真实WAM和已有全量BC权重走12条归档TRAIN测量；
它不是native物理rollout。随后需把该入口接入565维native runner，检查真实状态反馈，
再接PPO rollout/reward/done；缓存完成和离线BC误差不能替代这一验证。

不在本次实现内：通用Inspire executor、phase-conditioned WAM、接触力证书、
任意参考轨迹的可达性保证、完整任务阶段状态机。接口已留出任务意图与外部反馈位置，
但这些能力必须分别实现和验证。

实际接口smoke已通过：WAM step12000 + 全量BC step50000，12条连续归档TRAIN测量，
前6拍valid=False、随后6拍True；迁移后actor均值最大差2.384e-7，PPO log-prob回放误差0，
模拟原地修改sink不能改写canonical日志。耗时12.30s，峰值417MiB。
产物 `outputs/wam-rl-policy/full-scheduled-current-bc-20261018-local-r1/control-interface-smoke/`。
另有25项相关接口/调度/动作/τ/分片测试通过。这些不是native物理成功证据。
