# 2026-10-19 执行边界修复

用户授权修复reset、指标、位置目标、checkpoint绑定和比较终止合同。既有六卡训练
继续使用自身冻结代码；此次实现仅用于新运行，不覆盖既有checkpoint或重写历史结果。

## Reset 初始化与官方 PPO

旧时序在下一次pre_physics_step里reset，然后执行从旧末态采样的动作。真实GPU检查
进一步发现，仅调用indexed setter和refresh后，手部刚体位置仍可滞后于root状态：
对单手root施加0.15m平移，未积分时palm仍没有对应平移；不能直接据此采样新动作。
检查产物：`outputs/wam-rl-policy-v2/control-boundaries-20261019-r1/refresh.json`。

corrected协议保留上游reset实现，在pending环境的下一拍用canonical零动作替换旧动作，
与官方VecTaskPython.reset初始化相同，进行一个正常物理步以得到一致的新观测。
其他环境仍执行原策略动作，各环境物理时钟正常前进，不为全场景额外插入物理步。
该初始化拍reward置零、collector done置1以切断GAE，同时transition_valid=False，
从训练minibatch、KL、优势均值/标准差、episode日志中排除。真实终止transition和原reward
保留。初始化后的第一条策略transition使用新回合观测，WAM历史只清空对应环境。

沿用固定上游PPO的actor分布、loss公式和GAE递推；collector增加valid参数，storage
过滤初始化样本。每次更新仍5 epochs×4 minibatches；有过滤时平均每批样本数会减少。
多卡KL和梯度按实际有效样本数加权，优势统计也只计有效样本。此协议与原始官方collector
有明确区别，不能称其为逐行不变的官方复现。legacy路径保留原collector。

## 目标与指标

OrderedGoalScheduler默认仍支持完整se3_pose；UniDex位置任务入口默认position_only。
位置模式按平移投影/到达/穿越，输出goal朝向取当前物体朝向，避免要求恢复初始姿态。
InteractionObserver仍观测真实SE(3)运动，空间waypoint仍不代表0.8s可达终点。

native_success保留上游sticky定义；held60_lift10cm保留历史别名，等于ever_held60。
final_held60要求终止时仍连续满足60步；drop_after_hold记录曾达到60步后任一后续失守，
即使后来恢复也保持True。此处失守指位置/抬升条件中断，不等于已确认失去物理接触。
final_task_success要求终止位置距离≤0.05m且非workspace失败；final_stable_success另要求
终止前连续60步满足抬升≥0.1m和目标距离条件。终止原因单独记录，不取消官方历史sticky值。

## 身份与比较

新PPO checkpoint内execution_identity绑定实际WAM文件摘要、config/stats摘要、每个
object-scale实际准备几何摘要、scheduler参数、goal类型、reset协议及termination_policy。
多卡保存合并后的geometry映射；评估允许选择其中子集，其他变化默认报错。旧未绑定
checkpoint必须显式给出checkpoint-migration-reason。迁移记录保存源/目标合同与差异，
用于新的run_id；不把这种warmstart描述成相同条件下无缝续训。

native复现使用native终止；WAM原有保护使用wam_workspace。对照时可显式选
common_workspace，native无需调用WAM预测就执行相同当前物体anchor下的点域检查，
两路径从startup开始使用共同终止标准。论文复现与自定义匹配对照分别报告。

## 验证

回归覆盖单环境结束时其他环境连续执行、初始化动作/样本隔离、真实terminal return保留、
位置目标旋转、保持后失守、workspace失败、legacy身份拒绝和显式迁移。
真实官方PPO+WAM仿真通过tools/audit/check_episode_boundaries.py验证，结果待本次工程
smoke完成后补充；这是接口验证，不是成功率、收敛或WAM净收益实验。
