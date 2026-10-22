# S2d：原逐帧消费的 WAM 与 live-object GT 对照

执行前协议（Decision；2026-10-22）：区分“物体空间错位足以解释旧GT差”与“该修正仍不足”。
结果决定先处理GT锚定还是保留消费器/参考相位问题；最便宜为固定已见初态两链各5次，0训练。
本轮独立experiment_id=P-20261022-s2d-object-anchored-controls；run_id=s2d-controls-20261022-r1；
基点2851060a，detached worktree，不建分支；运行实现提交写入manifest。

- a=原R+WAM11000逐帧τ；c=同一原R+live-object GT。R、shape、scheduler-v2兼容快照、
  native reset/PD/安全阈值均复用当前已有a/b/c测量入口的冻结方式。当前生产executor不修改。
- 同s1_cylindersmall_pass_1、seed195/196、env1、30Hz；5个block是相同初态重复而非5任务。
  每次reset检查9字段与旧reference误差≤1e-6；ac/ca/ac/ca/ac顺序。
- 两链各≤1500controls（每例300），总≤3000、1800s、2GiB；首次窄成功或原安全gate即停。
  成功定义保留5cm+真实≥2指0.1N+耦合+离桌elapsed1s。1s是截断下界。GPU6仿真/5模型。
- c使用S2b真实205状态参考，t=3..196有完整8未来；无未来时invalid/零τ，不复制终帧。
  相位保持同30Hz时钟，未另加手几何相位匹配。目标G[k]=T_obj_live inv(T_obj_ref[t]) H_ref[t+k]，
  τ[k]=G[k]−H_live。所有未来共享参考当前物体锚点，保留参考未来物体抬升。
  G不依赖live hand，live hand只参与既有差分编码、R状态和encoder可用gate。
- 预定义描述性明显：c比a至少多2/5成功；其余报告同量级/更差。n5不构成统计证明。
  c提升只支持此“完整参考+物体重锚定”组合，不能单独识别空间错位；同量级也不能自动
  宣称参考质量是唯一瓶颈，时钟相位与R条件适配仍是混杂。c须物理实测后才谈上界资格。

复现命令、完整数值、反例检查与JSON将在此页追加。原始trace只存worktree的outputs。
