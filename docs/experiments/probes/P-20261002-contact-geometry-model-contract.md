# HF19 物理模型实现合同

主卡：[信息与机会门](P-20261002-contact-geometry-action-information.md)。本合同在
科学拟合前固定，采用已登记的每指几何/动作与密集一步+已知H10律监督。
工程seed570只用于实现审计，不进科学训练。科学held不调结构、loss或门。

掌与五指节点顺序0/3/6/9/12/15；各自q/qdot/PD绑定全18原生自由度的
0:6、6:8、8:10、10:12、12:14、14:18，补齐到6维。节点输入24维为
物体坐标相对位置3、相对线速度3、q6、qdot6、节点one-hot6；动作12维为
真实PD目标-q6及该块六专家权重6。旋转锚定标志另外1维。指端是link原点。
history10×69由每步pre-state49、当前存在bits2和上一步已执行raw动作18构成。
physical为每个手部body及物体的当前rawforce/(实际mg)的signed-log1p，
加当前mesh间隙、相对rest高度。没有post量进入输入或归一化。

模型historyGRU64、physical64、共享nodeMLP64×2，节点均值池化；拼接193维，
256/128 trunk；共享节点头(节点64+trunk128)→128→6，global128→16。
输出52维，先36为六节点各posdelta3/.005m、veldelta3/.1m/s；36:39为
物体dv/.1m/s，39:42为dx/.005m，42为mesh delta/.002m，43/44为下一步
手/物存在logit。45为H10末三步最低z-prez/.01m，46为末三步最低CLR-preCLR
/.002m；47/48为末三步手/物存在保持logit；49为曾clear后几何损失logit；
50为末三步同时存在/CLR>=2mm且rest高度>=3cm的logit；51为相同联合支持
下的绝对rest高度/.01m。控制score是51×10mm减当前rest高度mm。
H10仅在初始pre-state监督；一步在所有10个真实pre/post监督。

固定状态kinematic skip：节点pos用(v_relative-omega_object×pos_relative)*dt，
节点vel/dv用零变化，物体dx与CLR用当前v*dt；H10用当前vz常速延伸，
二值用当前状态的log(9) logits。skip没有动作或未来标签。
loss对节点pos/vel、物体dv/dx、CLR、下一步存在、H10高度/CLR、H10二值、
H10联合高度共9项等权；连续smoothL1，二值BCE。direct_score仅训练51，
其余输出严格保持状态prior，风险判据因此相同地检查当前状态风险，不能把
未训练风险输出解释成直接评分策略的风险预测。它共享候选、参考、增益/std门。

cm/state_only/shuffled/direct_score四条件，12601–12603，每模型1000更新。
每batch128个均匀初始窗口+128个均匀实际转移，所有条件相同。shuffled只在
fit内按当前early/clear与step分层，把node_action和law一起固定打乱；不打乱
当前状态/history或label。state_only屏蔽已归一化的动作/控制律。归一化只用
fit转移；node_state/action跨样本和节点求均值/std，history跨样本/时间；
std最低.001，标准化clip8。target与skip不做fit统计归一化。

一步metric按全部held转移，H10按held起点；支持/信息门沿用主卡。候选只用
初始pre和已保存的八个可执行programme首步PD、系数；actual候选输入必须
与该窗口实际初始输入相等。score/std/风险谓词固定，未执行后果不当真值。
拟合产物记录全部source/model/card hash、更新、GPU、fit-only归一化与原门；
独立复算全部标签/时序/actual候选和归一化后才接受结果。
