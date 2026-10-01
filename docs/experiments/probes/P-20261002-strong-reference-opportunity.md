# HF17 slot1：强rotation_cup之上的真实修正机会

分类Decision。区分：真实专家反馈的有界修正能否优于强rotation_cup；结果决定
是否值得拟合对应10步任务后果，最便宜方法是不训练NN的实际随机干预。
旧HF15/HF16的失败门和模型保持，不能把本轮叫做旧强控制门已经通过。

八程序：0/1/2/3/5分别cup与balanced/duck/mixed12/base/train5全命令平均；
4纯cup强参考；6仅XYZ与base平均、7仅手指与base平均。所有程序每一步
重新取当前腕部rotation锚点，执行器旋转目标为当前q，XYZ/手指重新观测反馈。
其余坐标保持专家的native command范围，完整PD耦合由环境实施。
programme只固定修正规则，不固定10步的专家命令。暂不使用Cm选择。

只从中心rest+3cm、源meshCLR>=2mm、连续3步归一化净力存在代理触发；
各初始episode最多8窗口、6步cooldown、prehistory10帧。每窗口9分配槽
均匀p=1/9，槽8重复参考4：实际参考p=2/9，其余p=1/9，重复槽分别保存。
先观察再私有抽签，执行完整10步，任何部分/reset污染使采集失败，不删失败行。
96env/650tick；冻结12科学seed531–542，私有13531–13542；工程530/13530。
不为旧held数据生成后验反事实真值，不把这批探索称正式Validation。

沿用9851/motion/start的SHA分组fit<50/cal<70/held>=70，整个初始组分开。
fit上每修正和参考>=24窗口/8ep/4组才有资格；在fit选择一个最高IPW
联合保留支持高度的修正，在held验收，不使用held选最好修正。
held两边各>=48窗口/12ep/8组；原始重复参考槽4与8也各>=24窗口/8ep/4组。
主指标末3帧joint净力存在且CLR>=2mm的正高度最小值减起点正高度，单位mm。
重复参考随机IPW差形成null；主候选收益>=max(2mm,2*abs(null差))，初始组和
episode各1000聚类bootstrap90%下界>0，CLR任一步损失point差<=2pp，
末3帧joint存在point差>=−5pp。所有满足PROMISING；支持充分失败UNPROMISING；
支持不足UNCLEAR。不追加seed、扫描混合比例、改metric/horizon追门。

单GPU0/1每phase重新admit，native<=240s,parent<=290s，slot1<=60min/8GiB
包含工程与失败重试/独立PD和专家反馈/原始力/源mesh重放及统计。所有输入
hash固定，checkpoint/专家不训练，不覆写输出。保存逐步preobs、反馈命令、
真实PD、实际力、全几何、全部后果及allocation；独立审计通过才接受结果。
此为机会Probe，净力仅presence代理、source-plane是VHACD近似；C3仍OPEN，
不直接进行最终成功率比较。正向才设计新Cm的H10任务物理后果与强参考修正选择。
