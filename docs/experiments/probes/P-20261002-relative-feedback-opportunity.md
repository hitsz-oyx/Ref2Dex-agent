# HF24：当前相对反馈程序的真实候选机会

experiment_id: P-20261002-relative-feedback-opportunity
family: HF24
probe_index_in_family: 1
kind: Decision / randomized executable-candidate opportunity Probe
status: ACTIVE

区分：保留专家抬升并响应手物相对运动，是否提供超过base/Cup重复噪声的候选
优势；若没有，不用Cm拟合解释不存在的机会。通过后才学与新programme对应的
接触支持抬升/接触loss/掉落物理后果，检验state-only/shuffled/base/best-fixed，
然后直接执行/重新规划；这些步骤与策略训练/最终成功率尚未完成。
最便宜办法是在已有六专家基座上随机执行固定反馈，暂不训练网络或做成功率矩阵。

实现按[反馈职责决定](../../decisions/D-20261002-contact-relative-feedback.md)，
[排除工程](P-20261002-contact-relative-feedback-engineering-result.md)已通过。
固定gain0.5，option0反馈base4，option1强rotation-Cup1，option2相对速度反馈，
option3相对位置反馈；后3项保持起始旋转、鲜活Cup XYZ/手指。相对程序每步
读取当前state，并使用片段开始相对锚点；位置程序第一步等Cup，不按初始PD
相同排除。旧固定权重H10 Cm不适用于此职责，采集不使用它。

独立seed671–676，assignment=seed+17000；排除工程669及旧623/590不进入
科学/训练。96env/max300tick，first episode每early/clear最多2个H10窗口，
currentjoint连续3步、rest>=5mm、warmup10、episode剩余>11、每片段后6tick
冷却，无reset污染；外部动作base。六slot均匀独立抽样[0,0,1,1,2,3]，实际
概率base/Cup/velocity/position=1/3,1/3,1/6,1/6，两个base及两个Cup槽程序相同，
提供随机分配零差，不能当同状态精确重放。选择前只读取当前信息，不过滤后果。

标签：H10末3步joint force/mg>.1且meshCLR>=2mm，取最小正高度减起始正高度
为支持抬升mm；任一步hand/object presence丢失为contactloss，末3步共同存在
为joint，曾CLR>=2mm后降至<2mm为geoloss。初始clear(height>=30mm且CLR>=2mm)
另报掉落支持与差值。力是presence proxy，不是识别手物接触对。H10不等于最终
稳定抓取45tick保持/后续掉落。所有实际PD/force/几何及前后native观测独立审计。

motion/start组按SHA13671固定bucket，fit<50、cal50–74、held>=75。fit只按实际
支持高度的Hájek IPW ratio选择两反馈候选中的较好者；各fit>=24窗口/8episode/
4组才可选；best-fixed在全部4臂中用相同资格选出。候选选择不使用cal/held，
后两者不参与更新gain或源采集；cal描述性报告，primary仅held。
主要估计器固定Hájek IPW ratio（恒定臂概率时等于各实际臂样本均值），原始HT
并列报告、不事后切换。分别以完整episode和motion/start组为单位bootstrap1000
次90%区间，种子15671/15672。primary是fit选定反馈相对强Cup与base，另列
两候选及fit选定best-fixed，但不据held后选赢家。

支持门：held>=128窗口/32episodes/8组，selected/Cup/base各>=24窗口/
12episodes/8组；重复base slot0/1和Cup slot2/3各至少8窗口。primary支持高度
相对Cup>=max(2mm,2abs(held重复base高度零差),2abs(held重复Cup高度零差))，
相对base>=2mm；两种对照group/episode90%下界都>0。相对Cup的contactloss
点差<=+2pp、geoloss<=+2pp、terminaljoint>=-2pp；这是Probe筛查，不是安全证明。
actual选定程序至少12窗口改变同观测Cup PD；保留真实programme动作幅度与覆盖。
初始clear比较若双方不足20实际窗口，仅写UNCLEAR掉落证据，不称零风险，后续
闭环效用需补独立初始抬升与掉落支持。primary通过为PROMISING，支持不足UNCLEAR，
有支持但原门失败UNPROMISING；全部原门不改，不追加seed或扫描gain追门。

总预算3600秒/2GiB，包含新工程162.771秒（其失败/修正已计入），科学准备
预留60秒、分析/审计180秒；source/audit预算按剩余时间受控。默认单空闲GPU5，
每seed native内部240/parent300秒、audit parent90秒；最迟总3300秒停止新任务，
留下分析成本。旧共享源/专家训练另报，不覆盖任何checkpoint/产物。
input/hash/运行/资源失败立即保留并停止，不静默重启。任务CPU只作统计/文件，
模型推理与仿真/mesh批处理GPU。当前会话直接执行，无子代理，Mission/claim不变。
