# HF16 slot2：实际执行器条件 Cm 的直接单步控制

在首轮fit前冻结补对照与执行设计。slot1原信息门PROMISING，数据39170真实
step；held已离桌6706/138ep/27组，heightRMSE4.149 vsstate4.824/shuf4.850，
CLRMAE2.271 vs2.855/2.864，jointBrier.002318优于persist.002982。时序/标签/
fit-only归一化独立审计通过。一步与旧两步/H10不是相同评价，不据此宣称旧模型
误差降低了某比例。核心问题：准确的执行器条件短预测能否改善真实局部选择？

- 冻结原物理Cm/shuffled权重，补state-only局部结果policy控制：同pre-history/
  原始力/物理场景，额外原生pre-observation；8程序头预测一步联合支持高度及
  support/loss2事件。它没有实际PD目标输入，programme索引可被当作动作条件，
  是强策略对照。3nets10601–10603、1000updates/batch256/Adam.0003/wd.0001，
  相同批次生成seed+30000，原physics 9nets不重训。
- 只用原HF15源数据fit/cal，保留9851motion/start分组；cal pre-clear数据对
  support/loss两logit做nonnegative slope/bias BCE校准200stepAdam.03，至少
  各5正/负；cal score75%绝对残差margin，下限2mm。不依赖长期V、不扫门限。
- 新五控制器cm/state_policy/shuffled/base4/fixed7，在pre-clear、连续3净力
  存在代理后p=.2随机分配完整10步owner。每一步重新观察、预测真实当前
  12独立PD目标减当前q，直接执行；rot6/7每一步重取当前rotation锚点，XYZ/
  fingers继续base/cup真实反馈。所有输入仅pre-step，联合score头避免边际
  高度×retention概率拼接。预测监督一步，验收仍是完整H10实际后果。
- 保持原已离桌状态要求中心rest+3cm/meshCLR2mm；只启用原held初始组，
  source数据旧held结果是探索选择资料，新rollout独立；不称未见正式Validation。
  96env/650tick/8window/firstepisode/6tickcooldown，完整非terminal窗口；
  所有部分/terminal使采集失败。工程seed510/private12510，科学511–522/
  private12511–12522；serial GPU0/1每phase重新admit，native<=240s/parent290s。
- 共享gain>calmargin+relative ensemble std、loss maxmember<=basepoint+2pp、
  supportpoint>=base−5pp；pre-history/force/native标准化abs>8回退base，
  motor-target OOD只否决对应候选。归一化/校准均fit/cal独立冻结。
- 保存全部preobs、history、实际PD/反馈raw/rawforce/geometry、五推荐、三
  模型分数/风险/std/OOD。固定和base影子命令仅为同真实状态参考，无反事实后果。

原判别标准保留：每policy>=96windows/24ep/8组，Cm相对base和fixed分别
>=48改变独立PD坐标的windows/12ep/8组；每完整window10次真实观察选择。
末3帧joint净力存在+meshCLR>=2mm保留正高度减起点正高度，已知p IPW
Cm减四对照均>=2mm，初始组与episode各1000bootstrap90%下界>0，geometry
失去间隙point差<=2pp、last3joint存在差>=−5pp。所有通过PROMISING；充分
支持下失败UNPROMISING，支持不足UNCLEAR。一步信息不能单独证明10–20步
预测或稳定抓取；只有实际控制门通过才进入策略学习设计/最终稳定率验收。

slot2<=60min/8GiB包括补3nets、校准、工程/所有失败尝试、12科学phase及
终态回放统计；保守另计slot1所有成本，不以新表示重置同slot预算。原HF15
失败和所有旧margin保持。输入/输出/源版本固定，C3OPEN。
