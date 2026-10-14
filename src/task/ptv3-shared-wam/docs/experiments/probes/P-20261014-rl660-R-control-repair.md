---
schema: ref2dex.probe.v2
probe_id: P-20261014-rl660-R-control-repair
experiment_id: P-20261014-rl660-R-control-repair
date: 2026-10-14
task: ptv3-shared-wam
branch: main
git_commit: db27db95
claim_id: C1
hypothesis_family: HF-rl660-R-control-repair
probe_index_in_family: 1
seed_pool: probe
seeds: [140, 141, 142]
decision_changed_if_positive: retain first-control objective and qualify R with matched native execution before new joint training
decision_changed_if_negative: distinguish optimization capacity and input coverage before expanding R or adding object conditioning
status: PROMISING
run_id: rl660-R-control-repair-20261014-r1
---

# RL660 GT-τ R-only：控制目标修复与可拟合性Probe

Result: PROMISING（局部离线控制目标）。matched A/B的dev第一控制XYZ mean/p95为50.78/161.68→45.80/148.05mm，clipping=0；仍未到可执行精度。256窗first-only拟合到8.64mm，但dev63.30mm，不能声称R已经修好。
Decision: 保留control-aware配方与独立checkpoint选择；现有R小集合可拟合，后续优先解决覆盖/优化和startup，再过K1 GT执行门，不直接扩R或增加物体输入。

## Decision Note

- 服务于自训练执行底座。ref2审计已确认all8等权、总objective选点及native数据接口问题，
  但未证明它们解释全部52–54mm偏差。该Decision Probe检验改控制目标能否在固定输入下
  改善第一控制，结果决定是否把修复配方带回联合训练；不测试WAM或Cm收益。
- 最便宜方法：保留正在跑的联合训练作为基线，在detached隔离工作树实现修复，避免触发
  活跃源码漂移保护。不建新分支。R从同一现存robot checkpoint暖启动，三臂optimizer均
  重置；不更新forecaster。code身份在启动回执记录。
- 六卡预算当前已满（本地5/7、远端0/2及既有1/3）。CPU工程基准128 batch、4线程约
  0.039秒/update，因此此小R诊断使用CPU FP32，避免中断GPU训练或等待约3小时。
  这不是默认CPU路线；三臂同runtime，不将其和旧GPU联合loss直接作因果比较。
- 总墙钟上限1200秒、产物100MB，4 CPU线程；全球六卡/400GB边界不变。不做GPU仿真，
  不触碰未知任务。超时/非有限/标签不符/超预算停本次诊断，保留已有曲线与候选。

## 固定协议

- source：`outputs/ptv3-shared-wam/rl660-corpus-20261014-r2`，保持589/71完整序列split、
  10cm整窗准入与父模型TRAIN-only state/τ/action normalizers。
- model：现有ChunkRetargeter128D、2层decoder；A/B完整8×12输出，C只保留第一个
  action query及对应输出normalizer，产生1×12。A/B参数与初值完全相同；C保留其余
  共享参数，查询数变化使C是结构诊断，不能单独视为纯loss消融。
- 共同拟合池：按原sampler均匀序列、startup/later各半抽8192窗（允许重复），draw seed141。
  三臂相同池、batch128、同一有放回batch流；AdamW lr3e-4、wd.01、clip1，2000更新。
  固定TRAIN报告512窗来自拟合池，panel seed142；dev使用原联合训练保存的512索引。
  model seed140，parent R覆盖初始随机权重。拟合池/评估索引保存以便复算。
- A：all8 standardized L1。B：历史 `4*first + mean(tail7) + PD0`。
  C：first-only standardized L1 + PD0。PD0为xyz/.01m、rotation/.1rad、finger/.1rad
  三组逐坐标MAE等权平均；训练raw输出保留越界梯度，报告先clamp并记录clipping。
- step0/500/1000/1500/2000报告TRAIN/dev的first xyz mean/p95、rotation/finger p95、
  clipping；startup(<16)/later分组；手点距采样CAD表面<=20mm只作为near几何代理。
  A/B同时报告tail7误差。候选均按dev first XYZ p95、再raw first MAE选，权重独立保存；
  不以不同目标的total loss数值互比，初始step0也可成为best。

## 预定判断与限制

先看TRAIN能否显著降低first PD误差，再看dev。若B相对A的dev mean及p95均降低且
clipping<1%，记局部PROMISING；若仅TRAIN改善，认为泛化/覆盖仍UNCLEAR，不直接扩训。
C能在TRAIN接近5–10mm但dev仍30–50mm时优先查覆盖；C在TRAIN仍高时先核对优化/容量，
不能直接证明输入不充分或inverse不确定。若B/C均无实质改善，则固定预算停止，不扫参。
这些量级是排查指引，不是正式统计显著性或最终抓取通过门。

本Probe改变了联合更新与单独更新的优化条件；只允许同组三臂作matched比较，不能把
相对parent的改善完全归因loss。单seed、8192抽样池，未覆盖全部TRAIN窗口，也不代表
未见物体泛化。native GT执行、预测τ的闭环与Cm因果增益均延后；修复接口不等于物理验证。

## 运行与产物

执行工具：隔离工作树中的 `src/task/ptv3-shared-wam/tools/run/probe_rl660_R.py`。
产物：`outputs/ptv3-shared-wam/rl660-R-control-repair-20261014-r1/`。
隔离工作树：`tmp/rl660-ref2-fix-worktree`，修复提交在验证后记录，不热改旧训练源码。

## Decision Note：固定256窗拟合检查

三臂2000更新已完成。B相对A的dev mean/p95约改善9.8%/8.4%，但B训练报告仍
49.8mm（selected1000），C最终TRAIN也有38.7mm，不能把剩余偏差直接归因输入不充分。
最便宜判别：从既有固定TRAIN报告窗取前256，使用同一parent R完整8-query结构，
只监督第一步（standardized L1 + PD0），固定2000更新、batch64、AdamW3e-4/clip1，
CPU4线程、300秒/20MB上限。直接测试局部可拟合性，不改输入/模型宽度/物体条件。
run_id=`rl660-R-memorization-20261014-r1`；沿用140/141/142子流，不作为独立seed。
若能低于10mm，只说明该小集合可拟合，优先查扩大覆盖后的优化；若仍高，预算到期
保留结果，继续区分优化与容量，不以该负结果证明inverse不确定。dev只作旁观，按固定
最后一步报告，不用dev挑选memorization权重。此阶段加前阶段仍远小于1200秒CPU预算。
原C移除7个query改变了初始函数，初始dev154.2mm而A/B为52.5mm；C是结构诊断，
不能用它与A/B的差异单独归因first-only loss。本小检查保留8个query来避免该混杂。

## 结果与独立核验

- 第一阶段实际代码db27db95，parent robot step11000（原代码b22978c1），三臂共178.07秒，
  CPU4线程、FP32、2000更新/臂。拟合池8192次抽样对应7304个不同TRAIN窗口。
  A/B的初始state_dict逐张量完全一致，初始dev mean/p95=52.50/176.01mm。
- 小样本检查实际代码1d1bb08a，共53.31秒；同一parent完整8-query R，256窗startup/later
  各128，固定2000更新、batch64、只监督第一步。两阶段合计231.37秒CPU运行，无新增GPU。
- 所有被选中的R-only权重独立保存，不覆盖原WAM/R；候选别名指向各自不可变step文件。
  R-only范围内共同按dev first XYZ p95再raw first MAE选点，三臂都用完相同更新预算。

| 分支 | 选中step | TRAIN XYZ mean / mm | dev XYZ mean / mm | dev XYZ p95 / mm | dev clipping |
| --- | ---: | ---: | ---: | ---: | ---: |
| A：all8 | 2000 | 59.04 | 50.78 | 161.68 | 0 |
| B：4first+tail+PD0 | 1000 | 49.84 | 45.80 | 148.05 | 0 |
| C：one-query first-only | 500 | 54.43 | 48.42 | 153.83 | 0 |

B相对A的dev mean/p95降低9.8%/8.4%，达到本卡局部PROMISING门，但绝对误差仍大。
B的dev rotation p95=.3711rad，A=.3672rad，未改善；finger p95=.2505rad，A=.2750rad。
B的dev tail standardized L1=.3702，A=.3524，有预期的后续chunk精度代价，禁止据此
直接声称K4执行可用。B最终2000的TRAIN mean43.26mm，但dev mean47.37mm、p95152.42mm，
因此1000仍是本面板候选，不能靠训练loss或末步直接选点。

C移除query使其初始dev mean154.17mm，与A/B函数不等价；其最终TRAIN mean38.73mm，
dev mean46.10mm/p95163.90mm。C说明该单query结构可以学习，不能作为纯loss消融，
也未提供比B更好的当前dev p95，不据此关闭first-only控制路线。

### 起步仍是明显问题

固定dev只有39个startup窗口。A/B startup mean=102.86/99.19mm，p95=317.18/327.44mm；
后段mean=46.49/41.40mm。整体改善主要不足以消除起步长尾。B的near-surface20mm组
mean40.84mm，far组54.48mm；这是采样表面几何距离代理，不是接触标签，也不支持
直接把高误差归因contact状态未观测。后续须保留这些分组，不用整体均值掩盖起步。

### 256窗能拟合，但不能泛化

固定最后2000步：TRAIN mean/p95从65.85/228.57降到8.64/16.40mm，startup mean8.42、
later8.86；rotation/finger p95=.0607/.0625rad。TRAIN clipping=1.56%，仍非执行合格门。
原dev mean/p95变成63.30/203.55mm（初始52.50/176.01），不能把memorization权重用于
执行。它说明现有宽度/输入下这个小集合可拟合，不能证明全数据inverse唯一，也不能
证明模型全局容量充分；下一优先级是扩大数据覆盖后的优化与泛化，而非凭训练误差直接
增加hidden state或object token。本次固定预算到此停止，不启动超参扫描或新的大规模联合训练。

### 工程复算

工具 `audit_R_repair.py` 从原corpus的admitted_ticks重建TRAIN/dev成员、独立NumPy
state/τ与原始PD targets，再用原生finger半scale/耦合转换复算first PD指标。A/B初值
完全一致、三臂dev选点一致；各mean/p95复算最大XYZ差约0.000101mm，角度差<=2.1e-7rad，
clipping完全一致。报告 `rl660-R-control-repair-20261014-r1/independent-audit.json` 为pass。

## 修复实现、部署与尚未验证的边界

修复提交链：db27db95（loss/选择/保存与R-only工具）、1d1bb08a（RL660原生输入）、
9a9b9ad1（候选R合同、预检和独立复算）。隔离工作树的56项Task测试通过。

- 新robot训练独立保存forecast_best/control_best与总best/latest；任一改善即保存完整
  checkpoint，清理保护所有别名，resume不把缺权重的历史分数当可用候选。
- 第一控制loss只由显式 `action_objective=control-aware-first.v1` 启用。旧all8配置不被
  静默改变。新模板为8GB run预算（多份完整101.6M恢复点），受全局400GB限制，当前
  没有启动该新联合配方；不能把它当作本次R-only100MB预算的扩大。
- RL660显式指定sequence，绑定同一record的CAD/reference/hand oracle与原始几何motion；
  修复server/native/audit旧单文件读取。环境实际motion/object身份和双方候选R路径须一致，
  完整24步未来不可越界，禁止补齐尾部假oracle。旧单文件入口仍兼容。
- 支持 `--retargeter B_control_best.pt`，不复制整个WAM；核对原corpus、PD单位、TRAIN
  normalizers、8-query模型结构。修复R强制显式K1执行；one-query C与256窗过拟合权重
  不冒充合格候选。真实5条dev序列输入回放通过，实际B候选的native launcher CPU预检
  为PREFLIGHT_PASS，报告在 `rl660-ref2-audit-20261014-r1/native-launch-preflight.json`。
- 当前无可用共享GPU额度，未运行新接口的Isaac/GPU物理smoke，更没有GT抓取通过结论。
  后续先做有界工程执行，再做K1 GT门，才决定是否值得把修复带入更长联合训练。

旧联合训练仍在使用原固定代码，远端v2不变。为避免源码保护停止旧进程，源码补丁
保存在 `tmp/rl660-ref2-deployment/repair.patch`（17文件、tip9a9b9ad1）。守护PID4007948
等待旧训练及相关peer退出，最多4小时；检查目标无其他工作区/index修改、补丁可干净
应用后才写入主工作树，不自动提交或覆盖用户修改。状态为该目录 `status.json`；冲突
则保留隔离修复并记录停止原因。当前修复可在隔离工作树审阅，主工作树活跃源码尚未热改。
