# 清洗native数据到v2模型的工程接入

## 范围与设计

用户授权继续推进清洗后的训练接入。本轮只回答六个native TRAIN来源能否通过真实完整v2模型的
前向、mask归一化loss、反向和优化器更新；不据四步loss形成学习有效性结论，不启动大规模混训。

- 输入是`native3d-clean-encoder-20261015-r1`的冻结索引：427,666 TRAIN/105,946 verified DEV。
  `CleanPoseMeshWindows`和`clean_collate`直接复用该索引，保留source/profile/anchor/observed-frame元数据。
- source→source内均匀窗口抽样，概率正比有效窗口数的平方根；DexYCB当前只有subject10的351窗，
  本工程草案将它限制为1%。不加入motion strata、不制造DexYCB DEV；HO-Cap不进入已核实时钟索引。
- 归一化由全TRAIN每源充分统计按同一概率组合成一套24×XYZ hand/scene均值和标准差，floor=1mm。
  它匹配global masked-point统计，不宣称各源/各窗口实际loss贡献相等；没有额外source loss系数。
- HOI4D有效插值标签按其30Hz网格参与loss；原观测/插值位另存provenance，不冒充新观测。
- 原v2构造器及三源统计门禁保持不变。native builder用真实已有main3统计构造原架构，
  在第一次forward前显式替换为native归一化模块；native统计从未改名为main3，无全局monkey patch。
  本轮模型参数从头初始化，不是从80k parent继续训练；后续parent初始化须单独处理输出归一化变换。

## 运行边界

代码在detached worktree `tmp/native-integration-code-20261015-r1`固定后运行，无新分支。
GPU3单卡、四步、batch12、最多600秒/100MB产物、剩余空间20GiB；监控每秒GPU利用率/显存，
发现总活跃GPU超过6则退出自己的工程检查。GPU4/5/6/7的恢复由用户负责，本任务不启动或停止它们。
工程产物放主工作区的`outputs/3d-data-preparation/native-v2-smoke-20261015-r2`，不写原checkpoint。

之前临时修改v2构造接口触发既有四卡run的源码保护，在step3340停止；接口已恢复原样，
旧run和step3000 checkpoint保留。用户明确接手恢复训练。本轮后续只在隔离快照开发，
最终仅同步3d-data-preparation目录的新接入文件，避免再次修改活跃模型源码。

## 检查与结果

结果：**ENGINEERING_PASS**。代码快照`ba50a60a`，GPU3，15.5155秒完成，完整模型98,591,142参数
（geometry11,092,000，shared85,054,464），四步batch12真实更新。41项Task测试通过。

| 来源 | TRAIN窗口 | 抽样概率 |
|---|---:|---:|
| UniDexGrasp | 264,439 | 40.1878% |
| TACO | 30,568 | 13.6636% |
| HOT3D | 71,070 | 20.8341% |
| DexYCB | 351 | 1.0000% |
| H2O | 5,691 | 5.8956% |
| HOI4D | 55,547 | 18.4188% |

这些是抽样概率草案，不是正式训练效果最优配比，也不是逐源loss贡献。
[`recipe.json`](../../../../../outputs/3d-data-preparation/native-training-recipe-20261015-r1/recipe.json)与
[`horizon-stats.json`](../../../../../outputs/3d-data-preparation/native-training-recipe-20261015-r1/horizon-stats.json)
绑定同一TRAIN窗口成员、索引size/mtime、source counts和probabilities；权重或成员单独改变会被拒绝。

六源各一真实窗口的**混合coverage batch**上，hand/scene两项loss分别到PTv3、shared、hand encoder、goal的
梯度范数均为有限正数；scene loss不通过hand预测头，hand encoder经PTv3路径也有梯度。
此为混合梯度检查，不能声称每个来源单独完成了逐任务梯度归因。
将无效hand/scene目标替换为1e6，actual-model loss逐位不变。
四步中geometry/shared两个参数组的实测参数变化均约1e-4，loss/梯度有限。

四步48次随机抽样为source4/5/6/9/10分别20/6/13/2/7；source8在六源coverage/梯度batch中出现，
随机优化batch未抽中，符合1%概率下的小样本现象。没有将四步曲线解释为收敛或收益。
DEV对source4/5/6/9/10各做一个真实样本米制检查；source8无DEV不伪造，HO-Cap不参与。

PyTorch峰值allocated2,252.54MiB、reserved2,716MiB；每秒nvidia-smi实测物理显存峰值约3.0GiB，
计算阶段观测到约77%利用率。读取/初始化期间GPU空闲计入完整监控，四步不作为吞吐benchmark。
记录：[`result.json`](../../../../../outputs/3d-data-preparation/native-v2-smoke-20261015-r2/result.json)、
[`gradient-audit.json`](../../../../../outputs/3d-data-preparation/native-v2-smoke-20261015-r2/gradient-audit.json)、
[`gpu.jsonl`](../../../../../outputs/3d-data-preparation/native-v2-smoke-20261015-r2/gpu.jsonl)。

首轮r1在gradient audit前触发瞬时六卡预算保护、0次更新，保留FAILED回执，不作实现或方法负证据。
r2加强CUDA分配前的预算检查，在占用回到四卡后重跑通过。独立worktree的全仓库验证因未复制历史outputs/tmp/ref文件而报链接缺失；
最终仅将本Task的接入文件集成回主工作区，`tools/verify.py --changed`已通过；原v2模型源码diff为空。

## 后续决策边界

本轮完成新增六源到完整v2架构的工程接入，尚未把原OakInk2/GRAB/ARCTIC三源并入这个recipe。
正式长训仍需确定九源/新增源的实际实验配方、初始化与匹配对照；若继续80k parent，须明确处理输出标准化变化，
不能直接覆盖buffer后宣称物理预测保持不变。当前原始数据、清洗版和工程recipe均`training_allowed=false`。
