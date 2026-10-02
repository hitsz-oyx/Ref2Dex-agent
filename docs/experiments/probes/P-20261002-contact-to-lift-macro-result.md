# HF18：接触到抬升的H10物理信息结果

按冻结[协议](P-20261002-contact-to-lift-macro.md)，slot1 `UNPROMISING`；
slot2不启动。不改变5%门槛，不追加epoch或seed。

原随机源的early cal抬升正例只有11个，先按补采协议采集669个fit/cal专用
真实H10片段，held保持原样。合并4586窗口，held early971窗口/144episode/
30初始组，形成抬升41正例/21episode/8组；cal48正例，原监督资格全部通过。

相同预算9个模型各1000更新：early held高度RMSE Cm14.476mm、state-only
15.173、action-shuffled15.208；间隙MAE5.819mm、5.927、6.078。
分别改善约4.60%/4.81%和1.81%/4.26%，未过两指标对两控制均至少5%的门。
形成抬升Brier .02598优于.02701/.02778和永不抬升基线.04222；联合支持高度
MAE5.940mm优于6.237/6.488。动作信息部分存在，不能据此认定排序、闭环
utility或稳定抓取成立，也不能把此负结果当作Cm核心思想反证。

[独立审计](P-20261002-contact-to-lift-macro-fit-audit.json)复算4586窗口的
pre输入、真实执行器目标、控制律、全部34输出标签及fit-only归一化，最大
误差均0；冻结数据/checkpoint hash一致，模型执行任务已exit0。
checkpoint SHA256：`d80db71a855d66273b3b34d93fb047d8aea821aadbfd63bf04cf27edd0b821c3`。

源采集使用GPU；进入拟合时当前执行环境CUDA不可用且nvidia-smi失败，按
[设备证据](P-20261002-contact-to-lift-device-evidence.json)在相同CPU上完成
全部9个对照，实际125.73秒，代码默认仍CUDA。含采集/资格/全部拟合/审计
666.51秒，另计失败审计1秒，共667.51秒/65.47MB。未执行新闭环或PPO。

下一步复盘接触作用表示：先用fit/cal实际转移核对日志接触力是否足以解释
物体速度增量，再决定端点力或周期冲量监督。C3仍OPEN。
