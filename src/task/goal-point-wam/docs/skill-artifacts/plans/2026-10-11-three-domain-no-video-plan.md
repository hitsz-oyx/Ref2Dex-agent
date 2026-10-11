# 三域无视频训练实现计划

本计划对应已审阅的 `2026-10-11-three-domain-no-video-design.md`。实现全部留在
`goal-point-wam` Task，复用 `cm-pointflow-effect-pretrain` 的数据、PTv3 vendor 和
刚体 effect 合同，不改历史 manifest、stats 或 checkpoint。

## 顺序与验收

1. **数据 adapter 与 collate**
   - 将 `MixedWindows` 样本适配为双手 22 点的 start/end goal、24 步 executed action、
     masked trajectory target 和 source id。
   - 逐 source 读取 train/val 样本，检查形状、有限值、时间/单位合同和缺失手 mask。
   - 验收：CPU contract test 通过，三源各至少一个 moving sample 可采样。

2. **goal-blind scene/world/trajectory 模型**
   - 新建 scene-only PTv3-small encoder；固定 workspace voxel，不把 goal/action 放入
     scene sparse input。
   - 新建 goal set encoder、trajectory query decoder、executed-action adapter 和 world
     decoder；world forward 只依赖 scene tokens、executed action、object tokens。
   - 验收：随机 batch forward 有限，goal 替换不改变 world 输出，两个 loss/head 都有
     梯度。

3. **单 batch / GPU engineering smoke**
   - 保存完整输入 identity、配置和参数 hash；运行少量 update，保存并精确 replay
     checkpoint。
   - 记录 GPU 利用率、allocated/reserved 显存、update 时间；遇到 OOM 或非有限值停止。
   - 验收：结果标记 engineering-only，不能替代科学 Probe。

4. **三域短 Probe**
   - 同一配置使用 OakInk2/GRAB/ARCTIC 的 fixed balanced validation panel，报告每源
     trajectory EPE、world point EPE、moving/static 分层和 goal/world isolation audit。
   - 输出存放 `outputs/goal-point-wam/<run_id>/`，不覆盖旧产物。
   - 验收：根据结果标记 PROMISING、UNPROMISING 或 UNCLEAR，并写入 Task 状态/实验卡。

## 明确停止条件

- 三域 manifest/hash、source split 或数据 shape 漂移；
- 无有效 moving sample、future/goal mask 泄漏、world 对 goal 敏感；
- loss/gradient/metric 非有限、GPU OOM、外部 GPU 进程冲突或超 Campaign 预算；
- 短 Probe 不能改变下一步决策时停止追加训练，不以单次 loss 形成正式结论。
