# τ 条件归因：未来端点诊断面板

## 结果解读表（预注册）

| 观测 | 含义 | 改变哪个决定 |
|---|---|---|
| `init@B` 明显优于 `init@A` | 旧 ckpt 在“给端点”时确实更强 ⇒ **端点信息本身有用**，旧 ckpt 的弱是**训推不一致**造成的 | 确认主因=条件对齐；并给“接回合法端点”（补记录路线①）更高优先级 |
| `init@B ≈ init@A` | 端点通道对这个模型几乎没被利用 | +36~46% 主要来自**数据/继续训练** ⇒ 下一步查 curated 数据与旧子集的差异，而不是 endpoint |
| `trained@B` > `trained@A` | 新 ckpt 仍能从端点获益 ⇒ 合法端点接回后**很可能还有增益** | 把“合法 goal 通道”列为主线必做项 |
| 两格都劣于 A 条件 | 端点在场反而有害（例如被当成忽略通道/零场干扰） | 记录为“当前架构无法安全消费端点”，需改消费方式后再谈 |

以上按用户原表预注册；下文必须对它的因果措辞加以限定：B 同时恢复未来端点与旧事件，不能独立识别端点贡献；条件变化也改变 goal token 的有效 mask。旧模型从 B 获益只证明条件整体有用，不能据此认定 +36~46% 全部或主要由条件对齐造成；继续训练本身可能学会绕过条件。合法计划端点的误差分布也不等于真实未来端点，诊断收益不是部署收益保证。

**B 的端点来自 `future_flow[-1]`，是未来观测，仅作诊断，禁止作为部署合法输入或可选产品配方。** 本轮不训练、不改训练配方、不新增数据、不运行物理回放。

## 决策、最小实验和操作化

- 分类：Decision，沿用 `P-20261023-tau-fixed-panel-metric`，run_id `tau-condition-ab-20261023-r1`，branch `main`。
- 区分：旧条件恢复是否解释旧模型在 A 下的弱表现；新模型在 B 下是否仍获益。结果改变合法 goal 路线与数据/继续训练审计的相对优先级。
- 最便宜方法：读取已有 512 锚点/124 母序列，两格 A 直接引用归档 NPZ，仅增加旧/新 checkpoint 在 B 的完整前向。每个权重首批额外重放 A 以核对归档，未重新抽样。
- 旧权重：`outputs/ptv3-shared-wam-v6/flowmatching-overnight-20261022-r2/training/step011000.pt`；新权重：`tmp/t38-r2/training/step001223.pt`。新权重从旧权重 weights-only 初始化，额外 1223 updates；goal、event、curated 数据和训练预算均变化，不是“只有 goal 不同”。
- 使用现有 `configs/tau-fixed-panel-v1.json` 和 `train-anchor.json`，与 brief38/41/43 离线 τ 面板共用；brief43 的物理执行 OLD/NEW 对比仍是另一证据层，不能混合。
- B：reader 不启用 goal/event 清零策略，`object_goal_valid` 全 True，goal delta 精确等于 `scene_target[:,0,-1]`；旧 event teacher 原值，无随机 dropout。仅评估进程覆盖 checkpoint 的两个 input policy 为 None，权重 strict load、不更新。
- 批量4，seed170，flow seed172，strict FP32。指标顺序固定为前8帧手 EPE、固定接触窗手 EPE、前8帧物体 EPE；越低越好。复用原 mask/权重、124 母序列配对 cluster bootstrap 4000 次。
- “明显优于”：对应 B−A 的配对95% CI 上界<0且均值相对改善≥2%；三项不一致逐项报告。“≈”须配对差值相对 A 的95% CI 完全位于 ±2%，仅 CI 跨0不等于等效。其余列为混合或证据不足。
- 每批逐位检查所有非条件字段与原 A reader 相同；首批相同 checkpoint 在 A 下预测须与归档 bit-exact；B 首批重复预测也须 bit-exact。

## 运行前资源与复现

本机 GPU4 正运行 brief43 OLD/NEW 执行器；GPU0–3 为其他作业。计划串行使用空闲 GPU6，保留 GPU5/7 空闲，不抢占任务。每格推理上限1200s，预计约30–60s；失败不自动重启训练。

写盘前 `/home2` 约128GiB 可用；现有单格 NPZ 约11MiB，两格加报告预计<32MiB，预算上限64MiB；不复制 checkpoint、不生成 cache/视频。低于20GiB停止新增写入。输出在 `outputs/ptv3-shared-wam-v6/tau-condition-ab-20261023-r1/`。

代码入口：[`evaluate_tau_legacy_condition_panel.py`](../../tools/run/evaluate_tau_legacy_condition_panel.py)。该脚本与 brief38 实际冻结 A scorer 的差异保存在输出目录 `scorer-vs-frozen-A.diff`；评分、配对 CI 和输出数组口径不变。

运行结果与环境在后续收据补充；此段协议先于两格 B 完整评估提交。
