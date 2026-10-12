# ViGAR：目标注入的一手证据核验

日期：2026-10-12。范围：为 Goal-Point-WAM 的小规模注入路径实验核对 `ref1_2.md` 引用；未运行作者代码。

## 核实的实验

[论文 §5.5 / Table 2a](https://arxiv.org/html/2610.02368v1#S5.SS5) 使用真实 goal image、同一组 2,500 条 RoboTwin Clean 示范、30k 训练步，报告 50 tasks 的成功率：

| 注入方式 | Clean | Random | 平均 |
| --- | ---: | ---: | ---: |
| 无 goal | 76.58 | 21.78 | 49.18 |
| Reasoner | 78.42 | 23.52 | 50.97 |
| Generator | 78.92 | 45.42 | 62.17 |
| 两者 | 82.16 | 41.28 | 61.72 |

[论文 §4.2](https://arxiv.org/html/2610.02368v1#S4.SS2) 的 semantic route 用 vision encoder tokens 输入 reasoner；geometric route 用与当前图像相同的 VAE 编码 goal，clean goal latent 输入 generator，并加零初始化的可学习 goal 类型 embedding。后者仍使用 latent tokens，论文没有测试直接把 goal 加到每个 trajectory query。

[作者项目页](https://dagroup-pku.github.io/ViGAR/) 说明 planner 的 look-ahead 处理相邻子任务边界，end-effector ROI 的 flow loss 权重为 4:1。论文 Table 2b 的 Random 为完整 67.02、去 look-ahead 65.92、去 ROI 65.64、两者都去 52.40；这些是另一组完整系统消融，不能与 Table 2a 的 30k GT-goal 试验拼成单个因果比较。[原始表格](https://arxiv.org/html/2610.02368v1#S5.SS5)

[官方仓库](https://github.com/DAGroup-PKU/ViGAR) 有 policy、planner 源码目录及训练/评估文档；本次只核对论文及仓库入口，未验证源码复现一致性。

## 对本仓库的推断与实验边界

**推断：**目标条件的表示和接入位置值得做最小干预。论文同时改变表示（semantic/geometry）和目的分支，不能证明我们当前同一个几何 GoalEncoder 的 memory 路径必然失败，或 query 路径必然成功。

建议本地三臂保持同一 GoalEncoder、尺度、模型参数集合、初始权重、数据顺序和 loss；仅改变 goal 输入到 memory / query / 被禁用。若 query 从 pooled goal token 产生条件，明确 pooling 是本地实现选择；若同时改成刚体位移表示，就不再是单一注入位置比较。

同时评估真实 held-out trajectory EPE 与固定 scene 的 goal 干预响应。响应增大只能证明模型使用了输入，不能证明方向正确、反事实标签成立或物理可执行。三臂均无效时不能直接认定目标无信息，也不能仅据此升级百万级训练。

look-ahead 的证据针对有子任务分段的视觉 planner，不支持在我们固定 24-step 几何标签上直接重标注；ROI 针对图像流监督，也不是本仓库静止窗口比例的审计证据。
