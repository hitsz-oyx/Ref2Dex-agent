# Ref1 可复用代码与现有数据核对

核对日期：2026-10-07。代码保留不等于历史原始输出仍然存在。
根级[恢复边界](../../../../docs/STATE.md)记录旧baseline目录删除后的资产丢失。

| 来源 | 可复用部分 | 不能直接继承 |
| --- | --- | --- |
| `third_party/DExplore/dexplore/evaluate_object_router.py` | 六自训练expert的原生player、逐模型RMS、对象/观测路由 | checkpoint目前缺失；不可改用官方actor冒充原专家 |
| `cm-interaction-oracle/tools/run/collect_interventions.py` | 连续完整episode、真实动作/PD目标/物体状态记录、同步reset、输入hash | 固定airplane/source_e260、旧4/8/16步干预与局部Y不是ref1新合同 |
| `cm-interaction-oracle/src/execution_geometry.py` | 当前object frame下FK/实际hand几何检查 | 实际未来hand-flow是事后量，不等价于可提前生成的24步proposal |
| `CmResidual/physical_value_contract.py` | sustained hold、后续drop的物理统计 | 不复用其旧value/reward/Y标签作为Robometer偏好 |
| `collect_oracle_y_candidates.py`、rolling候选包 | 合同审计经验、动作/状态一致性检查 | forked candidates不当作continuous rollout训练源 |
| `native3d-r6`（GRAB/ARCTIC） | 26,755个已审计hand/object运动窗口，可供WM预训练 | 没有本任务robot quality/progress/preference监督，不能直接训练evaluator |
| EgoDex官方test工程样例 | 几何/时钟工程检查 | 禁止训练/模型选择；估计pose也不是本任务GT仿真数据 |

本地 `outputs/cm-interaction-oracle/` 不存在；六专家路由配置中的六个checkpoint以及
`outputs/CmResidual/agent_contact_option_airplane_motions` 不存在。
已经读取外部 `/home2/wyy/oyx_ws/Ref2Dex` 与 `Ref2Dex-weights` 的相关文件清单，
外部DExplore输出有更早的checkpoint/manifest，但未找到固定六专家权重。
不能把这些不同身份的旧训练checkpoint自动替代成相同专家。

外部 `Ref2Dex/third_party/DExplore/dexplore/data/assets/` 有 Inspire URDF 和 airplane URDF；
它保持read-only。恢复本地采集时可引用或软链接，但先核对模型、motion、策略观测合同及依赖。
未复制/修改外部资产，未重启任何旧训练。

运行只读预检：

```bash
python src/task/consequence-evaluator/tools/audit/preflight.py \
  --output outputs/consequence-evaluator/<preflight-run>/inputs.json
```

如备份按原相对路径保存，可用 `--asset-root <backup-root>` 检查。
预检只检查可用性并保存期待SHA，文件存在后仍需实际hash/原生加载与数据语义核验。
machine-local预检输出已保留在仓库tmp，GPU资源状态只代表检查时刻。

下一步先接上有身份的备份，核对是否包含策略原始H、真实执行A24、连续object poses和可信
成功/失败事件；缺字段时只补采必需数据。不得从实验卡摘要补造样本，或把整体结局标到所有窗口。
