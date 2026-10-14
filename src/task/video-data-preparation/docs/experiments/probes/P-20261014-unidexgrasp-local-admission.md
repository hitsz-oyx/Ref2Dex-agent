---
schema: ref2dex.probe.v2
probe_id: P-20261014-unidexgrasp-local-admission
experiment_id: P-20261014-unidexgrasp-local-admission
date: 2026-10-14
task: video-data-preparation
branch: main
git_commit: c06fc665
claim_id: C1
hypothesis_family: HF-unidexgrasp-rigid-admission
probe_index_in_family: 1
seed_pool: probe
seeds: []
decision_changed_if_positive: use native rigid geometry and explicit ShadowHand FK, then assess rollout acquisition
decision_changed_if_negative: resolve asset scale, pose or kinematic semantics before expansion
status: RUNNING
run_id: unidexgrasp-local-subset-20261014-r1
---

# UniDexGrasp / ++ 刚体样例接入检查

Result: RUNNING. 已定位官方仓库自带单物体样例，待实读抓取标注、尺度mesh和手部FK。
Decision: 按用户授权暂停Open-AoE后续接入，先用UniDexGrasp自带样例检验几何与数据类型；不下载整库、不启动训练。

Class: Decision. 服务于Shared-WAM刚体手物监督来源的选择。
区分“原生刚体mesh/抓取标签可便宜解析”与“格式/尺度/手FK仍是主要blocker”，同时确认
发布文件是否提供真实连续手物轨迹。最便宜方法是一个官方示例物体、四个静态抓取，
读取原始数据与尺度约定，生成真实FK/mesh预览；静态标签不能通过复制或插值伪造future24。

## Decision Note / execution budget

- 用户明确要求先试UniDexGrasp和++；沿用main，不建分支，不修改既有训练和外部资产。
- 官方原仓库自带Car样例；使用固定上游提交`36c9bfcf7cedc987dd15e2384c548e51d5aadc21`。
  ++源码另在`third_party/UniDexGrasp2`，仅核验它复用的数据与策略接口。
- 有界获取≤64MiB必要样例/源码，总下载≤10分钟，每次网络请求≤30秒；
  本轮审计输出≤32MiB，静态FK/绘图CPU≤2分钟，无GPU训练/仿真。
  CPU用于少量参数几何与文件审计；如果需要策略推理/仿真，重新检查GPU预算，另记录运行。
- 输出`outputs/video-data-preparation/unidexgrasp-local-subset-20261014-r1/`，
  scratch `tmp/unidexgrasp-source-audit-20261014/`。两个第三方仓库先以blob:none/no-checkout克隆，
  不自动获取大模型或全量数据；当前GitHub API限流，改用固定SHA的raw文件。
- 格式/坐标不明、手与物体尺度严重不符、下载超限即停止扩量；保留失败记录。

## Limitations / future evidence

不以一个物体验证大规模质量/训练收益。已发布静态抓取、目标姿态和模型checkpoint与连续
rollout是不同产物；如无时钟和序列，本轮不生成完整Shared-WAM训练样本。
ShadowHand与当前Inspire的控制action不同，语义手点映射不意味着能混用Chunk-R action标签。
完整轨迹需实际策略仿真、记录hand/object/action/时间，再按30Hz构造H4+future24。
