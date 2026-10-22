---
schema: ref2dex.probe.v2
probe_id: P-20261023-bc-ppo-usefulness
experiment_id: P-20261023-bc-ppo-usefulness
date: 2026-10-23
task: wam-rl-policy-v3
branch: main
git_commit: recorded-in-run-manifest
claim_id: C1
hypothesis_family: HF-drift-intent-residual
probe_index_in_family: 2
seed_pool: probe
seeds: [293, 294, 295, 296, 297, 298, 299]
decision_changed_if_positive: schedule daytime 1e6 controls for the same algorithm if >=3 initials improve without regressions
decision_changed_if_negative: normal KL without gain means insufficient budget or coverage, never PPO invalidity
status: RUNNING
run_id: bc-ppo-usefulness-20261023-r1
---

# BC-only vs BC+残差PPO：实测探针

## 结果解读表

| 结果（待 preflight 和配对实测填入） | 含义 | 决定 |
|---|---|---|
| 1e5 内成功率↑或限位↓，且≥3初态改善、无回退 | PPO 本身有用 | 白天排1e6卡时 |
| 1e5 无改善、KL/裁剪比正常（KL 1e-3–5e-2量级） | 预算/覆盖不足 | 只扩预算，不下“PPO无效”结论 |
| KL异常或频繁回滚 | 实现未排除 | 只修实现，不评价PPO |
| 实测<8环境controls/s | 预算账不成立 | 报数字，缩小预算或另排时间，不硬跑 |

分类 **Decision**。区分修复后的小残差 PPO 是否能在固定 BC 起点产生闭环收益；
结果决定白天是否排同算法1e6 controls，最便宜方法为先10–20min preflight、
再同初态同controls预算配对。未改善不能否决PPO。

仅GPU7，两arm顺序；GPU6虽可按用户阈值借用，本runner固定GPU7避免抢占。
不重训v6/BC、不新增baseline、不改source/pose/audit/v13并发文件。
注册：`outputs/wam-rl-policy-v3/bc-ppo-usefulness-20261023-r1/registration.json`。
模型为历史`drift-intent-residual-20261022-r1/bc/bc.pt`，冻结同一R/WAM。
面板为旧v10注册按顺序最先六个不同初态，无当前结果筛选；历史已暴露，不称泛化。
两arm均使用698维残差输入、12维latent、同有界物理意图残差/686维R观测、
同GeometryGraspConsumer/相位机和600controls评分合同。BC用均值，PPO训练采样；
最终评估两者均用均值，同初态逐数组核对。

PPO Adam1e-5、2epochs、joint exact KL target .03、超门1/4lr回滚重试，
gamma .96、GAE .95、ratio clip .2；不因结果调整算法。每回合更新，记录optimizer
attempts/accepted/retry、exact KL/clip/entropy/reward，controls/s与updates/s分开。
preflight实际PPO3600/BC600 controls，19.11min。主配对**预算缩小：14400controls/arm**；完整1e5未测。
单卡顺序总时长、08:00余量及必要的“预算缩小”另写preflight报告。
总输出<5GB，逐run上限2GB、总量另监控；<20GiB磁盘可用停止，checkpoint每40回合及最终，
不写视频，不覆盖旧checkpoint。末尾预算截断回合标记partial，不冒充600controls完整评估。

代码直接复用旧geometry worker并另建专用runner，保护文件只读快照。
差异仅为GPU7、controls/wall/output预算、两arm一致reward日志、指标/finite检查、
checkpoint限频和固定runtime复用；原成功/安全/phase条件保持。
preflight的PPO checkpoint不进入正式规模训练；正式两arm从同BC权重开始。
输出、日志、原生Python与Torch/CUDA环境见各run manifest，不写入文档目录。

Result: preflight numerical PASS, 22 updates, KL .000112–.008408, clip0–.1317, zero rollbacks; BC3.124/PPO4.060 controls/s. Main paired budget reduced explicitly to14400/arm.
Decision: original100000/arm sequential estimate15.73h misses08:00; execute14400/arm on GPU7 after measured preflight report; retain original efficacy criteria.

[实测preflight与顺序耗时账](../../ppo-bc-usefulness-preflight-20261023.md)。
