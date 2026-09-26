# HF01–HF04 证据保全复核（2026-09-26）

范围：按已接受的 [Option A](../decisions/D-20260926-after-hf04-route-review.md) 对现有证据做只读检查。以下是文件存在、哈希和 Git 可追溯性检查，不是实验复跑或新的科学验证。

| Family | 证据入口 | 本次复核 |
| --- | --- | --- |
| HF01 | [初次 option-value Probe](../experiments/probes/P-20260925-cm-option-value.md)、[未调参复测](../experiments/probes/P-20260925-cm-option-value-retest.md) | 两次 collection 的顶层 `run_manifest.json` 和两份 `report.json` 均存在；报告中的 seed、320/320 行、固定路由 21/64 与 20/64，以及 `gate_pass=false` 与卡片一致。卡片结果提交 `2ba337e` 已进入 `main`。 |
| HF02 | [temporal expert credit 卡片](../experiments/probes/P-20260926-temporal-expert-credit.md)、[结果索引](../experiments/probes/P-20260926-temporal-expert-credit-results.json) | 结果索引与卡片固定的 SHA256 一致；索引中 24 个带路径和 SHA256 的文件均存在且哈希匹配。evaluator 提交 `83bab98` 已进入 `main`。 |
| HF03 | [contact-supported credit 卡片](../experiments/probes/P-20260926-contact-supported-credit.md)、[结果索引](../experiments/probes/P-20260926-contact-supported-credit-results.json) | run manifest、report、script、tests 的 SHA256 均与结果索引一致；实现提交 `326fed4` 已进入 `main`。 |
| HF04 | [trajectory credit 卡片](../experiments/probes/P-20260926-trajectory-credit.md)、[结果索引](../experiments/probes/P-20260926-trajectory-credit-results.json) | run manifest、report、script、tests 的 SHA256 均与结果索引一致；实现提交 `5475f59` 和结果提交 `9e579ea` 已进入 `main`。 |

HF01 本次固定的顶层文件 SHA256：

| 文件 | SHA256 |
| --- | --- |
| `outputs/CmResidual/agent_cm_option_value_dataset_20260925/run_manifest.json` | `b9b581ea3046d9bca46e09232e4659a34127410455a17d194f361a6babb1a0b7` |
| `outputs/CmResidual/agent_cm_option_value_probe_20260925/report.json` | `3a4027653af7e3ffd66267aa6f37913e370b74ba399df56ea4c6bd662d7fb72a` |
| `outputs/CmResidual/agent_cm_option_value_retest_s225/run_manifest.json` | `10f07b648e563a0570c2bb4d394801af22deee90f7baf89062659e79671aff9b` |
| `outputs/CmResidual/agent_cm_option_value_retest_probe_20260925/report.json` | `a9440a48691249f52106ccc914017353144d7174ec00766672d2c017396ff8b8` |

HF04 的运行清单和结果索引记录 `git_commit=16f345f`，这是预声明基线；卡片固定的 `5475f59` 是后续实现提交。运行清单中的 script SHA256 与该实现及结果索引一致，因此这两个提交号承担不同的 provenance 角色。

这些检查只确认现有本地文件与记录相互一致。HF01 的嵌套 collector 产物未逐项重算哈希；HF02–HF04 也未重跑 collector 或 fit。原有 `UNPROMISING`/`UNCLEAR` Probe 标签和 C3=`OPEN` 边界不变。未来若需要论文级复现，按 [Research Debt](../RESEARCH_DEBT.md) 的触发条件再安排。
