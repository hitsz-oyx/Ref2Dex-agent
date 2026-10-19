---
schema: ref2dex.probe.v2
probe_id: P-20261019-wam-final-swap
experiment_id: P-20261019-wam-final-swap
date: 2026-10-19
task: wam-rl-policy-v2
branch: main
git_commit: recorded-in-run-manifest
claim_id: C1
hypothesis_family: HF-scheduled-wam-executor
probe_index_in_family: 6
seed_pool: probe
seeds: [246]
decision_changed_if_positive: consider adapting executor training to the final WAM checkpoint
decision_changed_if_negative: retain old WAM and diagnose trajectory distribution shift
status: RUNNING
run_id: wam-final-swap-20261019-r1
---

Decision：固定PPO5500轮，比较旧WAM12000、新contractfix最终80000与DEV best30000，
判断直接替换是否改善闭环表现。GPU6/7，每条件相同16 TRAIN物体×4环境，
seed246，确定性动作，预热reset后重设seed再次reset，首个200步native回合。
每进程900秒，不更新PPO、不改主训练。graspenv；沿用已有评估协议与成功/保持指标。
新checkpoint导出model/config/stats/schema/step约400MB/份，不复制优化器；
远端已完成run为pretrain-nine-v4-contractfix-warm13000-20261018-r1，
另一个scale1024运行尚未完成，不混为本次最终模型。
新模型绑定dense manifest摘要1e2ec0cdba3eca9825232bdf2df45db9401fbba39482038c099fbcbbc14e53f7，
旧本地manifest不同；须提供匹配文件及几何payload，不关闭身份校验。
当前远端bound manifest权限000、wbcd无法读取，已请求用户提供可读副本/恢复权限。
产物目录outputs/wam-rl-policy-v2/wam-final-swap-20261019-r1/。

旧WAM对照已完成：PPO5500、64场景，54/64官方成功（84.4%），39/64保持（60.9%），
workspace_failures=0，optimizer_steps=0。与此前同配置结果存在几例波动，
本次应使用同期对照，不以历史79.7%作为固定真值。新模型未评估：bound manifest
权限000，需要所有者提供可读副本。权重正在下载；完成后后台CPU加载验证写入
outputs/ptv3-shared-wam-v4/local-control-final-20261019-r1/download-status.json。
