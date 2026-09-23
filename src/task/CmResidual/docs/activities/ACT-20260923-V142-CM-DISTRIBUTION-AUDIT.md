# V1.42 后期 s3 专家 CmLite 分布审计

- date: `2026-09-23`
- branch: `agent/v142-cm-distribution-audit`
- run_status: `COMPLETED`
- evaluation code commit: `9cd4d0e`
- official actor checkpoint: `null`

冻结设计与离线准确性门槛见 experiment card。使用
现有 `eval_dexplore_full_episode_grid.py --save-transitions`
与 `analyze_cmlite_on_policy.py`，不更改策略、Cm 或
评估定义。

back240/back260 的 seeds95/96 四份64环境严格评估
和首 episode 转移均 `COMPLETED`；独立 audit JSON
已输出。全样本真实动作 EPE 相对零位移改善按
seed95 e240/e260、seed96 e240/e260 顺序为
13.9%、15.7%、25.7%、17.8%，只有一份≥20%；
打乱动作相对真实 EPE 上升19.4%–31.0%，接触
precision 0.899–0.928。预注册全样本门槛未过。

进一步按接触子集审计显示 EPE 相比相同样本零
位移低约35%–41%，运动子集低约29%–39%。
这只是事后诊断，不能把预注册失败改称通过；
也不能从观察预测准确推出反事实选动作有效。
第一次 CPU 审计缺 `PYTHONPATH` 导入失败、未写
结果，同数据与同模型在显式模块路径下重启成功。
