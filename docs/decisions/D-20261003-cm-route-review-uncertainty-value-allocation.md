# Decision Memo：Cm uncertainty-guided value allocation

Date: 2026-10-03

## 当前需要决定的问题

HF30/HF31 已关闭 native uncertainty acquisition 后的 targeted transition fit：真实
targeted rows 能定位 disagreement regime，但小样本适配会过拟合。此前 synthetic-Q、
actor feature、action ranking 和 task-value auxiliary 接口也没有通过 policy gate。
是否把 Cm 的责任改成只分配 direct-Q 的拟合容量，检验 uncertainty 是否能改善困难状态的
task-value 学习，而不改变 Q target、actor、reward、action 或数据分布？

## 当前关键证据

HF28 的 held screen 显示，Cm ensemble disagreement 的 top-20% 行物理误差是 bottom-50%
的 `7.03x`，承载 direct-Q absolute residual 的 `32.5%`，且覆盖全部 `384` held episodes。
HF29 用同一信号重训 Cm dynamics 只改善高不确定性 physical RMSE `5.1%`，没有改善
conservative value target；因此本 Probe 不再修改 Cm predictor，也不生成 synthetic target。

## 选择的行动及理由

执行一个固定的离线 Probe：在同一 fit split 上训练 direct-Q copy，所有样本都使用真实
complete return；唯一变化是 Cm uncertainty top-20% 行的 squared-error 权重固定为 `2.0`，
其余为 `1.0`。held split、网络、updates、学习率和初始化保持 direct-Q 对照一致。这样
检验的是 Cm 是否能改善 value fitting 的样本分配，而不是再次把物理预测当成价值标签。

## 成本、门和停止条件

目标是单 GPU、少于 15 分钟、只读取冻结 collection 和 checkpoint。通过条件为：held
top-20% direct-Q residual RMSE 至少改善 `5%`，overall RMSE/MAE 不恶化 `1%` 以上，且
episode Spearman 不低于 direct-Q `0.01`；同时记录 fit/held uncertainty provenance。
若任一条件失败，关闭 uncertainty-guided value-allocation，不启动 policy training、
不扫描权重/阈值/seed，也不扩大数据。若全部通过，才授权一次 matched low-budget policy
Probe，并另写 Decision Memo。

## 外部授权边界

不突破现有 GPU、时间、磁盘或权限边界；不修改外部项目，不覆盖已有 checkpoint。该行动
仍在当前 mission 的 C3 policy-utility 目标内，不改变最终 claim。
