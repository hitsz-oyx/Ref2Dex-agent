# Ref2Dex Research Mission

## 1. Core idea

本项目希望研究：

> 是否可以学习一种短期手物交互转移表示，并利用其物理预测帮助机器人强化学习策略完成灵巧手抓取。

短期转移可以是一步或短多步；物理预测与长期动作价值是不同概念。
物体运动幅度不能直接替代抓取动作价值，后者取决于任务目标和后续执行策略。


---

## 2. Final objectives

项目最终需要同时解决两个问题。

### Objective A — Self-trained manipulation

在 DExplore 的物理环境中，不直接使用官方 DExplore actor checkpoint 作为最终策略基础，自行训练得到能够完成目标参考轨迹任务的策略。

---

### Objective B — Demonstrate that Cm matters

仅仅训练出一个能够抓取的策略不足以完成本项目。

还必须回答：

> 短期手物交互转移表示是否对最终策略产生了真实、可验证的增益？
