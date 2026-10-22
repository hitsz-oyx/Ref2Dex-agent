| 结果解读表（实测，随证据更新） | 含义 | 决定 |
|---|---|---|
| 检查①/②/③：**3/3 PASS**；入口goal mask全部False，delta/points全0且逐位一致；6000/6000有效监督窗口；25 tests passed | 条件合同已放行 | 立即进入资源preflight并启动唯一训练 |
| 05:53 GPU4/5/6 有 source-ceiling r16 进程；磁盘可用146,821,197,824 B | 授权不等于当前空闲；保留 brief34 授权池与 spare 豁免，其他门不变 | 不抢占，资源实测通过才启动 |
| 前8帧 / 固定 contact-window hand EPE、前8帧 object EPE、预测变化量：未测 | 必须同一 no-goal/no-event 条件配对 | 对照真实初始化@A；旧B1旧事件读数不能作条件删除效应 |
| 执行器完整任务进展：本轮不跑 rollout，未测 | 离线改善不能替代执行收益 | RELEASE / 安全 / 完整任务评分不降级 |
| OOM / 非有限值 / <20GiB / 07:15 北京时间 | 故障或硬停止门 | 停并报告，不换参重启，不删除历史 outputs；>32GiB产物告警 |

## 三项最小检查

实测145.639s：[checks.json](brief37-no-goal-no-event-20261023-r1/checks.json)、[真实入口NPZ](brief37-no-goal-no-event-20261023-r1/model-boundary.npz)。9个source代表batch及生产Steps实际8窗口batch均在真实SharedWAMV6.forward_pre_hook抓取；未执行网络，仅geometry构造seam，与brief36方法相同。训练reader/collate与OnlineV6Runtime.predict实测3个goal字段逐位相等；caller给有效目标/RELEASE事件与不给条件在A下也逐位相等。训练非goal字段6000/6000全部不变；实际手/物监督窗口均6000。loss及评分分母如下（旧→A）：

```json
{
  "old_hand": 6002577,
  "old_scene": 4608000,
  "old_contact": 293734400,
  "old_part": 64125646,
  "old_first8": 2001216,
  "old_contact_window": 4329318,
  "old_geometry_future_valid": 6000,
  "A_hand": 6002577,
  "A_scene": 4608000,
  "A_contact": 293734400,
  "A_part": 64125646,
  "A_first8": 2001216,
  "A_contact_window": 4329318,
  "A_geometry_future_valid": 6000
}
```

25项针对回归通过，含真实loss_components的hand/scene/contact/part/flow_hand/flow_scene加权分母与分子完全不变。geometry GT2048标签和所有geometry有效性输入6000/6000逐位不变；geometry损失无goal_valid依赖。现有TaskScheduler的3项RELEASE测试直接只读调用通过，要求真实释放、终点、support与dwell，不以抬升替代完整任务。

本轮执行机主 brief37 路径A裁决。原goal来源认证仍为合法0/6000、确证未来0/6000、不确定6000/6000；不重复普查。新训练明示 **no-goal / no-event**，unknown=False mask 与有效零位移严格区分。保留全部6000窗口及所有监督，不更改任务规范、执行器或评分器。

## 训练启动 / 停止实测记录

待检查与preflight；训练0 updates。上限6000 updates或2026-10-23 07:15:00+08:00，先到先停。

## 固定面板

固定512 TRAIN锚点、原contact标签>0.25窗口；未来标签只评分。init@A与trained@A均使用相同unknown goal和unknown event。若真实初始化就是step011000，不另造第三格。

## 预注册阈值（预注册，非实测）

相对真实初始化@A：前8帧hand EPE与contact-window hand EPE均降≥5%，前8帧object EPE恶化≤5% → τ预测改善。20个配对reset完整任务成功净增≥4且安全中止不增加 → 改善进入执行。本轮禁止新rollout，执行门保持未测；只离线改善写“τ预测改善，执行收益未达门槛”。两层无改善写“本轮训练未达改进目标”，均不外推成τ路线被否证。旧B1只是旧配方历史参考，差值不解释为事件消融因果效应。

## 复现命令

待冻结代码与证据命令。

## 写盘字节数

写前实测/home2可用146,821,197,824 B；计划保存真实初始化约0.5GiB、单完整Adam checkpoint约1.5GiB，上限约16GiB（运行时再按实测参数量/已有checkpoint核准）。保留stop20GiB、warn30GiB、artifact alarm32GiB门。无历史输出删除。
