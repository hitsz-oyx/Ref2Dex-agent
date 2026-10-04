我看完了 `agent/cm-interaction-oracle`，当前相关 HEAD 是 `7310580c`（`Close Gate 1 consequence bridge probe`）。

我的结论很明确：

> **这个分支已经开始做我们刚刚确定的 Gate 1，但当前这轮 `UNPROMISING` 还不足以关闭“GT effect + interaction → 长程 G”这个假设。**
>
> 不是因为结果不好就不认，而是因为当前实现和我们刚才最终确定的实验合同还有几个实质性偏差，其中至少一个是 interaction 定义本身的问题。

### 现在这个分支实际做了什么

它的外层设计是对的。

`assemble_gate1_dataset.py` 从真实 PhysX trajectory 构造：

\[
(H_t,A_t,E^{GT}_{t:t+32},I^{GT}_{t:t+32},G_t)
\]

其中 \(G_t\) 是：

\[
G_t=\sum_{\tau=t}^T0.99^{\tau-t}r_\tau
\]

是真实 Monte-Carlo return，没有 bootstrap。

然后 `fit_gate1_value_bridge.py` 做了：

\[
V_H,\quad V_{HA},\quad V_{HE},\quad V_{HI},\quad V_{HEI},\quad V_{HAEI}
\]

并且按 **episode** 切 train/test，而不是随机拆 frame。这些都符合我们的方向。

结果是 41 个 episode、21004 个 H32 windows，4 个 split 中 `V_HEI` 相对 `V_H` 的 MAE 变化：

\[
-41.6\%,\quad +9.5\%,\quad -4.9\%,\quad -4.0\%
\]

所以它按预注册 gate 判了 `UNPROMISING`。

这部分流程没有明显“偷结果”。

---

## 但是第一个问题很严重：`H` 根本不是 history

我们刚刚明确说：

\[
H_t^-=
\{s_{t-L+1:t},a_{t-L+1:t-1}\}
\]

应该是一段过去历史。

但当前代码里：

```python
"H": torch.cat((
    dataset["state"],
    dataset["previous_action"],
    dataset["context"]
), dim=1)
```

也就是：

\[
\boxed{H=s_t+a_{t-1}+context_t}
\]

实际上只是**单帧**。

没有：

- 过去的 contact evolution；
- object velocity trend；
- contact establish/release history；
- hand-object relative motion history；
- 连续动作历史。

这和我们刚才最后确定的实验并不一致。

特别是在抓取这种问题里：

```text
contact:
0 0 1 1 1 1
```

和：

```text
1 1 1 1 1 0
```

当前帧可能完全一样，但一个是在建立抓持，一个是在脱离。

所以这轮不能叫我们讨论的最终版 Gate 1。

---

# 第二个问题更关键：当前 `I` 并不是真正的“未来 hand-object relative interaction”

我看了 `assemble_gate1_dataset.py`，这里我认为需要直接修。

现在它计算未来 hand position 的方式是：

\[
R_{o,t}^{-1}
\left(
p_{h,t+\tau}-p_{o,t}
\right)
\]

代码本质上是：

```python
inverse_current_quaternion = inverse(q_object_t)

relative_hand_position =
    R_object_t^-1 (
        p_hand_future - p_object_t
    )
```

quaternion 也是：

\[
q_{o,t}^{-1}q_{h,t+\tau}
\]

但如果我们真正想描述：

> **未来那个时刻，手相对于物体是什么关系**

应该是：

\[
\boxed{
R_{o,t+\tau}^{-1}
\left(
p_{h,t+\tau}-p_{o,t+\tau}
\right)
}
\]

以及：

\[
\boxed{
q_{o,t+\tau}^{-1}q_{h,t+\tau}
}
\]

这是两个完全不同的量。

现在的 \(I\) 实际描述的是：

> “未来手在**初始物体坐标系**中的运动。”

它会同时混进去：

- object motion；
- hand motion；
- interaction geometry。

而我们需要的是：

> **未来手和物体之间的相对关系。**

比如整个手和物体一起向上移动 5 cm，真实 interaction 完全没变。

正确的：

\[
p_h-p_o
\]

应该基本没变。

但当前表示会认为 hand 发生了大幅位移。

所以当前 `I` 对“抓得稳不稳”的语义并不干净。

**这一点我认为足以要求重做 Gate 1 assembler。**

而且好消息是：

> 不一定需要重新跑 PhysX。

因为 raw transition 已经保存了每一步的：

- `object_root`
- `hand_body_position`
- `hand_body_quaternion`

所以可以从现有 shard **离线重新计算正确的 interaction**。

---

# 第三个问题：现在的 interaction 仍然太贫乏

Gate 1 collector 保存的是：

```python
hand_body_position
hand_body_quaternion
hand_force
object_force
```

文档还明确说：

> pair identity unavailable

而且当前 fresh collector 实际是 `_contact_body_ids`，卡片里写的是 **5 个 contact-body**。

这与我们最开始 OI-Cmv2 的 interaction 思路还是有距离的。

我们真正关心的是类似：

\[
I=
\{
\text{relative geometry},
\text{contact topology},
\text{support},
\text{slip},
\text{retain/release}
\}
\]

当前主要是：

\[
\text{body pose}+\text{net force}
\]

缺少：

- contact point / surface location；
- contact partner attribution；
- 接触分布；
- tangential relative motion；
- slip；
- surface support pattern。

所以当前实验最多能说：

> “这一版压缩的 body-relative/force interaction 表示没有稳定提高 G 预测。”

还不能说：

> “GT interaction 没用。”

---

# 第四个问题：时间序列被直接 flatten 了

\(E\) 和 \(I\) 虽然是 H32 序列，但最后：

```python
"E": dataset["effect"].flatten(1)
"I": dataset["interaction"].flatten(1)
```

然后塞进：

```text
Linear → ReLU → Linear → ReLU
```

所以其实没有真正的 temporal modeling。

也就是说：

\[
I_{t+1:t+32}
\]

被当作一个巨大的静态向量。

但我们真正关心的模式恰恰是：

\[
\text{建立接触}
\rightarrow
\text{保持}
\rightarrow
\text{滑移}
\rightarrow
\text{释放}
\]

我更希望：

```text
History encoder       GRU / temporal encoder
Effect encoder        GRU / TCN
Interaction encoder   per-body encoder + temporal encoder
                       ↓
                     fusion
                       ↓
                       G
```

不需要很大，但至少语义上得匹配。

---

# 第五个问题：21004 个 sample 看起来很多，其实只有 41 个独立 episode

这是当前结果剧烈摇摆的主要嫌疑之一。

你们是 episode-group split，所以真正独立的数据量不是：

\[
N=21004
\]

而更接近：

\[
\boxed{N_{\text{independent}}\approx41}
\]

20% test 意味着一次 test 只有大约 8 个 episode。

所以出现：

\[
-41.6\%,+9.5\%,-4.9\%,-4.0\%
\]

这种 split 间剧烈变化，一点都不奇怪。

大量 overlapping H32 windows 并没有提供 21000 个独立 trajectory。

---

更糟糕的是，原 P0 数据：

\[
\boxed{\text{stable success}=0/41}
\]

所以：

- `stable_success` 是常数；
- `drop_after_success` 也是常数。

后面 e420 diagnostic：

\[
26\text{ episodes}
\]

也只有：

\[
1\text{ stable success}
\]

这实际上已经说明：

> 当前数据分布非常不适合验证“什么 interaction consequence 导致抓取长期变好”。

\(G\) 虽然不是常数，但它可能主要在反映：

- phase；
- approach reward；
- tracking；
- 当前 lift 高度；

而不是我们真正关心的：

\[
\text{稳定 interaction}
\rightarrow
\text{长期成功}
\]

---

# 第六个问题：只测了 H=32

我们之前刚刚说应该从同一批 raw trajectory 切：

\[
H\in\{1,3,5,10,16,32\}
\]

当前固定：

\[
H=32
\]

约 1.07 秒。

这可能太长。

因为 32 步 future consequence 里面已经包含很多：

\[
a_{t+1},a_{t+2},...,a_{t+31}
\]

产生的效果。

所以：

\[
C_{t:t+32}
\]

已经不只是当前 \(a_t\) 的 consequence，而是：

\[
\boxed{
a_t+
\text{未来策略行为}
\rightarrow C
}
\]

而这些未来 action 并没有进入 `V_HAEI`。

当前 `A` 只有：

\[
a_t
\]

不是 action history / future action sequence。

所以 `V_HAEI ≈ V_HEI` 也不能严格支持：

\[
A\rightarrow(E,I)\rightarrow G
\]

的充分中介结论。

---

# 因此我现在怎么评价这个分支

我会把它拆成两部分。

### 旧 oracle 部分

`effect-interaction-oracle-r2` 做得其实比较扎实。

它已经可靠证明：

\[
\boxed{\text{原来那 8 个固定动作银行饱和了}}
\]

P0：

\[
5/12
\]

state/effect/interaction/joint：

\[
8/12
\]

剩余 4 个失败 sample 的全部 32 个 candidate 都失败。

所以继续在那套 8-action candidate 上研究 effect/interation 没意义。

这一结论我接受。

---

### 新 Gate 1 部分

**方向对，当前实现不能作为最终 Gate 1。**

我不会接受现在 `docs/adr/0003-gate1-consequence-value-route.md` 所暗示的：

> 这条 exact bridge recipe 已经足够检验过，所以关闭。

更准确应该写：

> `Gate1-v1 implementation UNPROMISING / contract insufficient for hypothesis-level decision`

因为当前结果混合了：

1. 没有 history；
2. interaction 坐标语义不对；
3. interaction 信息不充分；
4. temporal encoding 太粗；
5. 独立 episode 太少；
6. outcome 几乎没有成功变化；
7. 只检查 H32。

---

# 我建议下一步非常具体，不再换方向

**不要训练 Cm，不要蒸馏，不要回 candidate。**

直接做 `Gate1-v2`。

而且第一阶段很多事情甚至**不用重新采数据**。

先用现有 raw shards 修 assembler：

\[
H^-_{t-L:t}
\]

真正构造过去 L 步：

\[
L=10
\]

包括：

- state；
- action；
- contact/event；
- context 只保留合理 phase/reference。

然后 interaction 改成真正 contemporaneous object-relative：

\[
p^{rel}_{h,\tau}
=
R_{o,\tau}^{T}(p_{h,\tau}-p_{o,\tau})
\]

\[
q^{rel}_{h,\tau}
=
q_{o,\tau}^{-1}q_{h,\tau}
\]

relative velocity 也从这个 relative trajectory 求，而不是初始 object frame 中的 hand velocity。

然后用同一 raw trajectory 同时 assemble：

\[
H_f=3,5,10,16,32
\]

不要只押 H32。

模型改成很小的 temporal encoder，而不是 flatten：

\[
Encoder_H(H^-)
\]

\[
Encoder_E(E_{1:H})
\]

\[
Encoder_I(I_{1:H})
\]

再 fusion 预测：

\[
G_t
\]

仍然保留完全一样的：

\[
V_H,V_{HA},V_{HE},V_{HI},V_{HEI},V_{HAEI}
\]

这样科学问题没有变。

---

但**真正重新下结论之前必须再采更多 episode**。

我认为现在最大的数据问题不是“21004 rows 不够”，而是：

\[
\boxed{41\text{ episodes，而且 0 success}}
\]

至少需要一个能够同时产生：

- 稳定成功；
- 中途掉落；
- 普通失败；

的 rollout distribution。

否则我们一直在研究：

> “失败轨迹内部 reward 的细微差异”

而不是：

> “什么 effect/interaction 对抓取长期价值重要”。

---

所以我的总体判断是：

> **这个分支终于在做正确的大问题了，但 AI 又过早把一个第一版实现的 negative result 写成了 Gate closeout。**

好消息是，这次不是整个路线要推倒。

**raw collector、Monte-Carlo \(G\)、episode split、ablation 框架都可以保留。真正需要重做的是 `H/I` 语义、temporal encoder、horizon sweep 和数据覆盖。**

如果这些修正之后：

\[
V_{HEI}
\]

依然稳定不优于：

\[
V_H
\]

那时我才会开始真正怀疑：

\[
\boxed{\text{effect + interaction 并不是足够有价值的中间表示}}
\]

现在还没到那一步。