可以，而且我觉得这三项现在是**最合理的一组改动**。但第三项“让预测 \(\hat\tau\) 参与 \(E\) 预测”一定要按两阶段做，不要直接把整个 WAM 联合端到端放开，否则很容易再次出现共适应。

我会把这一版定义成：

\[
\boxed{\text{WAM-v2 = Hand-anchored PTv3 + Chunk-R + Predicted-}\tau\text{ World Consistency}}
\]

三项分别解决不同问题，不是重复改同一件事。

---

## 1. \(R\)：单步 MLP → trajectory-conditioned chunk Transformer

这个我认为应该第一个做。

当前：

\[
(s_t,\tau_{1:24})\rightarrow PD_t
\]

而且 24×33 直接 flatten。

改成：

\[
\boxed{
(s_t,\tau_{1:24})
\rightarrow
a_{t:t+7}
}
\]

结构不要太大：

```text
τ1  33D → MLP → token
τ2  33D → MLP → token
...
τ24 33D → MLP → token

state 87D → MLP → state token

[state, τ1,...,τ24]
          ↓
  Transformer memory

8 learned action queries
          ↓
2-layer Transformer Decoder
hidden=128, heads=4, FFN=512
          ↓
      8 × 12D action
```

我建议这里直接监督现有数据里的 **12D independent native applied action**，而不是再输出 PD target。

因为仓库现成的 `retargeter_data.build_windows()` 本来就有：

\[
\tau_{1:24}^{GT}
\rightarrow
a_{1:8}^{applied}.
\]

部署仍然是 receding horizon：

\[
\text{预测8步，只执行第1步}
\]

所以下一 tick：

\[
(s_{t+1},\hat\tau_{t+1:t+24})
\rightarrow a_{t+1:t+8}
\]

重新规划，不是 ACT 那种开环执行 8 步。

### 第一关必须独立过

先完全不用 WAM：

\[
\boxed{
\tau^{GT}\rightarrow R_{\rm chunk}\rightarrow action
}
\]

看 native simulation 能不能稳定抓。

如果 GT \(\tau\)+新 \(R\) 都不行，就别继续改 WAM。

---

# 2. WAM：改 PTv3 的 hand readout，不是重写 PTv3

这里要特别说清楚。

**当前 hand 已经进入 PTv3。**

现在：

\[
\text{object points + hand points}\rightarrow PTv3
\]

是成立的。

真正的问题在 PTv3 后面。

当前代码：

```text
object + hand
     ↓
    PTv3
     ↓
packed point features
     ↓
统一抽样 ≤256 scene tokens
     ↓
trajectory decoder

另外：
raw hand XYZ/history
     ↓
hand_query MLP
     ↓
trajectory query
```

所以 trajectory decoder 并没有明确拿到：

> “这一个拇指尖经过 PTv3 与周围物体交互后的 feature”。

我建议改成：

\[
\boxed{
PTv3\ output
\rightarrow
\text{explicit gather 2×11 hand features}
}
\]

得到：

\[
F_H\in\mathbb R^{B\times2\times11\times128}.
\]

然后 trajectory query：

当前：

\[
q_{t,i}
=
e_t+e_i+f^{raw}_{hand,i}
\]

改成：

\[
\boxed{
q_{t,i}
=
e_t+e_i+
f^{PTv3}_{hand,i}
+
f^{raw}_{hand,i}
}
\]

也就是说 raw hand adapter 先别删，当 residual information。

这样：

```text
object + hand
      ↓
     PTv3
   ↙      ↘
scene     explicit hand tokens
memory          ↓
   \            /
    trajectory decoder
```

### 这个改动其实不大

不需要：

- 换 PTv3；
- 加深 PTv3；
- 上大模型；
- 重新定义数据。

只是改变：

\[
\boxed{\text{PTv3 output → trajectory decoder 的接口}}
\]

而且 hand token 必须保证不会被 256-token sampling 丢掉。

我会把它看成比“继续扩大数据”更值得优先验证的结构改动。

---

# 3. WAM 两阶段训练：让 \(\hat\tau\) 真正参与 \(E\)

这个方向我赞成，而且我认为这是三项里**最像真正 WAM 改进**的一项。

因为当前实际上是：

\[
\hat\tau=A(H,G)
\]

但 world 用的却是：

\[
E=W(H,\tau^{GT/executed}).
\]

也就是说：

\[
\boxed{\hat\tau\text{ 根本没有进入 }E\text{ 的预测。}}
\]

所以 actor 和 world 现在没有形成真正的：

\[
H,G\rightarrow\hat\tau\rightarrow\hat E.
\]

---

## 两阶段应该这样做

### Stage 1：Teacher-forced World

保持现在的方式：

\[
\boxed{
W(H,\tau^{GT})\rightarrow E^{GT}
}
\]

同时：

\[
A(H,G)\rightarrow\tau^{GT}.
\]

训练：

\[
L_1
=
L_{\tau}
+
\lambda_E L_E
\]

其中：

\[
L_E=
D(W(H,\tau^{GT}),E^{GT}).
\]

这一阶段的目的：

> 先保证 \(W\) 真的理解“手怎么动 → 物体怎么动”。

不要让一个还很差的 \(\hat\tau\) 去污染 world。

---

# Stage 2：Predicted-\(\tau\) consistency

这是关键。

得到：

\[
\hat\tau=A(H,G).
\]

然后构造 world branch 所需的 9D hand action feature。

当前 world 的 `ExecutedActionAdapter` 吃的不是单纯 3D displacement，而是类似：

\[
[p_t,\ p_t-p_0,\ p_t-p_{t-1}]
\]

这种 9D future hand representation。

所以从 \(\hat\tau\) 可以确定性恢复：

\[
\hat p_t=p_0+\hat\tau_t
\]

然后：

\[
\Delta\hat p_t^{current}=\hat\tau_t
\]

以及：

\[
\Delta\hat p_t^{step}
=
\hat p_t-\hat p_{t-1}.
\]

第一步用：

\[
\hat p_1-p_0.
\]

于是得到：

\[
\hat A^{hand}_{1:24}\in\mathbb R^{24\times11\times9}.
\]

再喂：

\[
\boxed{
\hat E=
W(H,\hat\tau)
}
\]

---

## 但这里一定要 freeze World

这是最重要的一条。

Stage 2：

\[
\boxed{
W\text{ 的 action adapter + world decoder 参数全部冻结}
}
\]

然后：

\[
L_{\rm effect}
=
D(
W_{\rm frozen}(H,\hat\tau),
E^{GT}
).
\]

梯度：

\[
\hat E
\rightarrow
\hat\tau
\rightarrow
A
\]

回到 hand forecaster。

但是：

\[
\boxed{
\frac{\partial L_{\rm effect}}
{\partial W}
=0.
}
\]

否则就会发生：

```text
错误 τ_hat
   ↓
World为了适配它自己发生变化
   ↓
仍然预测正确 E
```

最终：

\[
\hat E\approx E^{GT}
\]

但：

\[
\hat\tau
\]

根本不是物理合理动作。

这和我们前面 WAM+\(R\) unrestricted joint training 出现的 representation drift 是一个问题。

---

## 我甚至建议 Stage 2 把 scene 输入也 detach 一下

因为 world 和 actor 共享 PTv3。

如果：

\[
L_E
\]

同时通过 world 的 scene branch 更新 PTv3，那模型还可能通过：

\[
H\rightarrow scene\ feature
\]

这条路径去作弊，而不是逼 \(\hat\tau\) 改善。

所以计算 predicted-\(\tau\) consistency 时最好：

\[
\boxed{
\hat E=
W(
\operatorname{sg}(Z_{scene}),
\hat\tau
)
}
\]

其中：

\[
\operatorname{sg}
\]

就是 stop-gradient。

这样：

\[
L_E
\]

唯一能改善的路径基本就是：

\[
\boxed{
L_E
\rightarrow
\hat\tau
\rightarrow
trajectory\ decoder/PTv3
}
\]

而不是修改 world 对 scene 的解释。

这非常干净。

---

# Stage 2 最终 loss

我建议：

\[
\boxed{
L_{\rm stage2}
=
L_{\tau}
+
\lambda_a L_{\rm action}
+
\lambda_e L_{\rm effect}
}
\]

其中：

### hand trajectory

\[
L_{\tau}
=
D(\hat\tau,\tau^{GT})
\]

### R grounding

\[
L_{\rm action}
=
D(
R_{\rm frozen-param}(s,\hat\tau),
a^{GT}
)
\]

这个只更新 WAM，不更新 \(R\)。

### World consistency

\[
L_{\rm effect}
=
D(
W_{\rm frozen}(\operatorname{sg}(H),\hat\tau),
E^{GT}
).
\]

所以 \(\hat\tau\) 同时必须满足三个条件：

\[
\boxed{
\begin{cases}
\text{像真实 hand trajectory}\\
\text{能产生正确 robot action}\\
\text{能导致正确 object effect}
\end{cases}}
\]

这就非常像我们真正想要的：

\[
\boxed{\text{World Action Model}}
\]

而不是单纯 trajectory BC。

---

# 三项不要同时从头训

我建议实际开发顺序：

### A. 新 R

固定旧 WAM。

先做：

\[
\tau^{GT}\rightarrow R_{\rm chunk}
\]

确认 GT execution gate。

---

### B. Hand-anchored PTv3 WAM

暂时仍用旧：

\[
W(H,\tau^{GT})\rightarrow E.
\]

比较：

\[
\text{old WAM}
\quad vs\quad
\text{hand-anchored WAM}.
\]

重点看：

- h1；
- first8；
- nominal；
- recovery；
- goal sensitivity。

别先看 closed-loop。

---

### C. Stage 2 predicted-\(\tau\) World

等 A、B 各自证明至少不退化，再接：

\[
\hat\tau\rightarrow W_{\rm frozen}\rightarrow\hat E.
\]

这时才开始检查：

\[
\text{effect consistency}
\]

到底有没有进一步提升 hand trajectory / action / closed-loop。

---

# Goal 暂时完全不改

这里我建议很明确：

\[
\boxed{\text{先保留现在的32-point endpoint + memory injection。}}
\]

因为你们已经系统比较过：

- memory；
- query；
- none。

目前结果是：

\[
\boxed{\text{observed hand + memory goal + world1}}
\]

最值得保留。

所以**这一版不要再同时上 3PoinTr**。

否则一次同时改变：

- R；
- PTv3 hand anchoring；
- world training；
- goal temporal representation；

变量太多。

3PoinTr-inspired temporal goal 可以作为下一版：

\[
P_T
\rightarrow
P_{1:T}
\]

的升级，但不是现在。

---

## 我会把最终架构冻结成这样

```text
                    32-point object endpoint goal
                              │
                              ↓
                         Goal Encoder
                              │
                              ↓
object + hand ─────────────→ PTv3
                              │
                  ┌───────────┴───────────┐
                  ↓                       ↓
           scene context          explicit hand tokens
                  │                       │
                  └───────────┬───────────┘
                              ↓
                    Hand Trajectory Decoder
                              ↓
                         τ_hat 1:24
                         /         \
                        /           \
                       ↓             ↓
              Chunk Transformer R    Frozen World
                    ↓                  ↓
                 a_1:8               E_hat
                    │                  │
              execute a_1        effect consistency
```

训练 World 时还有：

```text
GT τ → World → E_GT
```

作为 Stage 1。

---

所以答案是：**可以，而且我认为这三项正好组成一版完整、逻辑闭环的升级。**

不过它们的优先级是：

\[
\boxed{
R_{\rm chunk}
\rightarrow
PTv3\ hand\ anchor
\rightarrow
predicted\text{-}\tau\ world\ consistency
}
\]

不是三项一起打开。

如果这三项完成后仍然：

\[
\tau^{GT}+R_{\rm chunk}\text{可以执行}
\]

但：

\[
\hat\tau+R_{\rm chunk}\text{仍长期失败}
\]

我们才真正有资格把问题定位到 **WAM forecaster 本身**，而不是继续在执行层、world coupling 和数据覆盖之间来回猜。