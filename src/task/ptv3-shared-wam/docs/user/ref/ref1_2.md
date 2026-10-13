我看了最新 `main`。现在已经新建了一条独立的 `ptv3-shared-wam` 主线，HEAD 是 `98342d8`，前一个核心实现提交是 `ff505f5`。整体已经不是“方案文档”，而是**模型、预训练、robot joint finetune、Chunk-R、K-step buffer 都已经接通了**。

但有一个很重要的问题：**当前仓库实现和我们刚才最后确定的方案还有一处关键不一致。**

## 现在仓库实际实现

整体已经是：

\[
S_{\le t}\xrightarrow{\rm PTv3}Z_S^0
\]

\[
H_{\le t}\xrightarrow{\rm HandEncoder}Z_H^0
\]

然后：

\[
[Z_S,Z_H]
=
F_{\rm interaction}([Z_S^0,Z_H^0]).
\]

具体是：

- 256 个 scene tokens；
- 22 个 hand tokens；
- 4 层 interaction Transformer；
- hidden 128；
- 4 heads；
- FFN 512。

这部分和我们想要的基本完全一致。

而且 hand 的地位确实提高了：每个 hand point 从输入开始就是独立 token：

\[
x_{h,k}
=
[p,\text{H4 dynamics}]
\]

再加：

\[
E_{\rm handID}+E_{\rm keypointID}+E_{\rm type}.
\]

不再是以前那种“最后给 decoder query 加一个 hand residual”。

---

### U 和 V 也已经变成并行双头

Scene：

\[
\hat U=D_U(Z)
\]

输出：

\[
[B,O,24,32,3].
\]

Hand：

\[
\hat V=D_V(\cdot)
\]

输出：

\[
[B,24,2,11,3].
\]

也已经彻底取消：

\[
V\rightarrow W\rightarrow U
\]

这种旧 action-conditioned World 路径。

所以从架构主体看，现在确实已经非常接近我们说的 PointWAM-style shared forecaster。

---

## Goal 也已经改成了我们刚才说的 endpoint delta

现在不是 +8/+16/+24。

也不是整条任务的最终位置。

而是：

\[
\boxed{
\Delta P_G=P^{goal}_{t+24}-P_t
}
\]

32 个 token，每个：

\[
\boxed{
[p_i,\Delta p_i]\in\mathbb R^6.
}
\]

而且代码专门做了 contract：

`current-point-horizon-delta.v1`

旧 absolute-goal checkpoint 会直接拒绝加载，防止语义混掉。

这一点做得比较干净。

---

# 但关键偏差就在这里

现在代码实际上是：

\[
\boxed{
Z=F(S,H)
}
\]

scene：

\[
\boxed{
U=D_U(Z)
}
\]

而 hand 是：

\[
\boxed{
V=D_V(Z,G)
}
\]

也就是说 **goal 只进入 hand branch**。

`GoalConditioner` 是：

```text
shared hand tokens
        ↓
cross-attend goal tokens
        ↓
hand future
```

而不是：

```text
scene tokens
hand tokens
goal tokens
     ↓
shared interaction transformer
     ↓
U / V
```

源码甚至明确写的是：

> `U=D_U(Z), V=D_V(Z,G); Z sees only observed scene and semantic hands.`

并且测试里专门要求：

> 改 goal 时 scene prediction 和 shared representation 必须完全不变。

所以这不是偶然 wiring，而是当前版本**主动设计成非对称条件**。

---

# 这和我们刚才最后讨论的方案不一致

我们最后已经讨论到：

既然现在 \(G\) 只是：

\[
\Delta P_{t\rightarrow t+24},
\]

不是完整未来 trajectory，

那可以直接做：

\[
\boxed{
Z=F(S,H,G)
}
\]

然后：

\[
\boxed{
U=D_U(Z),\qquad V=D_V(Z).
}
\]

也就是 scene/hand 两个 head **完全共享同样的任务条件**。

我现在仍然认为，这更符合你想做的那一版。

因为否则对于同一个：

\[
S,H
\]

不同 object goal：

\[
G_1,G_2
\]

scene head 都必须输出同样的：

\[
U.
\]

这实际上让 scene forecast 学的是：

\[
p(U\mid S,H)
\]

而我们真正想做的 task-conditioned WAM 更自然是：

\[
\boxed{
p(U,V\mid S,H,G).
}
\]

这也更对应 PointWAM：

\[
p(U,V\mid S,H,L).
\]

只是他们的 \(L\) 是 language，我们的 \(G\) 是 endpoint displacement。

---

# Robot finetune 这部分已经按我们想要的方式实现了

这个倒是做得比较完整。

现在机器人阶段不再是 decoder-only。

完整更新：

- PTv3；
- HandTokenEncoder；
- Interaction Transformer；
- Scene head；
- Hand head；
- Goal conditioner；
- Chunk-R。

其中 PTv3：

\[
lr=10^{-5}
\]

新增模块：

\[
lr=10^{-4}.
\]

也就是：

\[
\boxed{\text{full forecaster + R joint finetune}}
\]

已经真正实现了。

---

## Loss 也加入了我们之前要求的 GT-\(\tau\) anchor

现在：

\[
L_{\rm robot}
=
L_H+L_S+L_{A,\hat V}+L_{A,V^*}.
\]

其中：

\[
L_{A,\hat V}
=
\|R(s,\hat V)-A^*\|_1
\]

会同时更新：

\[
WAM+R.
\]

而：

\[
L_{A,V^*}
=
\|R(s,V^*)-A^*\|_1
\]

只更新：

\[
R.
\]

这个设计正好针对我们之前 joint training 时出现的问题：

\[
R(V^*) : 10.6\rightarrow44\text{ mm}.
\]

所以现在不会再让 R 为了适应 predicted-\(\tau\) 随便漂掉 physical mapping。

这一块我认为是正确的。

---

# Chunk 也已经真正按 chunk 做了

现在 R 还是：

\[
24\times33
\rightarrow
8\times12.
\]

但执行端已经增加 buffer：

\[
K_{\rm exec}=1/4/8.
\]

默认：

\[
\boxed{K_{\rm exec}=4}.
\]

也就是现在可以真正做：

```text
query WAM/R
↓
得到8步action
↓
执行前4步
↓
重新观测
↓
再query
```

而不是原来的每 tick 只拿第一个 action。

这一块也已经按我们想法实现。

---

# 当前还只是工程接通，没有真正开始科学训练

这个要特别注意。

现在 repo 里看到的是：

- 17 个 contract tests 通过；
- 三域 delta-goal pretrain **4步 smoke**；
- robot joint finetune **4步 smoke**；
- native delta-goal **4步 smoke**；
- 之前 absolute-goal 版本也做过 DDP / K=1/4/8 工程测试。

但是：

\[
\boxed{\text{还没有正式三域大训练}}
\]

也没有：

\[
\boxed{\text{airplane overfit正式训练}}
\]

更没有：

\[
\boxed{\text{新的稳定抓取结果}}.
\]

所以目前结论只能是：

> 架构已经接通，梯度/张量/执行合同基本成立。

还不能判断新 Shared-WAM 是否有效。

---

# 我现在最建议先改哪一处

**先不要启动正式长训。**

我会先把当前：

\[
U=D_U(Z),\quad V=D_V(Z,G)
\]

改成我们刚才确定的：

\[
\boxed{
Z=F(S,H,G)
}
\]

\[
\boxed{
U=D_U(Z),\qquad V=D_V(Z).
}
\]

也就是 GoalEncoder 产生 32 个 goal tokens：

\[
Z_G
\]

然后：

\[
[Z_S,Z_H,Z_G]
\]

一起进入 Interaction Transformer。

这样三种 token：

```text
256 scene
22 hand
32 goal
```

总共：

\[
310
\]

个 token。

token type 变成：

\[
E_{\rm scene},
E_{\rm hand},
E_{\rm goal}.
\]

然后 U/V 两个 head 都只读 shared representation。

这才是真正的：

\[
\boxed{\text{task-conditioned shared world-action forecaster}.}
\]

---

唯一需要配套做一个防止 trivial copying 的检查：

因为 input 有：

\[
\Delta P_{24}
\]

U 又预测：

\[
U_{1:24}.
\]

所以至少要报告：

- h1；
- first8；
- middle；
- h24；
- target object；
- non-target objects。

最好再做一个：

\[
\text{goal-only baseline}
\]

看只用 endpoint delta 是否就能把 U 的 h24 做得很好。

如果 h24 很好但 first8/middle 根本不好，就说明只是在抄终点。

这个诊断足够了，不必为了防泄露再把 goal 从 U 分支完全拿掉。

---

## 总结现在仓库状态

新路线已经有了，而且大部分实现是正确的：

\[
\boxed{
PTv3
+
22\ hand\ tokens
+
Shared\ Interaction\ Transformer
+
U/V\ 双头
+
Full\ robot\ joint\ finetune
+
GT\text{-}\tau\ R\ anchor
+
Chunk\ execution
}
\]

现在唯一明显和我们**最新决定**不一致的是：

\[
\boxed{\text{goal 目前只进入 V，不进入 shared trunk/U。}}
\]

我建议先把这个改成**对称 task conditioning**，然后这版架构就基本可以冻结，直接进入：

\[
\text{三域正式预训练}
\rightarrow
\text{airplane overfit}.
\]

不要再在结构上继续绕了。