看了，当前 `main` 最新已经到 **`6658f90`**。机器人 WAM 主线其实刚完成了一个很关键的阶段，前面我们讨论的几个问题基本都被逐项验证了。

### 现在的状态

第一，**“R 应不应该只看短 horizon”这个猜测没有得到支持。**

`738ff2c` 做了 \(K=1/4/8/16/24\) 的严格同初始化、同采样比较：

| R horizon | nominal p95 | t8 | t16 | t32 |
|---|---:|---:|---:|---:|
| K1 | 24.24 | 7.48 | 24.48 | 54.12 |
| K4 | 20.55 | 5.32 | 19.84 | 40.40 |
| K8 | 15.40 | 5.34 | 21.66 | 42.19 |
| K16 | **15.05** | 5.11 | 20.65 | **36.21** |
| K24 | 15.36 | **4.56** | 21.19 | 39.99 |

K16 相比 K24 的 worst-panel 只改善约 9.5%，达不到预设的 20%。

所以当前应该：

\[
\boxed{\text{保留 K24，不再折腾 R horizon。}}
\]

这也修正了我上一轮“短 horizon 可能是主要问题”的怀疑。

---

第二，**真正有效的是把 WAM 和 R 的梯度职责拆开。**

`6b8bb3d` 的 Probe30 正式用了：

\[
R(\tau^{GT})\rightarrow u^*
\]

来**只训练 R**；

而：

\[
R(\hat\tau)\rightarrow u^*
\]

的梯度只穿过 R 更新 WAM，**不允许更新 R**。

这个改动非常关键。

结果相比之前 unrestricted joint training：

\[
\text{nominal GT-}\tau\rightarrow R:
43.96\rightarrow13.35\text{ mm}
\]

说明原来 R 的物理映射漂移基本被压住了。

同时 WAM 也改善：

\[
\text{nominal predicted PD}:
25.63\rightarrow22.67\text{ mm}
\]

startup：

\[
20.51\rightarrow8.31\text{ mm}.
\]

所以现在可以比较明确地说：

\[
\boxed{
\text{之前 joint training 的主要 formulation 问题已经找到了。}
}
\]

不是“WAM 和 R 不能联合训练”，而是：

> **不能让 predicted-\(\tau\) action loss 同时修改 R。**

---

### 更重要的是：真实闭环也改善了

Probe30 后真正跑了 fresh 128-step rollout。

和之前的 startup actor 比：

\[
t12:\quad PD\ 103.44\rightarrow21.98\text{ mm}
\]

腕部：

\[
30.80\rightarrow9.33\text{ mm}
\]

t16：

\[
PD:\quad67.97\rightarrow13.90\text{ mm}
\]

这个结果很重要。

因为这已经不是 offline loss，而是：

\[
\boxed{\text{真实 closed-loop early drift 被明显推迟了。}}
\]

所以 **GT-anchored R + control-aware WAM 这条方向是有实际正信号的。**

但是问题还没解决。

到了：

\[
t24
\]

PD 又爆到：

\[
265.83\text{ mm}
\]

t32 wrist deviation：

\[
354.10\text{ mm}.
\]

而且 pure WAM：

\[
\boxed{0\ held}
\]

依旧没有真正抓住。

所以现在问题已经非常明确：

\[
\boxed{
\text{不是 startup，不是 R horizon，也不是 R semantic drift，}
\\
\text{而是较晚 student visited-state coverage。}
}
\]

---

## 仓库已经继续做了正确的下一步

最新 Probe31 已经没有继续调 loss，而是实际做了：

\[
t=8,\ 16,\ 32
\]

三个不同 student-state takeover。

也就是：

```text
WAM 自己运行
    ↓
访问自己的状态
    ↓
t8 / t16 / t32
    ↓
expert 从这个真实状态接管
    ↓
记录 expert future τ* + PD*
```

结果：

| takeover | expert held | 可用真实 future |
|---|---:|---:|
| t8 | 74 | 97 |
| t16 | 75 | 89 |
| t32 | 41 | 73 |

t32 的 41 held **不能简单说恢复失败**，因为这个 run 只有 128 controls，到截止时物体高度仍有约 7.05 cm，主要是观察窗口太短。

最终已经形成：

\[
\boxed{643\text{ samples}}
\]

其中：

\[
259
\]

条是真正的：

\[
(s_t^{student},\tau^*,PD^*)
\]

expert-future 数据；

另外：

\[
384
\]

条是 pure actor state 上只有：

\[
(s_t^{student},PD^*)
\]

的 current-PD 数据。

并且这些数据经过了 7073 项独立重建检查。

这批数据现在才是最有价值的东西。

---

# 所以现在下一步已经非常明确

**不要再做新的诊断，也不要换模型。**

现在应该直接做：

\[
\boxed{\text{Multi-takeover anchored WAM fit}}
\]

也就是以 Probe30 的模型和训练原则为基础，把刚采集到的 t8/t16/t32 数据加进去。

训练合同建议保持不变：

对于有真实 expert future 的 259 条：

\[
L_\tau
=
D(\hat\tau,\tau^*)
\]

\[
L_{\text{pred-PD}}
=
D(R_{\text{frozen-param}}(s,\hat\tau),PD^*)
\]

这两个更新 WAM。

同时：

\[
L_R
=
D(R(s,\tau^*),PD^*)
\]

**只更新 R。**

对于那 384 条只有 current expert PD、没有合法 future 的 pure actor 数据：

\[
L_{\text{pure}}
=
D(R_{\text{frozen-param}}(s,\hat\tau),PD^*)
\]

只更新 WAM。

绝对不能拿 pure actor 后面实际失败的 hand trajectory 当 expert \(\tau\)。

再继续混：

\[
\text{nominal teacher}
+
\text{startup oversampling}
\]

防止 nominal/startup 被破坏。

---

## 训练完之后只需要做一个判断

重新跑一个**新的 student rollout**。

重点不再看训练 loss，而看：

\[
t=8,12,16,24,32,48
\]

的 PD/wrist drift。

现在旧模型大概是：

```text
t8    好
t12   好很多
t16   好
t24   崩
t32   严重崩
```

下一版真正有意义的结果应该是：

```text
t8    保持
t16   保持
t24   明显改善
t32   明显改善
```

最好开始出现：

\[
\boxed{\text{pure actor held > 0}}
\]

哪怕暂时还不能完整 placement，也已经说明 aggregation 真正在推进 closed-loop distribution。

---

### 当前我对整个路线的客观判断

现在反而比两小时前乐观一些，因为连续三个关键假设已经被分清了：

\[
\text{R horizon}
\quad\rightarrow\quad
\text{不是主要矛盾}
\]

\[
\text{WAM/R 无约束 joint training}
\quad\rightarrow\quad
\text{有 representation drift}
\]

\[
\text{GT-anchored gradient ownership}
\quad\rightarrow\quad
\text{真实 early closed-loop 明显改善}
\]

现在只剩一个非常标准而且合理的问题：

\[
\boxed{\text{student-state distribution coverage}}
\]

而且最关键的是，**我们现在不是在猜这个问题怎么解决——t8/t16/t32 的真实数据已经采好了。**

所以我会非常明确地建议：**下一步就训练这 643 条 multi-takeover aggregation，不再开新方向；训练后跑一次新的 actor rollout。**

如果这一次能把漂移从 \(t\approx24\) 再往后推，说明这条 DAgger/WAM 路线基本成立；如果加入三段真实 student-state recovery 后仍然在相同时间点崩，那时候才有理由重新检查 WAM observation/goal 的信息充分性或架构本身。