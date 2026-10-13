看了。最新 `main` 是 **`9cfb13b`**。这几轮把我们刚讨论的两个方向都推进完了，而且结论比之前清楚很多。

### 1. 数据清洗审计已经做完：不能简单删“静止 goal”

全量统计了 **4,378,478 个 TRAIN windows**。按当前真实采样分布，约：

\[
\boxed{44.67\%}
\]

的训练曝光满足：

\[
g<2\text{ mm}
\]

也就是 object endpoint goal 基本没变。这个比例确实很高。

但进一步拆开后发现，真正“没信息的 idle”其实很少：

- static-goal + target附近交互：**26.06%**
- static-goal + hand明显运动但离目标远：**14.70%**
- 真正 idle-away：只有 **0.65%**

更关键的是那 14.70% 里面：

\[
12.07\%
\]

其实手是在靠近**另一个物体**，只是当前被指定的 target object A 没动。

也就是说 OakInk2 里有很明显的：

\[
\boxed{\text{target-hand association mismatch}}
\]

问题：

> 手实际上操作 B，但 actor 的 goal 却拿了静止的 A。

所以我们刚才猜的“静止 goal 会污染 trajectory supervision”是有依据的，但正确做法不是：

\[
g\approx0\Rightarrow delete
\]

而是更像：

\[
\boxed{\text{降低 target-mismatch window 的 actor loss}}
\]

同时 world/scene supervision 仍保留。

而 GRAB/ARCTIC 的静止窗口很多其实是 departure/contact phase，不能删。

---

### 2. 还有一个很重要的数据问题：GRAB/ARCTIC 缺 far-to-contact

当前 native preprocessing 只允许：

\[
\text{near}[t]\land \text{pose valid}[t]
\]

进入训练。

审计发现 GRAB/ARCTIC 中：

\[
\boxed{0}
\]

个 TRAIN window 是“两只手当前都离 target >50 mm”。

也就是说它们的数据天然缺：

```text
手还很远
→ approach
→ near contact
```

的起始阶段。

这件事其实很可能和我们 Gym 中：

\[
t32\rightarrow t56
\]

接触前 transition 学不好有关。

所以现在数据侧最值得做的不是“大清洗”，而是两件分开的事：

\[
\boxed{
\text{OakInk：修 target association / actor weighting}
}
\]

和

\[
\boxed{
\text{GRAB/ARCTIC：补 far-to-contact coverage}
}
\]

---

# 3. robot proprio + temporal goal 已经按我们说的加了

现在 WAM 新增：

\[
48D=[q_{18},\dot q_{18},a_{t-1}^{12}]
\]

再加：

\[
27D=[
\Delta T^{ref}_{t+8},
\Delta T^{ref}_{t+16},
\Delta T^{ref}_{t+24}
]
\]

其中每个位姿：

\[
3D\ translation+6D\ rotation.
\]

都是 zero-gate 初始化，所以初始时严格保持 C-final 输出不变。

Human 数据全部 mask 这两个 robot conditioner。

这正是我们刚讨论的：

\[
\text{geometry prior}+\text{robot-specific control context}.
\]

---

## 4. 它离线确实改善了 contact 区域

先用 C-final 自己跑到：

\[
t32,\ t40,\ t48
\]

然后 expert takeover。

这三个状态都是可恢复的：

\[
75/74/71\ held
\]

所以不是“C 已经走到不可逆状态”。

训练后，在**精确 C takeover state**上：

| state | C first-PD | 新模型 |
|---|---:|---:|
| t32 | 51.76 mm | **1.98 mm** |
| t40 | 119.76 mm | **83.49 mm** |
| t48 | 351.89 mm | **33.69 mm** |

所以新增状态信息 + temporal goal 对 contact-region 确实有帮助。

---

# 5. 闭环也明显比 C-final 好，但仍然没抓稳

C-final：

\[
held=0,\qquad max\ height=0
\]

新 robot-context 模型：

\[
held=0
\]

但：

\[
\boxed{max\ height=276.6\text{ mm}}
\]

而且：

\[
\min\ hand-object\ gap=0.913\text{ mm}.
\]

所以这次不是“完全没碰到物体”。

它实际上：

```text
接近
→ 形成接触
→ 物体被带/推到空中
→ 手和物体迅速分离
→ 掉落
```

clipping 也从：

\[
9.23\%\rightarrow1.48\%.
\]

并且 t48/t56 的 wrist drift：

\[
174\rightarrow47\text{ mm}
\]

\[
394\rightarrow83\text{ mm}.
\]

因此：

\[
\boxed{\text{补充控制输入确实解决了一部分问题}}
\]

但还没有形成稳定 grasp。

---

# 6. 现在出现了一个非常关键的新证据：R 其实并没有“彻底解决”

这个我认为是目前仓库最重要的新发现。

之前我们说：

\[
GT\ \tau+ChunkR\rightarrow441\ held
\]

所以认为 R 基本 solved。

但那只是 **nominal distribution**。

现在把 C 自己 t32/t40/t48 的真实状态拿出来，再把这些状态下 expert 的真实未来：

\[
\tau^{GT}
\]

喂给同一个 Chunk-\(R\)，结果 first-PD error：

### 精确 takeover state

\[
t32:32.8\text{ mm}
\]

\[
t40:59.0\text{ mm}
\]

\[
\boxed{t48:159.9\text{ mm}}
\]

如果看 takeover 后 first16：

\[
34,\ 172,\ 223\text{ mm}.
\]

也就是说：

\[
\boxed{
R(s^{nominal},\tau^{GT})\text{很好}
}
\]

并不意味着：

\[
\boxed{
R(s^{C\text{-visited}},\tau^{GT})\text{也好}
}
\]

这非常关键。

因此现在不能再简单说：

> WAM 是唯一 blocker，R 已解决。

更准确的是：

\[
\boxed{
\text{WAM和R都存在 closed-loop state coverage 问题}
}
\]

而且现在的新 actor 有时预测一个“不完全 GT 的 \(\tau\)”反而能让 R 输出更接近 expert action。

这说明 actor 在某种程度上已经开始**补偿 R 的误差**。

这个方向比较危险，因为：

\[
\tau
\]

就不再保持纯粹的“真实未来手轨迹”语义了。

---

# 7. 还有一个训练数据细节必须重视

新 recovery 数据一共有：

\[
195
\]

个 window。

但真正属于 C actor 自己状态的只有：

\[
\boxed{3}
\]

个：

\[
t32,\ t40,\ t48.
\]

剩下：

\[
192
\]

个都是：

> expert takeover 以后继续执行形成的 expert-continuation states。

所以即使训练里写着“50% recovery”，实际上：

\[
\boxed{\text{真正 autonomous visited-state 的密度还是极低}}
\]

这和我们之前怀疑的完全一致。

---

# 所以我现在对主因的排序会调整

之前：

1. WAM contact-state coverage；
2. goal / proprio 不足；
3. World exploitation；
4. R 放大误差。

现在应该改成：

\[
\boxed{
1.\ own-state\ coverage不足
}
\]

\[
\boxed{
2.\ R在own-state恢复分布上泛化不足
}
\]

\[
3.\ WAM输入不足 / goal欠约束
\]

\[
4.\ 数据target association问题
\]

\[
5.\ World consistency偏差
\]

其中第 2 项现在证据已经很强。

---

## 我建议下一步不要继续放大 robot-context 模型

它已经证明：

> 给 \(q,\dot q,a_{t-1}\) 和 temporal goal 有价值。

这个问题回答了。

但现在闭环失败不是因为“adapter训练得还不够久”。

下一步应该先把：

\[
\boxed{
(s^{C},\tau^{expert})\rightarrow a^{expert}
}
\]

这个 R 合同修好。

也就是用 t32/t40/t48 的 actual own-state expert continuation 去训练/校准 Chunk-\(R\)，同时保留 nominal replay。

并且必须单独看：

\[
\text{exact takeover states}
\]

而不是把 takeover 后 60 多个 expert state 当成同一种“own-state recovery”。

与此同时，数据清洗那条线可以做一个很便宜的短 A/B：

\[
\text{原 sampler}
\]

vs

\[
\text{OakInk target-mismatch actor downweight}
\]

只改 actor loss 权重，不删数据、不改 world。

这个现在已经有足够数据依据去做，但我不会把它和 R 修复混在一个实验里。

所以当前仓库其实已经把问题切得比昨天清楚很多：

\[
\boxed{
\text{输入更丰富}\Rightarrow\text{接触显著改善}
}
\]

但

\[
\boxed{
\text{own-state监督太稀}
+
\text{R own-state泛化差}
\Rightarrow
\text{仍不能稳定抓持}
}
\]

这两项我认为是现在最应该优先处理的。