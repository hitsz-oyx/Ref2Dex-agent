对，先把范围收紧到 **v5 的数据与 C 定义**，不讨论 C 怎么进 PPO。你问的 4 点可以很明确地回答。

## 1. v4 其实也有 contact 正负不平衡，只是没 v5 这么明显

v4 的 `objectives.py` 里，contact loss 就是对所有有效 query 直接做 BCE：

\[
L_C=\frac{1}{N_{\text{valid}}}\sum_i \operatorname{BCE}(c_i,\hat c_i)
\]

没有 positive reweight，也没有 focal。只有 part loss 是只在 `contact > 0` 的位置计算。

所以**从损失形式上说，v4 已经存在这个问题**。

但 v4 有两个因素让它没那么严重：

第一，v4 只有：

\[
24\times128
\]

个 contact query；v5 变成了：

\[
24\times2048
\]

如果真正接触面积很小，v5 会多出大量远处负样本。

第二，v4 的 contact 定义其实比较“肥”：

\[
1\text{ cm voxel}+\text{dilation}
\]

手骨架附近一大片 voxel 都会被标成 contact，所以正样本区域相对大。v5 现在定义成：

- 3 mm 内 = 1；
- 3–33 mm 指数衰减；
- 33 mm 外 = 0。

所以虽然 v5 是 soft label，但**高置信度 positive 会更局部**。

另外 v4 数据层也做了一层缓解：`contact_static/contact_moving` 窗口全部保留，`approach/static_idle` 会降采样。因此是“窗口级别”增加 contact 数据比例，但**没有解决一个窗口内部 128 个 query 的正负失衡**。

所以我的判断是：

\[
\boxed{\text{v4 有这个问题，但 v5 更值得重新统计}}
\]

现在不应该直接改 loss，先统计全量：

\[
P(c>0),\quad P(c>0.25),\quad P(c>0.5)
\]

最好按 source 分开。

如果发现例如：

\[
P(c>0.25)<2\%
\]

那就应该改 balanced BCE。

如果其实有 10–20%，就没必要过早复杂化。

---

# 2. Shadow 应该怎么处理？CEDex 是不是只用了 MANO？

这里需要区分 **CEDex 的“共享表示来源”** 和 **最终机器人数据**。

### CEDex 的共享 contact prior，确实主要来自人手

它的核心流程是：

\[
\text{human grasp data}
\rightarrow
\text{MANO hand}
\rightarrow
\text{human-like contact map}
\]

也就是先从人手 MANO 表面定义：

\[
C^h,\ P^h
\]

然后 CVAE 学：

\[
Object\rightarrow \text{human-like contact}
\]

所以**shared contact representation 的源头是 MANO 人手**。

但是 CEDex 整套方法并不是只用 MANO。

后面它针对：

- ShadowHand
- Allegro
- Barrett
- Robotiq-3F
- LEAP

分别加载真实机器人 hand model，通过 FK、robot surface 和对应 topology mapping，把 human contact intent 转成机器人抓取姿态。

所以准确说是：

\[
\boxed{
\text{contact prior 来自 MANO human}
\rightarrow
\text{robot-specific geometry 实现}
}
\]

---

## 我们这里不应该把 Shadow 伪装成 MANO

现在 v5 对 Shadow 的处理方向其实是正确的。

你仓库现在：

```text
MANO 数据
 → 真实 MANO 778 vertices
 → 16 parts

Shadow 数据
 → Shadow MJCF
 → FK
 → collision primitives
 → 778 surface samples
 → 16 parts
```

然后二者都进入统一：

\[
C:
\quad
\text{object surface}
\rightarrow
\{palm,\ thumb_{prox/mid/dist},\ldots\}
\]

这就是我们应该保留的。

也就是说：

\[
\boxed{
\text{共享的是语义，不是几何}
}
\]

MANO 的 index distal 和 Shadow 的 index distal：

- mesh 不一样；
- 长度不一样；
- DOF 不一样；

但都映射到：

\[
\text{index-distal}
\]

这个 canonical semantic part。

这比把 Shadow hand 强制拟合成 MANO 好得多。

### 所以 Shadow 我建议继续现在这条路

\[
q^{Shadow}
\rightarrow FK
\rightarrow H^{surface}_{Shadow}
\rightarrow C
\]

人手：

\[
MANO\ params
\rightarrow H^{surface}_{MANO}
\rightarrow C
\]

然后：

\[
\boxed{C\text{ 的定义保持一致}}
\]

真正要修的是我上一条说的：

> MANO 用 mesh vertices，而 Shadow 用 collision primitive samples，导致最近离散点距离统计不完全一致。

这是 **teacher metric** 要统一，不是要把 Shadow 换成 MANO。

---

# 3. object normal 怎么处理？不是“再加一个 normal 距离输出”

对，我不是说把 C 从：

\[
[c,P(part)]
\]

变成：

\[
[c,d_{\text{normal}},P(part)]
\]

我更倾向于 **C 的输出格式完全不变**：

\[
\boxed{
C=[contact\ confidence,\ part]
}
\]

只是生成 `contact confidence` 的时候，不再纯粹使用：

\[
d=\|h-o\|
\]

而是使用 object normal 判断：

> 这个靠近物体点的 hand surface，到底是在这个 surface 的正确一侧，还是隔着物体壳体靠近。

例如 object query：

\[
o_i
\]

外法线：

\[
n_i
\]

最近 hand surface point：

\[
h_j
\]

令：

\[
r=h_j-o_i
\]

普通做法只有：

\[
d=\|r\|
\]

我建议至少加入：

\[
a=
\frac{r\cdot n_i}{\|r\|+\epsilon}
\]

这个量。

如果：

\[
a\approx1
\]

说明 hand point 位于 object surface 外法线方向上，比较像真正接触。

如果：

\[
a<0
\]

说明 hand point 在该 surface 的背面。

于是可以定义：

\[
d_{\text{eff}}
=
d\cdot g(a)
\]

然后：

\[
c=f(d_{\text{eff}})
\]

CEDex 用的就是这种 normal-aware / capsule-style 思想。

### 一个直观例子

薄盒子：

```text
hand
  ↓
 ●

──────── top surface
──────── bottom surface
```

顶部 hand 到顶部 surface：

\[
d=2mm
\]

到底部 surface 可能：

\[
d=5mm
\]

如果只看欧氏距离：

> 两个 surface query 都可能得到很高 contact。

但底部 surface 的 outward normal 朝下：

\[
r\cdot n<0
\]

因此应该被明显抑制。

---

## 我们 v4 以前有没有这个问题？

**结构上有，而且可能比现在还严重。**

v4 的 GT 是：

\[
\text{hand scaffold voxel}
\cap
\text{object query voxel neighborhood}
\]

它根本没用 object normal。

也就是说只要空间足够近，就算 contact。

而且还是：

\[
1cm\ voxel + dilation=1
\]

实际上 interaction band 相当宽。

所以对于薄物体：

```text
hand scaffold
      ↓
   [voxel]
────────── face A
────────── face B
```

两面的 query 完全可能落进同一个或邻近 voxel 区域。

因此：

\[
\boxed{\text{v4 也存在“只看空间接近、不看表面朝向”的问题}}
\]

甚至定义更粗。

但我要强调：

> **我们目前没有实验直接证明 v4 因为这个原因产生了多少错误标签。**

所以这是 **structural risk**，不是已经证实的主要 failure cause。

---

## v5 要不要现在就改成完整 CEDex formula？

我反而不建议机械照搬。

更稳妥的是先做一个小 audit：

从每个 source 抽几十帧，比较：

### A. 当前版本

\[
c=f(d_{\text{euclidean}})
\]

### B. normal-aware

\[
c=f(d,a)
\]

然后统计：

\[
\Delta C
\]

主要发生在哪些 object points。

如果变化主要集中在：

- thin objects；
- concave objects；
- opposite surface；

那就说明值得改。

如果 99% 几乎一样，就没必要增加复杂度。

而且这是 offline teacher，所以做这个 audit 很便宜。

---

# 4. 先不管 C → PPO，我同意

这个我赞成暂时冻结。

现在主线就保持：

\[
\boxed{
H,G
\rightarrow
(\tau,O,C)
}
\]

当前阶段只问三个问题：

1. hand representation 是否足够；
2. \(C\) 是否是合理的 morphology-invariant interaction label；
3. WAM 是否能够把这些量学出来。

暂时不要加入：

\[
C\rightarrow PPO
\]

也不要把 CEDex robot grasp data 混进来。

否则你现在又会同时修改：

- 数据；
- contact 定义；
- WAM；
- executor observation；
- PPO。

出了结果根本没法定位是谁起作用。

---

# 所以我现在会把优先级改成这样

当前 **不要立刻改四五个东西**。

先做两个非常小的检查：

### A. v5 contact prevalence

统计：

\[
P(c>0),\quad
P(c>0.25),\quad
P(c>0.5)
\]

分别看：

```text
GRAB
OakInk2
ARCTIC
Shadow/UniDex
DexYCB
...
```

确认 BCE 是否真的失衡。

### B. normal-aware contact audit

随机抽大约：

\[
100\sim500\ frames
\]

比较：

\[
C_{\text{distance-only}}
\]

和：

\[
C_{\text{normal-aware}}
\]

重点看薄物体与接触阶段。

**只有这两个结果出来后再决定是否改 teacher/loss。**

至于 Shadow，现在的总体方向不用改：

\[
\boxed{
\text{MANO 用 MANO surface，Shadow 用 Shadow surface，
统一映射到 canonical 16-part C}
}
\]

这是当前 v5 最合理的 cross-morphology 设计。