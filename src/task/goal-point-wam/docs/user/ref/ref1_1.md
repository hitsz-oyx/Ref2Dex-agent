看了，主分支现在已经把我们刚才讨论的 WAM 真正做出来了一版，但**当前结果暴露了一个很关键的问题：模型现在基本没有真正使用 goal。**

当前 `main` HEAD 是：

```text
3e2f8528  Record WAM fitting and goal-conditioning diagnostics
```

而且新建了独立任务：

```text
src/task/goal-point-wam/
```

### 现在实际实现的架构

已经不是之前口头讨论了，代码现在是：

\[
\text{Scene history}
\xrightarrow{\text{PW PTv3}}
Z_H
\]

然后：

\[
G\xrightarrow{\text{GoalEncoder}}Z_G
\]

再：

\[
(Z_H,Z_G)
\xrightarrow{\text{TrajectoryDecoder}}
\hat\tau
\]

世界分支则严格 goal-blind：

\[
(Z_H,\tau_{\rm executed})
\xrightarrow{\text{WorldDecoder}}
\hat E.
\]

这部分设计方向是对的。

而且后续版本已经修正成：

- 从现有 **PointWorld step-50000 checkpoint** 初始化；
- PTv3 + `scene_proj` 真正继承；
- 一共继承 **50,436,224 参数**；
- 整个 WAM 约 **52.26M 参数**；
- world branch 保留完整 **528 个 action tokens**，没有再做错误的时间平均。

所以“继续用 PointWorld backbone”已经真正落地。

---

## 但这里出现了一个和我们刚刚讨论不一致的地方

现在 `GoalEncoder` 输入的不是：

\[
\boxed{\text{物体 start cloud + object goal cloud}}
\]

而是：

\[
\boxed{\text{当前 hand 22 点 + t+24 future hand 22 点}}
\]

代码直接叫：

```python
start_hand
goal_hand
```

GoalEncoder 实际产生的是：

\[
2\times2\times11=44
\]

个 hand endpoint tokens，再加一个 null token。

所以目前这版其实是：

\[
\boxed{
\text{“给定手的终点，预测完整手轨迹”}
}
\]

而不是我们后来确定的：

\[
\boxed{
\text{“给定物体目标点集，规划手轨迹”}
}
\]

这个区别很大。

换句话说，**现有 goal 其实已经相当接近 action answer 了**。

---

# 当前训练结果怎么样

第一版简化实现后来发现有两个问题：

1. 没加载 PW pretrained backbone；
2. 528 个 hand/action tokens 被压成了 24 个 time tokens。

后来已经修了。

修正后的 `ref1-alignment`：

- PW pretrained；
- full 528 action memory；
- 600 updates；
- best step 400。

### Trajectory EPE

| 数据 | 当前 WAM |
|---|---:|
| OakInk2 | 33.78 mm |
| GRAB | 152.95 mm |
| ARCTIC | 72.10 mm |

### moving-object h24 world EPE

| 数据 | WAM | static |
|---|---:|---:|
| OakInk2 | 30.30 | 30.65 mm |
| GRAB | 199.96 | 201.57 mm |
| ARCTIC | 93.10 | 93.61 mm |

提升只有：

\[
1.14\%,\quad0.80\%,\quad0.55\%.
\]

预注册要求是至少两个数据源：

\[
\ge5\%.
\]

所以：

\[
\boxed{\text{UNCLEAR}}
\]

基本还没有形成有意义的 WAM prediction gain。

---

# 然后他们怀疑是不是学习率太小

做了固定 6-window memorization。

这一步其实很有信息量。

把 LR 从：

\[
10^{-4}\rightarrow10^{-3}
\]

之后，固定训练样本可以学得非常好。

最终 trajectory EPE：

- OakInk2：2.58 mm
- GRAB：4.00 mm
- ARCTIC：3.94 mm

world EPE：

- 2.30 mm
- 6.68 mm
- 3.14 mm

所以：

\[
\boxed{\text{模型容量和基本反传没有坏。}}
\]

它是**能拟合**的。

---

# 但换到正常 mixed-source 训练，依然不行

用同样：

\[
lr=10^{-3}
\]

重新正常训练 600 updates。

trajectory：

| | old LR | new LR | static hand | linear goal |
|---|---:|---:|---:|---:|
| OakInk2 | 35.24 | 35.30 | 34.80 | **9.16** |
| GRAB | 141.70 | 141.41 | 141.32 | **67.14** |
| ARCTIC | 71.54 | 71.97 | 71.63 | **34.20** |

这个表很值得注意。

WAM：

\[
\approx\text{static hand}
\]

而一个最简单的：

\[
\boxed{\text{current hand → goal hand 线性插值}}
\]

反而远远更强。

这说明现在 network 根本没有学好：

> “goal endpoint → 中间 trajectory”

这件事情。

---

# 更关键：最新 diagnostic 直接证明 goal 基本被忽略

最新做了一个很漂亮的测试：

保持当前 scene 完全不变，只把 goal endpoint：

\[
+0.1m\quad X
\]

然后看预测的 hand endpoint 跟不跟着变。

如果模型真的用 goal，理想 response gain 应该接近：

\[
1.
\]

结果：

### 原 1e-4 model

\[
1.25\times10^{-7}
\]

### 1e-3 model

\[
2.89\times10^{-6}
\]

### 连固定 6-window 都拟合得很好的 model

\[
6.16\times10^{-5}.
\]

也就是说给 goal 整体平移：

\[
100\text{ mm},
\]

模型输出只动了大约：

\[
0.0004\sim0.036\text{ mm}.
\]

实际上：

\[
\boxed{\text{几乎完全无视 goal。}}
\]

这个才是现在最核心的问题。

---

## 甚至 attention 也不是完全没看

有趣的是，他们又检查了 memory attention：

scene / goal / action token norm 都差不多：

\[
\sim20.
\]

Trajectory decoder 对 goal 的 attention mass：

\[
0.162\sim0.171.
\]

所以不是简单：

> “goal token 太小，被 scene token 数量淹没了。”

它**确实 attend 到 goal token**，但没有学习到 goal 的几何意义。

这个区别非常重要。

---

# 所以现在问题已经收敛得比较清楚

不是：

- PTv3 太小；
- 参数不足；
- PW checkpoint 不好；
- LR 单纯太低；
- goal 完全没 attention；
- world branch 偷看 G。

而更像是：

\[
\boxed{
\text{Goal 的几何表示 / 训练合同本身有问题}
}
\]

现在网络容易学成：

\[
H\rightarrow\text{未来手运动的平均预测}
\]

而不是：

\[
(H,G)\rightarrow\tau.
\]

---

# 我认为这里还有一个更根本的问题

当前代码的 goal 是：

\[
\boxed{\text{hand start/end}}
\]

这跟我们后来讨论的最终目标不一致。

我们的真正需求是：

\[
\boxed{
G=(P^{object}_{start},P^{object}_{goal})
}
\]

或者当前 scene 已经包含 start 时，甚至只给：

\[
P^{object}_{goal}.
\]

所以我反而**不建议现在继续围绕 hand-goal encoder 调结构、LR、attention**。

否则可能花很多实验把一个我们最终不会用的 condition 修好。

---

# 我觉得当前最合理的下一步应该回到我们刚刚确定的语义

也就是改 GoalEncoder contract：

现在：

\[
(start\ hand,\ goal\ hand)
\rightarrow Z_G
\]

改成：

\[
\boxed{
(start\ object\ points,\ goal\ object\ points)
\rightarrow Z_G
}
\]

训练目标：

\[
(H,G_{obj})\rightarrow\tau_{hand}.
\]

而 world branch保持现在正确的：

\[
(H,\tau_{\rm executed})\rightarrow E.
\]

这才真正是在回答我们的问题：

> **给物体一个短程目标状态，手应该怎么运动？**

而不是：

> 给手一个终点，补出手的中间轨迹。

---

## 还有一个非常好的消息

`cm-pointflow-effect-pretrain` 已经在今天被正式 merge 回 main：

```text
4d7a2784 Merge cm-pointflow-effect-pretrain as current PW implementation
862a2ee5 Complete PW branch integration
```

旧 worktree 也退休了。

所以现在：

- PointWorld source；
- pretrained checkpoint；
- OakInk2/GRAB/ARCTIC data；
- GoalPointWAM；

都已经在同一个 `main`。

工程上继续改已经方便很多。

---

## 另外 official DExplore 跨 reference 也已经补测了

README 顶部新增了一组：

**12 条 corrected reference、同一个 official Inspire ckpt。**

结果：

- 11 条都是 **8/8 hold**；
- 手机只有 **2/8**；
- controlled placement 合计 **68/96**。

其中：

- s3/s9 airplane；
- toothpaste；
- water bottle；

都是：

\[
8/8\text{ placement}.
\]

Retake airplane：

\[
1/8
\]

duck：

\[
2/8.
\]

phone reference position RMSE：

\[
1.56m.
\]

这说明之前我们的判断也是对的：

\[
\boxed{\text{official DExplore 并不是跨所有 trajectory 都100%。}}
\]

后续 WAM 是有 headroom 的。

---

# 所以我对现在仓库状态的判断

**好的一面：**

1. WAM 已经真正独立成 task；
2. PW backbone 已经 merge + pretrained transfer；
3. goal-blind world branch接法正确；
4. full action token memory 已修；
5. model 能拟合固定数据；
6. goal-response 有专门 causal diagnostic，不再只看 loss。

**当前真正 blocker：**

\[
\boxed{\text{Trajectory branch没有真正学会使用 goal。}}
\]

而且目前 goal 还是 **hand endpoint**，不是我们最终想要的 **object point goal**。

所以我现在不会继续加训练量。

下一步应该先把 **Goal contract 从 hand endpoint 改成 object start/goal point set**，然后先做一个非常便宜的 response test：

\[
P^{goal}_{obj}\rightarrow P^{goal}_{obj}+10cm\,x
\]

看：

\[
\hat\tau
\]

是否出现明确、合理的响应。

**这个过不了之前，不应该再上机器人、DAgger、RL，也不应该继续扩大视频数据。**