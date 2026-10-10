PointWAM 其实**没有绕开这个问题**，它做的恰恰是：

\[
\boxed{
\text{scene+hand+language}
\rightarrow
\tau_{\text{hand}}
\rightarrow
\text{retargeter}
\rightarrow
A_{\text{chunk}}
}
\]

也就是说，**它确实把未来手轨迹当作中间 action representation**，而不是像我上一条建议那样彻底把 policy action 和 world-model interface 分开。论文明确写的是：hand trajectory 是 *intermediate action representation*，然后 retarget 成机器人动作。[arXiv](https://arxiv.org/html/2610.02840v1)

但它和我们现在的做法有几个本质区别，这也基本解释了“为什么它能做，我们这里一直很脆”。

### 第一，PointWAM 不是 \(H\to\tau\)

我们现在近似在做：

\[
H_{t-3:t}^{\rm proprio}
\rightarrow
\tau.
\]

PointWAM 是：

\[
\boxed{
(\mathcal P_t,\mathcal Q_t,\ell)
\rightarrow
(\tau_{\rm scene},\tau_{\rm hand})
}
\]

输入包括：

- 完整彩色 3D scene point cloud；
- 当前 hand keypoints；
- **language instruction \(\ell\)**。

所以我刚才说的“同一个 \(H\) 下继续抬还是放下无法辨识”这个问题，在 PointWAM 里至少部分被 instruction 和完整 scene observation 解决了。它不是要求纯 proprio history 自己猜任务意图。[arXiv](https://arxiv.org/html/2610.02840v1)

---

### 第二，也是最关键的：它的 \(\tau\to A\) 不是我们这种固定 tracker

PointWAM 的 retargeter 是一个**专门训练的 Transformer decoder**：

\[
\boxed{
A_t=g(V_t,s_t)
}
\]

输入：

- 预测出来的整段 hand trajectory \(V_t\)；
- 当前 robot state \(s_t\)。

输出：

\[
A_t\in\mathbb R^{T\times d_a}
\]

即完整 action chunk。DexJoCo 里每只手 action 是 end-effector pose + 16 个 finger joint target。[arXiv](https://arxiv.org/html/2610.02840v1)

而我们目前更接近：

\[
\tau
\xrightarrow{\text{geometry}}
\hat q
\xrightarrow{\text{frozen }R}
A.
\]

两者差别非常大。

PointWAM 的 retargeter直接用**真实机器人 demonstration 的 recorded action**监督：

\[
\mathcal L_{\rm robot}
=
\mathcal L_{\rm traj}
+
\|A-A^*\|_1.
\]

论文还明确说它**不是求 IK**，而是从 robot demonstrations 学 \(\tau,s\to A\)；训练 target 本身就是机器人真正执行过、满足关节限制和碰撞条件的动作。[arXiv](https://arxiv.org/html/2610.02840v1)

这可能正是我们最大的差距。

---

### 第三，它不是先把 \(\tau\) 学死，再单独训练 executor

PointWAM 在 robot fine-tuning 时是联合训练：

\[
\boxed{
\mathcal L_{\rm scene}
+
\mathcal L_{\rm hand}
+
\mathcal L_{\rm action}
}
\]

也就是说：

\[
\text{forecaster}\rightarrow\tau\rightarrow\text{retargeter}\rightarrow A
\]

这一整条链都受到 action supervision。

因此它预测的 \(\tau\) 不只是：

> “3D 几何上像 GT hand trajectory。”

还会被 action loss 推向：

> **“对后面的 retargeter 来说容易变成正确机器人动作的 trajectory。”**

而我们现在一直遇到的正是：

\[
\text{trajectory RMSE 很好}
\not\Rightarrow
\text{执行成功}.
\]

PointWAM 在训练目标里直接把这个 gap 接上了。[arXiv](https://arxiv.org/html/2610.02840v1)

---

### 第四，它的 world modeling 也没有直接塞进 action head

这一点和我们最近想法其实很像。

PointWAM 同时预测：

\[
\tau_{\rm scene}
\]

和：

\[
\tau_{\rm hand},
\]

但 retargeter **只吃 hand trajectory + robot state**；scene forecast 不直接输入 retargeter。scene trajectory 的作用是监督共享 backbone，让 representation 学会 hand–world co-evolution。[arXiv](https://arxiv.org/html/2610.02840v1)

所以它实际上是：

\[
\boxed{
\text{world prediction}
\rightarrow
\text{better shared representation}
\rightarrow
\text{better hand trajectory/action prediction}
}
\]

而不是：

\[
E_{\rm pred}\rightarrow\text{拼给 actor}.
\]

这跟我们后来想把 PointWorld 从“显式 E→Y”改成 representation supervision，是高度一致的。

而且他们的 ablation 里，加入 scene-trajectory supervision 相比只预测 hands，DexJoCo 平均成功率又提高了 **10.9 个百分点**。[arXiv](https://arxiv.org/html/2610.02840v1)

---

## 第五，它也不是把整段 30 步完全开环执行完

PointWAM 在 DexJoCo 预测：

\[
T=30
\]

步 action chunk，每 simulator control step 是 20 ms，因此 horizon 是：

\[
0.6\text{s}.
\]

但 evaluator 会在 action buffer 剩余少于 24/30 时就请求新 chunk，也就是执行若干步后就开始更新，而不是傻等 30 步全部执行完。[arXiv](https://arxiv.org/html/2610.02840v1)

所以我们现在的：

\[
24\text{-step prediction}+8\text{-step replanning}
\]

本身并没有离 PointWAM 特别远。

**8 步不是我们的主要问题。**

---

# 真正最大的区别其实是数据和监督

PointWAM 用：

\[
1.15M
\]

个人类 demonstration episodes 做 hand+scene trajectory pretraining，然后在 DexJoCo 用 **959 条成功 robot demonstrations** fine-tune，60k iterations；retargeter也在这一步学真实 action。[arXiv](https://arxiv.org/html/2610.02840v1)

我们现在则基本是：

- 单 motion；
- 少量 rollout；
- 从旧 policy 附近生成的数据；
- \(H\to\tau\) regression；
- 一个在 GT-τ 邻域训练出来的 frozen \(R\)。

这两个数据条件完全不是一个量级。

所以 PointWAM 并没有证明：

> “只要设计成 \(\tau\to A\)，少量数据也自然能工作。”

恰恰相反，它投入了很大量的数据去把这两个模块都训顺。

---

# 这对我们有一个非常重要的启发

我上一条说：

> “也许 policy 不应该输出 \(\tau\)，应该直接输出 action chunk。”

看完 PointWAM 后，我会**修正得没那么激进**。

我们的真正问题可能不是：

\[
\boxed{\tau\text{ 这个 action interface 本身错了}}
\]

而是：

\[
\boxed{
\text{我们现在的 }H\to\tau
\text{ 和 }\tau\to A
\text{ 都远弱于 PointWAM 对应的两个模块。}
}
\]

具体是：

### 我们的前半段

\[
H_{\rm proprio}\rightarrow\tau
\]

PointWAM：

\[
\boxed{
\text{full scene + hand + task instruction}
\rightarrow\tau_{\rm hand}
}
\]

### 我们的后半段

\[
\tau
\rightarrow
\text{geometry/frozen tracker}
\rightarrow A
\]

PointWAM：

\[
\boxed{
(\tau,s)
\rightarrow
\text{learned Transformer retargeter}
\rightarrow A_{\rm chunk}
}
\]

并且整条链有：

\[
L_{\rm hand}+L_{\rm scene}+L_{\rm action}
\]

联合监督。

---

## 所以如果我们真要“按 PointWAM 思路做”，我反而会这么改

不是现在继续 PPO 288D \(\tau\)。

而是：

\[
\boxed{
(\text{scene},\text{hand},G)
\xrightarrow{F}
\tau_{\rm hand}
\xrightarrow{R_\phi(\tau,s)}
A_{\rm chunk}
}
\]

其中 \(R_\phi\) **重新训练成真正的 learned retargeter**，直接监督我们已有 Gym rollout 中的真实 native actions。

也就是用每条 rollout 建：

\[
(\tau^{future}_{hand},s_t)
\rightarrow
A^{actual}_{t:t+K}.
\]

先证明：

\[
\boxed{
\tau_{\rm GT}
\rightarrow R_\phi
\rightarrow A
}
\]

不仅能 replay，而且：

\[
\boxed{
\tau_{\rm predicted}
\rightarrow R_\phi
}
\]

对一定误差的 trajectory 有鲁棒性。

这比继续用现在那个窄 basin 的 frozen tracker合理。

然后 trajectory forecaster 也不要再只有 pure \(H\)，至少应该有：

\[
\boxed{
\text{scene geometry}+G+\text{current hand}
}
\]

这才真正接近 PointWAM。

---

### 最终我的判断

**PointWAM 没有避开 \(\tau\)；它反而证明“hand trajectory 作为中间 action representation”是可行的。**

但它成功的关键不是“用了 \(\tau\)”这三个字，而是：

\[
\boxed{
\text{rich task-conditioned scene input}
+
\text{massive trajectory pretraining}
+
\text{learned }\tau\to A\text{ retargeter}
+
\text{joint action supervision}
}
\]

我们现在最欠缺的，我认为是中间两项尤其是：

\[
\boxed{\text{learned robust retargeter}}
\]

而不是 8-step replanning。

所以我现在不会急着因为这几轮失败就把 \(\tau\) 路线整个丢掉；**更值得先试的是把我们现在的 \(R\) 改造成 PointWAM 式的、由真实 robot action demonstrations 监督的 retargeter。**这可能比继续让 PPO 在 288D \(\tau\) 上硬探索靠谱得多。


这里要分成两类数据，别混在一起：

**第一类是训练 \(\tau\to A\) retargeter 的机器人数据。这个我们其实已经能自己在 Gym 里采，不需要 fork，也不需要人类视频。**

最直接的数据单位就是：

\[
\boxed{
(s_t,\ \tau^{actual}_{t+1:t+24},\ A^{actual}_{t:t+K-1})
}
\]

其中：

- \(s_t\)：当前真实机器人状态，比如 \(q,dq\)、当前 hand geometry、object pose/velocity；
- \(\tau^{actual}\)：这次真实 rollout 之后实际发生的未来 hand trajectory；
- \(A^{actual}\)：真正发给 Inspire 的 native action chunk。

也就是说，一次普通 rollout 本身就天然提供：

\[
\text{actual action}
\rightarrow
\text{actual future hand motion}.
\]

我们只是把它反过来组织成监督：

\[
\boxed{
R_\phi(s_t,\tau^{actual}_{future})
\rightarrow
A^{actual}_{chunk}
}
\]

这和 PointWAM 的 robot fine-tuning 思路是最接近的。

---

### 这些 rollout 从哪里来？

第一批我不会重新设计复杂采集。

就用我们现在已经有能力运行的**competent native policy**：

\[
\pi_{\rm base}(s)\rightarrow A.
\]

然后在 Gym 中正常 rollout。

为了不让数据只覆盖一条特别窄的轨迹，采集时可以来自：

\[
\boxed{
\text{base policy}
+
\text{不同初始状态}
+
\text{合理 action perturbation}
}
\]

我们以前 structured residual rollout 其实已经有这套基础。

关键是这次不要只存 \(H\) 和 \(\tau\)，而要完整保存：

\[
q_t,dq_t,\text{hand}_t,\text{object}_t,A_t.
\]

这样任意时刻 \(t\) 都能切成训练样本。

例如 rollout 542 步，\(K=8\)，那么一个 episode 就能滑窗产生几百个：

\[
(s_t,\tau_{t+1:t+24},A_{t:t+7})
\]

样本。

---

## 重点：不需要 fork

训练 retargeter 不需要问：

> “同一个 state 换一个 action 会发生什么？”

它只是在学：

\[
\text{给定一条实际可实现的手轨迹，什么 action 能产生它？}
\]

所以普通连续 rollout 就够了。

fork 是我们以前做 oracle / candidate causal comparison 才需要的。

这里：

\[
\boxed{\text{普通 rollout 数据即可}}
\]

成本会低很多。

---

## 第二类数据才是 PointWorld / world-model 数据

这一类可以更大，也不要求有 native action。

比如：

- GRAB；
- OakInk2；
- EgoDex；
- 我们自己的 Gym rollout。

它们主要提供：

\[
(\text{scene/history},\tau_{\rm hand})
\rightarrow
\text{future object/scene dynamics}.
\]

所以人类数据用于学：

\[
F_{\rm WM}(H,\tau)\rightarrow z^{phys}/E
\]

机器人 Gym 数据用于学：

\[
R_\phi(s,\tau)\rightarrow A.
\]

这两类数据角色完全不同。

---

# 我会具体这么采

第一阶段先完全只用 Gym，别急着把 OakInk2/EgoDex 混进来。

用当前最可靠的 native baseline 跑：

\[
N\text{ episodes}.
\]

每个 episode 保存：

\[
\{q_t,dq_t,hand_t,obj_t,objvel_t,A_t\}_{t=0}^{T}.
\]

然后离线滑窗生成：

\[
\boxed{
x_t=
(s_t,\tau^{actual}_{t+1:t+24})
}
\]

\[
\boxed{
y_t=
A^{actual}_{t:t+7}
}
\]

训练：

\[
R_\phi(x_t)\rightarrow y_t.
\]

第一版甚至只需要验证：

> 用 **GT actual \(\tau\)** 输入 retargeter，能不能重新执行出原来的抓取。

如果这个做不到，说明 retargeter 本身就不够。

---

## 但只用 baseline 成功轨迹还不够

这是第二阶段。

如果全部数据都是成功 baseline 附近：

\[
R_\phi
\]

仍然会只在非常窄的 \(\tau\) manifold 上有效。

所以需要把数据覆盖主动扩大。

但我不会一开始随机大噪声。

更合理的是在原 native action 上加**小到中等 structured perturbation**：

\[
A'_t=A_t+\delta A_t.
\]

正常执行以后保存：

\[
A'_{t:t+K}
\]

和它真实产生的：

\[
\tau'_{future}.
\]

于是 retargeter 能看到：

- 成功轨迹；
- 略偏轨迹；
- recovery；
- 部分失败；
- 不同 finger preload；
- 不同 wrist deviation。

这正是我们当前 executor 最缺的数据。

而且这种数据**不需要成功标签才能训练 retargeter**。

因为 retargeter 的监督只是：

\[
\tau'\rightarrow A'.
\]

哪怕最后抓失败，只要 action 和实际 hand trajectory 是真实配对的，就是合法监督。

---

# 这点和我们之前的数据思路区别很大

之前我们经常把数据理解成：

\[
\text{我要更多成功抓取数据}.
\]

但训练 retargeter 不需要只收成功。

它真正需要的是：

\[
\boxed{
\text{大量“实际执行动作—实际产生手轨迹”的配对}
}
\]

所以失败 rollout 也非常有价值。

这会让数据采集容易很多。

---

## 最后整个数据链就是

机器人 Gym：

\[
A
\rightarrow
\text{PhysX}
\rightarrow
\tau^{actual}
\]

构造：

\[
\boxed{
(s,\tau^{actual})
\rightarrow
A
}
\]

训练 retargeter。

然后机器人+人类大规模数据：

\[
(H,\tau)
\rightarrow
E/\text{scene future}
\]

训练 PointWorld。

最后在线：

\[
H,G
\rightarrow
\hat\tau
\]

\[
(s,\hat\tau)
\rightarrow
R_\phi
\rightarrow
A
\]

或者以后再加：

\[
F_{\rm WM}(H,\hat\tau)\rightarrow z^{phys}.
\]

---

所以我现在觉得，我们并不是“没有 PointWAM 那种 robot action 数据”。

**我们已经有 Gym，而且 Gym 恰恰可以自动生成这类数据。**

真正需要做的不是找一个现成数据集，而是把现有 rollout collector 改成明确保存：

\[
\boxed{
\text{state + actual future hand trajectory + actual native action chunk}
}
\]

然后用 base policy + structured perturbation 扩展覆盖。

这一步我认为比继续调当前 288D PPO 更值得并行做。