可以，把我们现在路线 B 的训练方法压缩成下面这套。

### 1. 训练目标

我们要训练的是一个 **execution / retarget model**：

\[
\boxed{
R(\tau^{hand}_{t+1:t+T},\,s_t)\rightarrow A_{t:t+T-1}
}
\]

其中：

- \(\tau^{hand}\)：未来手部关键点轨迹；
- \(s_t\)：当前机器人状态，至少包含 \(q_t,\dot q_t\)；
- \(A\)：**完整机器人控制命令**，不是 residual。

也就是说学：

> “我希望未来手这样运动，现在机器人是这个状态，那我应该发什么控制？”

---

### 2. 数据怎么来

直接大量 rollout，不需要 fork。

每条普通机器人轨迹天然有：

\[
s_t
\]

\[
A_t,A_{t+1},...
\]

以及实际执行之后得到的：

\[
\tau^{hand}_{t+1:t+T}
\]

所以自动组成训练样本：

\[
\boxed{
(\tau^{hand}_{future},s_t)
\rightarrow
A_{future}
}
\]

一条 500 多步 rollout 可以产生几百个 window。

而且这里**不要求 rollout 成功**，因为我们学的是执行映射，不是在学“什么动作好”。

---

### 3. rollout 怎么采

不要纯随机 `[-1,1]`，而是做**结构化随机控制**，覆盖：

- wrist 平移；
- wrist 旋转；
- 手指开合；
- 不同 finger target / preload；
- 接触前；
- 接触建立；
- 稳定抓持；
- wrist + finger 联合运动。

重点增加接触区间的数据，因为现在最难的就是：

\[
\tau^{finger}\rightarrow A^{finger}
\]

---

### 4. 模型输入

第一版尽量贴近 PointWAM：

\[
\boxed{
\tau^{hand}_{future}+q_t+\dot q_t
}
\]

暂时不要一开始就塞太多东西。

如果 finger/contact 仍然学不好，再加：

\[
A_{t-1}
\]

和 hand-object relative geometry。

即：

\[
R(\tau,q,\dot q,A_{t-1},I_t)\rightarrow A
\]

---

### 5. 输出

输出直接是：

\[
\boxed{A_{\text{full}}}
\]

不是：

\[
\delta A
\]

也不再依赖：

\[
A=\pi_{\text{baseline}}(H)+\delta A
\]

这样 execution model 和 baseline policy 解耦。

---

### 6. target 怎么定义

监督标签必须用 rollout 中**实际发送给控制器的 commanded action / PD target**。

不要用：

\[
q_{t+1}^{actual}
\]

代替 command。

因为接触时：

\[
q_{\text{actual}}\neq q_{\text{command}}
\]

它们之间的差值恰恰对应夹持 preload。

---

### 7. action 表示

最好像 PointWAM 一样，相对于当前状态表示：

\[
\Delta A_\tau
\]

例如：

- wrist position：相对当前位置；
- wrist rotation：相对旋转；
- finger target：相对当前 finger q。

同时：

\[
\boxed{\text{每个 horizon、每类 action 单独 normalization}}
\]

不要把 wrist translation、rotation、finger 全塞进同一个统一尺度。

---

### 8. loss

第一版直接：

\[
\boxed{
L_A=
\|\hat A-A^{GT}\|_1
}
\]

比 MSE 更接近 PointWAM。

同时可以分别报告：

- wrist translation error；
- wrist rotation error；
- finger command error；
- contact-onset finger error；
- hold-stage finger error。

尤其不要只看一个总 action loss。

---

### 9. 训练数据划分

必须按 **episode split**：

```text
train rollout
val rollout
test rollout
```

不能把同一条 trajectory 的不同 window 随机分到 train/test。

而且这次不要再只有 `3 train / 1 val / 1 test`。

既然 rollout 便宜，我建议直接采一个比较像样的小规模集，比如：

\[
100+\text{ episodes}
\]

先做第一版；如果便宜，可以再扩大。

---

### 10. 最关键的验证

先不要训练：

\[
H\rightarrow\tau
\]

而是直接给 execution model **真实 GT future hand trajectory**：

\[
\boxed{
\tau^{hand}_{GT}
\rightarrow
R
\rightarrow
A
}
\]

然后真正放进 Gym 执行。

我们要看的是：

- 能不能稳定抓住；
- held 是否接近 teacher；
- finger contact 是否保持；
- clipping 是否严重。

例如 teacher：

\[
held=484
\]

那我们希望至少：

\[
held\gtrsim436
\]

也就是 90% 左右。

只有这一步通过，才证明：

\[
\boxed{\tau^{hand}\text{ 是一个可执行的 action representation}}
\]

---

### 11. 之后才接完整路线

执行模型通过以后，再训练高层：

\[
H\rightarrow\tau^{hand}
\]

得到 baseline：

\[
H
\rightarrow
\tau^{hand}
\rightarrow
R
\rightarrow
A
\]

最后再接 PointWorld / evaluator：

\[
H
\rightarrow
\{\tau_i\}
\rightarrow
PW
\rightarrow
Q
\rightarrow
\tau^*
\rightarrow
R
\rightarrow
A
\]

所以当前真正应该做的就两件事：

\[
\boxed{\text{大量结构化 full-action rollout}}
\]

然后：

\[
\boxed{\text{训练 }R(\tau^{hand},s_t)\rightarrow A_{\text{full}}}
\]

先把 GT-\(\tau\) execution upper bound 做通，再碰 PointWorld 和候选选择。