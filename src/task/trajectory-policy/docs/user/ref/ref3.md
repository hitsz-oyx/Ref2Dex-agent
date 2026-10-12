第一阶段完全不要 PointWorld：
\[
\boxed{
(H,G)
\rightarrow
\hat\tau
\rightarrow
\hat q^{des}
}
\]
用 competent DExplore rollout 做 BC。
监督：
\[
L_\tau
+
\lambda L_{q^{des}}.
\]
先把一个真正能抓东西的 trajectory policy 做出来。
这是 baseline。
然后第二阶段才接 PointWorld：
\[
(H,G)
\rightarrow
\pi_\theta
\rightarrow
\tau
\]
与此同时：
\[
(H,\tau)
\xrightarrow{PW}
z_{\rm phys}.
\]
critic：
\[
\boxed{
Q_\phi(H,G,\tau,z_{\rm phys})
}
\]
然后通过 RL 告诉 actor：
哪些 \(\tau\) 不只是像 teacher，而是真正在物理环境里好用。

这时候 PointWorld 的作用才真正对应我们的核心假设：
\[
\boxed{
\text{理解动作后果}
\rightarrow
\text{帮助策略优化}
}
\]