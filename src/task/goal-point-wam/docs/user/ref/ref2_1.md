看了，最新 `main` 已经到 **`91e1d3c4`：`Record partial WAM thaw outcome and audit unsupported autonomous recovery states`**。现在问题比上一轮更清楚了：**startup 已经基本修住，但 WAM 一旦进入自己访问到的状态，策略仍然很快失效；继续单纯解冻更多 backbone 暂时没有解决。**

startup continuation 的结果其实很好。它把 frame0 的 WAM→R PD XYZ 误差从约 **225 mm 降到 11.7 mm**，第一步 wrist 偏差也从 **25.3 mm 降到 1.65 mm**。离线 startup validation 的 PD XYZ p95 也从约 **200 mm 降到 18.3 mm**。所以之前“起步样本太少”这个诊断是成立的。

但完整 542-step 闭环依然失败：teacher 和 GT-\(\tau\)-R 都能 **hold 441 帧并完成 placement**，WAM-R 还是 **0 held**，而且 clipping 到 **4.06%**。关键是它不再是 frame0 立即崩，而是在：

\[
t\approx 9\sim12
\]

就开始重新漂移；tick16 时 PD XYZ 已经约 **58.5 mm**，wrist 偏差约 **53.7 mm**。也就是说：

\[
\boxed{\text{startup distribution 已修，student visited-state distribution 仍没修。}}
\]

这实际上已经把问题从“初始化预测差”推进到了很典型的 **covariate shift / recovery-state coverage**。

随后仓库真的做了我们前面说的 teacher-query 思路：让 WAM 先跑到自己的状态，再让官方 DExplore 在**那个真实 visited state**上给 expert PD。这里没有拿 nominal teacher future hand 冒充 recovery label，而是实时查询 expert。tick16 接管以后，expert 能重新抓起来，达到 **435 held frames**，但最后 placement 失败，因此整条 recovery 只能算 `UNCLEAR`，不能把后半程所有 expert labels 都纳入训练。

因此仓库只取了**早期 approach/recovery 段**的 audited expert PD 做一个 PD-only correction。结果并不好：训练状态 PD XYZ p95 从

\[
136.6\rightarrow44.3\text{ mm}
\]

但 held live validation 只从

\[
404.2\rightarrow384.5\text{ mm},
\]

几乎没解决，而且 nominal PD 还从 **25.6 mm 恶化到 82.8 mm**。所以没有继续闭环。

随后又检查了是不是冻结的 \(R\) 本身表达能力不够。结论是否定的：对每个 recorded state 单独优化 \(\tau\)，通过同一个冻结 \(R\)，PD 能做到 train/val 约 **0.61 / 0.87 mm**。所以：

\[
\boxed{R\text{ 能表达正确 PD，主要问题仍在 actor 如何从 }(H,G)\text{ 产生正确 }\tau.}
\]

接下来按你之前提过的“逐步解冻”做了一个小规模 partial thaw，只解冻 PTv3 最后的 decoder stage `dec0`，外加 trajectory head / hand adapter。结果同样没有解决关键问题：训练 live PD 从 **44.2→33.3 mm**，但 held validation 反而：

\[
384.8\rightarrow398.7\text{ mm}.
\]

更关键的是把 live 状态拆开以后很明显：

\[
\text{pure-WAM role2, ticks32:48}:
433.1\rightarrow435.2\text{ mm}
\]

基本完全没改善；

而：

\[
\text{expert-recovered role3, ticks64:80}:
31.9\rightarrow33.8\text{ mm}
\]

本来就比较容易。

这说明现在最大的 blocker 已经不是“head 容量不够”或者“再解冻一层就行”，而是：

\[
\boxed{
\text{WAM 自己跑偏后产生的 autonomous states，
在现有监督里仍然是严重 OOD。}
}
\]

所以我认为现在仓库里最重要的新判断是：**我们已经到了该做真正 robot-domain data aggregation 的阶段，而不是继续 nominal teacher BC。**

也就是说，后面应该更接近：

\[
\text{WAM rollout}
\rightarrow
s_t^{student}
\rightarrow
\text{teacher query / actual consequence}
\rightarrow
\text{重新训练 WAM}
\]

而不是只在原始 DExplore nominal rollout 上继续微调。

另外两条并行线也有进展。EgoDex 的远端 10k mixed auxiliary training 已经同步完成，但它仍主要是 hand-trajectory auxiliary，不是完整 robot/world fine-tune。视频 point supervision 这边已经把 shared-backbone 接口做通，并且把之前 33 mm 的跨 frontend depth 偏差定位为 preprocessing 不一致；改成同一次 offline clip reconstruction 后，部分 train/dev window 已通过几何审计，现在已经预注册了首个 **300-update offline video mixed learning probe**，但最新 HEAD 还没有看到这个 probe 的最终结果。

所以现在整条主线我会这样判断：

\[
\boxed{
\text{Human WAM 本身已经有 goal 信息}
\rightarrow
\text{startup robot adaptation 已成功}
\rightarrow
\text{真正 blocker = student visited-state / recovery distribution}
}
\]

这其实比之前清楚很多。下一步不应该继续“再解冻更多 PTv3 看看”，而应该正式把**机器人自主 rollout 产生的状态和真实 consequence / expert PD** 纳入 WAM robot fine-tuning 数据闭环。这样才是真正对应我们前面讨论的：

\[
\text{human pretrain}
\rightarrow
\text{robot teacher FT}
\rightarrow
\text{student consequence aggregation / DAgger}
\rightarrow
\text{RL}.
\]