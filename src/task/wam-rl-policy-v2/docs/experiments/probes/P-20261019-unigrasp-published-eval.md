---
schema: ref2dex.probe.v2
probe_id: P-20261019-unigrasp-published-eval
experiment_id: P-20261019-unigrasp-published-eval
date: 2026-10-19
task: wam-rl-policy-v2
branch: main
git_commit: recorded-in-run-manifest
claim_id: C1
hypothesis_family: HF-unigrasp-release-evaluation
probe_index_in_family: 1
seed_pool: probe
seeds: [246]
decision_changed_if_positive: use released UniGraspTransformer as an additional local evaluator and investigate its data and observation design
decision_changed_if_negative: resolve release dependencies and protocol mismatch before using it as a baseline
status: RUNNING
run_id: unigrasp-published-jar-20261019-r1
---

Decision：用户授权仅评估UniGraspTransformer，不训练。最便宜的判别是官方状态通用
Transformer权重 + 动态PointNet编码器严格加载，在已审计的单jar物体/.06上运行
256并行环境、200步官方test循环；GPU6、graspenv、每进程900秒/4GiB上限。
后续仅在接口可运行时，按结果扩展有限面板。全程禁止optimizer.step，不采集训练缓存，
不保存视频。主六卡训练不改。

上游固定为`58753be89024267bb2348d0bf9a224fadbe5f35a`，以Git子模块登记。
入口`tools/run/evaluate_unigrasp.py`把上游复制到独立runtime，执行官方run_online.py
--algo dagger_value --test，保留actor/encoder、原生400D观测、动作和成功判据。
安装缺失的纯Python/二进制依赖到此实验独立deps目录，不修改现有环境。
复用已审计的官方mesh/feature/prior；官方FPS ZIP只按HTTP Range读取所需物体数据。

## 已知协议限制

- 官方85MiB通用权重名称为model_best_isaacgym_3.pt；本机已安装IsaacGym Preview4，
  不是论文Preview3原环境。本次首先是兼容性评估。
- 官方meshdatav3_init.zip下载页、uc及usercontent均返回404。
  [官方issue #26](https://github.com/microsoft/UniGraspTransformer/issues/26)也报告文件缺失，
  未见替代链接。显式采用上游支持的Modes.static_init=False，保留其动态reset实现；
  因此不具备论文的固定测试初态，不把成功率当作严格论文复现或与当前WAM的配对比较。
- 兼容修改仅在runtime中记录diff：np.float别名、PointNet包导入统一前缀，以及上述reset
  配置选项；增加只读结果记录及禁止训练更新的守卫。

产物：`outputs/wam-rl-policy-v2/unigrasp-published-jar-20261019-r1/`。
准备阶段依赖：`outputs/wam-rl-policy-v2/unigrasp-eval-20261019-r1/deps/`。

准备r1发现官方下载FPS ZIP内部前缀为meshdatav3_pc_fps_new，修复路径映射后r2准备成功；r1未运行物理评估。补齐GPUtil/h5py/ipdb/addict/yapf/termcolor/nvidia-ml-py至独立deps目录，原环境未改。移除未使用sorcery debug导入，PointNet及Transformer严格加载。GPU6开始r2，256环境，采样约3.4GiB/53%。在接口已执行的基础上，追加单camera-9726.../.10、256环境/seed246/GPU7，同样900秒上限，run_id unigrasp-published-camera-20261019-r1；用于区分单个形状偶然性与可执行的本地多形状策略，不作为全面基准。

动态fallback首轮：jar 57/256（22.27%），camera 29/256（11.33%），均optimizer_steps=0。审计发现原始Modes.static_init切换同时改变hand起点y=0.08→0，不能视作只更换物体初始化来源。追加Decision对照：相同两物体、seed246/256环境，动态物体初始化不变，只将手部构造起点恢复训练默认y=0.08m；GPU6/7、每进程900秒、同权重与官方test循环。run_id unigrasp-published-{jar,camera}-handpose-20261019-r1。结果无论正负均结束此轮兼容性尝试，不做基于成功率的参数搜索；需要原始静态测试包才能给出严格复现结论。
