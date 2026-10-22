# S2e：WAM-τ vs 几何-τ，同 v7 接口（2026-10-22）

**结果：WAM 1/5、几何0/5，WAM描述性领先1例/20个百分点；没有达到预登记≥2/5的优势门。**
这批不支持“预测未来知识已经明显改善执行质量”，也没有证明两种τ信息等价或WAM无用。
按当前资源决策，不追加几何/预测跟踪筛选；保留后续研究WAM对RL样本效率/泛化的价值，
但须先解决执行器资格，本轮不启动RL或重训。此结论仅针对本单轨迹、固定v7与两种具体τ来源。

## A：几何定义与禁止live手贴合

新增`src/wam_rl_policy_v3/geometry_tau.py`：从S2b完整205状态参考逐帧计算
`L[j]=inv(T_ref_object[j])@H_ref[j]`；当前30Hz时钟t取j=t+1..t+24，再生成
`G[k]=T_live_object@L[j]`。**目标API没有live手参数，没有手相位匹配或以当前手做平移/旋转对齐。**
R输入需差分时才用`τ=G[:8]-H_live`，它只是编码；独立反事实改变live手后重建目标误差
≤2.22e-16m。共同的腕姿约简/IK读取live手属于执行转换，不改变几何绝对目标G。

完整24帧供与WAM完全相同的`reduce_to_wrist_plan`、`grasp_intent_from_retargeter`和
`MeasuredGraspConsumer`消费；两臂均以同一冻结R适配腕部/手指意图。几何τ来源不经过网络，
下游共享R仍是网络。每个未来帧使用**它自己的参考物体坐标系**，消去source刚体抬升；
接触后由共同v7按物体任务目标闭环抬升。不同于S2d“参考当前锚点保留整段source抬升”。

相位按参考时钟，无live手选择；不足完整24帧时invalid，不复制终帧。H4/encoder gate和
approach/close预测触发规则对两臂相同。几何有效658/827controls，WAM1070/1114；差异
包括几何参考过期与实际状态domain gate，不宣称两臂拥有同等未来覆盖或同等任务信息。

## B/C：唯一配对n=5批次

| 同v7链；block0→4 | 最长持握elapsed秒 | ever/final success | 输出clip/controls | 内部R clip | 原关节安全中止 |
|---|---|---|---|---:|---|
| WAM11000-τ | [1,0,0,0,0] | 1/5；1/5 | 0/1114 | 0/1114 | 2/5 |
| 几何-τ | [0,0,0,0,0] | 0/5；0/5 | 0/827 | 5/827 | 4/5 |

输出裁剪与内部R裁剪分开：几何内部5次在意图适配前已clip，不能把最终输出0裁剪写成全链
无裁剪。裁剪为任一canonical轴raw→clip差>1e-6。其余物理安全原因均0。
WAM controls为[103,232,179,300,300]，几何[106,156,300,141,124]；300为上限删失，
安全中止保持原gate，不改阈值。WAM一次实测持握交接，几何0次；几何所有控制都在approach。

指标沿原≥5cm、真实≥2指0.1N、耦合、离桌连续elapsed1s窄witness。成功即停，最长1s
为运行下界；**不是完整RELEASE、精确8cm目标或S2c阶段分数**。成功WAM例最终物体目标误差
.100793m，因此不能将其报成精确物体目标完成。历史原R WAM2/5及S2d0/5不合并、不与本批
相减；本批WAM与几何同为v7接口，只有τ来源分支不同。

同已见`s1_cylindersmall_pass_1`初态，native/model seed195/196、1env/30Hz，顺序WG/GW/WG/GW/WG；
两臂同checkpoint、shape、task、PD、物理/模型gate与每例最多300controls。9可见初态字段
对旧source最大误差0；不能据此保证隐藏物理接触状态逐位相同。5次重置不是5独立任务/seed。

## 执行与验证

- experiment_id：`P-20261022-s2e-geometry-tau`；work_version：`s2e-geometry-v1`。
- run_id：`s2e-geometry-20261022-r1`；实际执行提交`4ce9b0f35a461db37e5c120aa76855526c8c13ef`。
- detached worktree：`tmp/s2e-geometry-20261022`；只新增自己的实现、单测、报告/卡片，未改共享文件。
- 登记上限3000controls/1800s/2GiB；实耗**1941controls /207.844s /20MiB /0updates**，10/10完成。
- GPU5模型约787MiB、GPU6物理约4081MiB/利用率约18–19%；单环境CPU/socket亦有开销，
  4保留空闲，0–3/7未使用。native正常退出0，GPU5/6均释放至2MiB。
- client实测graspenv Python3.8.20/Torch2.4.1+cu121/NumPy1.24.4；native固定
  dexplore_v120_train Python3.8.10/Torch2.0.1+cu118。完整运行manifest与冻结源码保存环境/协议。
- 几何及共享v7语义单测**13 passed**；从存档状态独立重算10/10物理指标一致，安全/裁剪/
  episode/tick/packedτ/actualPD合同通过。actualPD最大误差2.89e-7；独立齐次矩阵核对
  几何绝对目标误差8.86e-8m、差分τ误差2.39e-7m；几何WAM调用0。
- `PASS`仅指合同与重算一致，不是效果优势PASS。仅此一批，无失败后重选参数或追加重复。

## D：结论与下一步决策

当前WAM略好，差1/5，仍属本probe预定义未分出明确优势的区间。两链均未获稳定执行资格，
几何还存在时钟相位与固定R条件分布适配的限制；本结果不能唯一归因为“未来预测知识”。
另一方面，几何0/5也不支持“直接几何已达到可靠抓取、无需WAM”。近期不扩大这套跟踪比较
或训练v6；若继续投入，先修共同执行资格，再用匹配预算检验RL样本效率/泛化。后者是
尚未完成的研究问题，不是本批实验结论。

## Diff、复现命令、期望数值

在隔离worktree根执行；依赖原有只读资产及S2b参考。输出必须fresh。

```bash
git diff 5e187dc0 HEAD -- src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/geometry_tau.py \
 src/task/wam-rl-policy-v3/tools/run/run_geometry_tau_comparison.py \
 src/task/wam-rl-policy-v3/tools/audit/summarize_s2e_geometry.py \
 src/task/wam-rl-policy-v3/tests/test_geometry_tau.py \
 src/task/wam-rl-policy-v3/docs/audit/s2e-geometry-20261022.md \
 src/task/wam-rl-policy-v3/docs/experiments/probes/P-20261022-s2e-geometry-tau.md

PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
 /home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q -p no:cacheprovider \
 --basetemp=tmp/s2e-repro-tests src/task/wam-rl-policy-v3/tests/test_geometry_tau.py \
 src/task/wam-rl-policy-v3/tests/test_grasp_intent.py
# 期望13 passed，物体SE3等变、无live手输入、完整未来guard、v7传递合同通过。

# 检查GPU5/6空闲后，fresh输出；已有私有data软链接保持只读。
REF2DEX_ASSET_ROOT=/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent \
 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
 /home2/wyy/miniconda3/envs/graspenv/bin/python -u \
 src/task/wam-rl-policy-v3/tools/run/run_geometry_tau_comparison.py \
 --output outputs/wam-rl-policy-v3/s2e-geometry-repro --native-gpu 6 --model-gpu 5
# 期望COMPLETED/10回合/0updates/≤3000controls。物理结果如表，不保证跨run逐位确定。

PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
 /home2/wyy/miniconda3/envs/graspenv/bin/python \
 src/task/wam-rl-policy-v3/tools/audit/summarize_s2e_geometry.py \
 --run outputs/wam-rl-policy-v3/s2e-geometry-20261022-r1 --output tmp/s2e-rescore.json
# 本存档期望PASS/1941controls/WAM1/5/geometry0/5、全分布如表、初态误差0。
```

[独立重算与逐回合JSON](s2e-geometry-20261022.json)；
[实验卡/运行前协议](../experiments/probes/P-20261022-s2e-geometry-tau.md)。
原始NPZ、forecast24、raw/clip动作、actualPD、consumer/stages、manifest、冻结runtime与退出回执
在共享树根相对路径`tmp/s2e-geometry-20261022/outputs/wam-rl-policy-v3/s2e-geometry-20261022-r1/`。
日志在同worktree的`outputs/wam-rl-policy-v3/s2e-preflight/r1.log`，无checkpoint新增/覆盖。
