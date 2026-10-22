# S2d：原逐帧消费的 WAM 与 live-object GT 对照（2026-10-22）

**结论：本批 a=0/5、c=0/5；物体重锚定没有救回原 R，c 尚未获得物理上界资格。**
a 也未复现旧 WAM 的2/5，差异超过“小抖动”。不能用这批结果断言“参考质量才是瓶颈”；
参考时钟相位、R对新τ条件的适配、数值/接触敏感性仍未分离。下一步应优先交付此失败见证
给消费器路线，不据此追加WAM训练、重复筛选成功批次或放宽安全gate；三臂RL仍HOLD。

执行前登记为Decision：区分“物体空间错位足以解释旧GT差”与“该修正仍不足”；最便宜为
两链各5次、零拟合。预定义c≥a+2/5才算描述性改善，本次未满足。
experiment_id=`P-20261022-s2d-object-anchored-controls`；run_id=`s2d-controls-20261022-r4`；
实际运行提交`402b42f8087f73e737bb098dcca482fba6bd8ab7`；detached worktree，无新分支。

| 原R链（block0→4；各n=5） | 最长持握elapsed秒 | ever/final | clip/controls | 关节安全中止 | 首高度失败tick |
|---|---|---|---|---|---|
| a：WAM11000逐帧τ | [.333333,0,0,0,0] | 0/5；0/5 | 1/992 | 3/5 | [83,null,null,null,null] |
| c：live-object GT逐帧τ | [0,0,0,0,0] | 0/5；0/5 | 0/600 | 5/5 | [null,null,null,null,null] |

高度失败定义为**首次达到5cm后再次低于5cm**；其余9例从未达到5cm，故null不代表保持
成功。a block0在68达到5cm，held为72..82，83首先高度失败；所有c在112/119/144/113/112
触发原`physical_joint_limit`，无held区间。裁剪定义为任一canonical轴raw→clip差>1e-6。
历史source-PD5/5、原R+clockGT0/5、原R+WAM2/5均保持原记录，本次没有重跑source或合并分母。

**公平性与反例：** 两链共用原Scheduled686 r4 step50000 R、相同executor参数、30Hz native11
的8步world future-minus-current语义、H4启动/encoder gate与原PD/安全阈值。c只换τ生成：
`G[k] = T_obj_live @ inv(T_obj_ref[t]) @ H_ref[t+k]`，`τ[k] = G[k] − H_live`。
目标G仅取参考和当前物体位姿；当前手只用于τ差分编码、R状态和可用gate。所有未来共享
**参考当前物体锚点**，保留参考未来抬升，不能逐未来帧用各自物体逆变换消掉抬升。
使用S2b真实205状态，仍按同30Hz时钟t（不做手位置相位匹配），完整未来可用t=3..196；
缺未来则零τ/invalid，不复制终帧。c的600实际controls中585有效，全部在参考过期前中止。

反例检查：当前手平移[.17,−.11,.09]m后，τ+H重建G最大误差5.66e-9m；物体做旋转+
[.4,−.2,.3]m平移时目标等变误差2.22e-16m；与参考对齐时恢复真实第1..8未来误差
2.22e-16m。“目标=当前手”的假实现偏差.284150m，被拒绝。独立齐次矩阵重算所有c的τ
最大误差1.20e-7m。**消除空间开环不等于已消除时钟相位/条件分布失配**；完整参考也比旧
81状态延长，因此即使c改善，也只能先归于“完整参考+重锚定”的组合，不能唯一归因。

**与旧基线差异：** 9个可见初态字段逐数组误差0，checkpoint、flow_eval_seed、FP32合同
未换；首回合对旧WAM的tick0..2观测/动作完全相同，tick3首次τ差8.94e-8m、动作差
5.96e-8；超过1e-6的差分依次见于观测4、τ6、action8、q9、hand22、object55、force58。
这支持早期微小数值差随后被闭环/接触放大，不能认证唯一根因或GPU隐藏warm-start相同。
旧2/5、另一批3/5、本批0/5说明该单轨迹短屏对运行数值条件敏感，**未达成稳定基线复现**。

**执行边界与验证：** 同`s1_cylindersmall_pass_1`、native/model seed195/196、env1；5个block
是同初态重复而非5个独立任务。顺序ac/ca/ac/ca/ac，每例成功/原安全abort/300controls即停；
窄成功仍为≥5cm、真实≥2指0.1N、耦合、离桌elapsed1s，不替换完整RELEASE。
预算每链≤1500、总≤3000controls/1800s/2GiB；实耗**1592controls/164.851s/11MiB/0updates**。
模型GPU5、native GPU6，4空闲；0–3/7未使用。模型graspenv Python3.8.20/Torch2.4.1+cu121，
native固定dexplore_v120_train/Torch2.0.1+cu118；GPU采样5约769–787MiB、6约3575–4081MiB。
正常退出0、GPU5/6均释放至2MiB。独立评分10/10一致，raw/clip/expanded、episode/tick、
packedτ合同通过，actualPD重算最大误差2.61e-7。JSON的PASS仅指记录/重算通过，不是效果PASS。

当前共享提交的测量入口本来就以旧scheduler-v2兼容快照保持a链身份，本次沿用其`freeze`
方式（基点1f41bcdb+原source.diff），没有把生产scheduler回滚或改executor文件。隔离worktree
额外只读引用现有vendor/config/data；所有新产物留在worktree。r1缺PointWorld、r2缺native
train YAML、r3缺env YAML，均setup失败、0controls；日志及退出回执保留，不算物理失败回合。

**复现与期望数值（在隔离worktree根）：**

```bash
# 输入目录不存在时，在worktree内链接已存在的只读数据；正常主树无需再建。
mkdir -p data/processed_data
# 若不存在：ln -s /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/data/processed_data/inspire_rl_object_dexplore data/processed_data/inspire_rl_object_dexplore
nvidia-smi  # 仅5/6空闲时运行；fresh输出目录
REF2DEX_ASSET_ROOT=/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent \
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/home2/wyy/miniconda3/envs/graspenv/bin/python -u \
 src/task/wam-rl-policy-v3/tools/audit/measure_s2d_controls.py \
 --output outputs/wam-rl-policy-v3/s2d-controls-repro --native-gpu 6 --model-gpu 5
# 工程期望COMPLETED/runs10/updates0；物理分布如表，但不保证跨run逐位或成功数相同。
PYTHONDONTWRITEBYTECODE=1 /home2/wyy/miniconda3/envs/graspenv/bin/python \
 src/task/wam-rl-policy-v3/tools/audit/summarize_s2d_controls.py \
 --run outputs/wam-rl-policy-v3/s2d-controls-20261022-r4 \
 --output tmp/s2d-rescore.json
# 对本存档期望PASS，1592controls，a/c=0/0，持握与表相同，反例检查通过。
git diff 2851060a HEAD -- src/task/wam-rl-policy-v3/tools/audit/measure_s2d_controls.py src/task/wam-rl-policy-v3/tools/audit/summarize_s2d_controls.py
```

[原始逐回合记录与独立重算JSON](s2d-controls-20261022.json) 包含全分布、失败分项和反例数值；
原始NPZ/raw动作/actualPD/τ、manifest、runtime源码及native回执在
`tmp/s2d-controls-20261022/outputs/wam-rl-policy-v3/s2d-controls-20261022-r4/`（相对共享树根）。

`git diff --check`与两脚本编译通过；仓库`tools/verify.py --changed`失败：隔离worktree缺少
历史本地产物、并有旧绝对链接越出worktree（241项，S2d文件0项）。未修改无关文档；本页JSON链接存在。
