# Im2Flow2Act：现有 EgoDex 的源码兼容性审计

日期：2026-10-13。本地检出：`third_party/im2flow2act-egodex-audit-20261013`，
提交 `f44b368b76195c30bf38074f6419fc8b9f8feb5f`。
这里只核对源码和少量真实数据接口；没有安装包、下载权重、训练或执行机器人。

## 结论

**下游 flow-conditioned robot policy 不能直接用我们的 EgoDex 训练。**
作者模型输出7维机器人末端/夹爪动作，loader 必须读取已执行 action 与 proprioception。
本地真实 EgoDex clip 是双手关键点和场景轨迹，缺少这两个机器人语义字段。
修改 action_dim 也不会自动建立人体与机器人动作的映射。

其上游 human-video flow generation 不要求机器人动作，格式转换后有继续核查价值；
但它是语言条件的轨迹生成器，和用户直接给定目标点的执行 Actor 是不同阶段。
此处只证明部分几何/轨迹 helper 可复用，没有证明上游训练或完整下游前向已跑通。

## 关键源码链路

- [训练配置](../../third_party/im2flow2act-egodex-audit-20261013/config/diffusion_policy/train_flow_conditioned_diffusion_policy.yaml)：`action_dim:7`；proprioception projection input_dim7；pred_horizon16/obs_horizon1/action_horizon8，默认128点/32个plan槽；相机pickle路径固定；object_sampling、right像素区域过滤默认开启。
- [replay_buffer.py](../../third_party/im2flow2act-egodex-audit-20261013/im2flow2act/diffusion_policy/dataloader/replay_buffer.py) 151–233行：逐episode首先读取 `action`、`proprioception`，另外读取初始depth、SAM tracks/moving mask，缺robot mask时回退为全True。存储是Zarr，不是EgoDex HDF5或我们的compact NPZ。`mode='a'`可能写入输入目录；本轮未对原始/远端数据调用loader。
- [diffusion_flow_bc_dataset.py](../../third_party/im2flow2act-egodex-audit-20261013/im2flow2act/diffusion_policy/dataloader/diffusion_flow_bc_dataset.py) 183–207行只加载一组K/camera pose，392–583行生成current/initial flow和plan，利用初始深度与固定camera pose构造pointcloud。
- [训练调用](../../third_party/im2flow2act-egodex-audit-20261013/scripts/controll/train_flow_conditioned_diffusion_policy.py) 126–210行：将真实机器人action加噪，预测噪声并做MSE；不接收human hand的逐关节/逐时刻valid loss。文件顶层还直接初始化NCCL，需要正确distributed launcher；本轮没有导入或启动该入口。
- [mujocoEnv.py](../../third_party/im2flow2act-egodex-audit-20261013/im2flow2act/simulation_env/environment/mujoco/mujocoEnv.py) 470–491/530–548行：动作和proprioception是末端position、Euler rotation、夹爪值；不是人体手XYZ或我们的ACTIVE12。

## EgoDex 需要的适配

1. **目标身份与点采样。**作者 `process_point_tracking_data` 231–345行依赖episode moving mask和robot mask，并使用固定图像位置过滤。EgoDex的图像运动包含头部相机运动，moving points不自动等于目标物体；原过滤区域也不能当作人体手mask。应使用已声明的当前目标mask，保留同点身份。
2. **2D和3D不能交换含义。**模型flow `[B,T,N,3]` 的最后一维是 `(u,v,visibility)`，不是米制 `(x,y,z)`。我们的point_target不能只改维度名后直接传入，必须投影/归一化，并明确固定参考视角还是移动相机视角。
3. **方形图像假设。**`normalize_points` 214–224行使用x/H、y/W；默认256×256时没有区别，640×360原尺寸直接使用会错。模型 [flow_diffusion_policy.py](../../third_party/im2flow2act-egodex-audit-20261013/im2flow2act/diffusion_policy/flow_diffusion_policy.py) 145–150行同样有方形假设。需要同步resize图像、K和UV，或改归一化。
4. **掩码不是自动正确。**默认 [FlowPos2dConcat](../../third_party/im2flow2act-egodex-audit-20261013/im2flow2act/common/flow_network.py) 83–128行在pre/post_zero_out均False时读取visibility但不使用它。缺失点的零坐标仍会产生左上角位置编码；不能把我们masked future置零后假定已忽略。遮挡需明确适配编码、attention和loss。
5. **当前相机在移动。**公开loader只有单个pose_matrix；EgoDex必须用逐帧camera transforms稳定到同一参考系。仅把当前点和future点投影到虚拟首帧相机可以构造几何目标，但当前RGB仍是实际移动视角，尚需处理二者对应，不能称完整策略适配已完成。
6. **上游生成数据也不是即插即用。**[AnimateFlowDataset](../../third_party/im2flow2act-egodex-audit-20261013/im2flow2act/flow_generation/dataloader/animateflow_dataset.py) 135–220行读取 `bbox_point_tracking_sequence[N²,T,3]`、`rgb_arr`、`task_description`，将点排列为grid_size32的网格。我们的voxel筛选后不规则场景点不具有这个网格布局。源码会重复点补齐不足数量，这只填shape，不能制造独立物体观测。需要从当前bbox/mask重新生成query并保持对应，再转换Zarr和时钟。

## 实际执行

使用graspenv Python3.8.20，torch2.4.1/torchvision0.19.1。所有计算在CPU上：
小数组投影、确定性位置编码和import，不涉及模型批量推理或训练。

| 检查 | 结果与范围 |
|---|---|
| 原版common.pointcloud/utility.arr导入 | 成功 |
| 原版机器人policy/flow-generation dataset导入 | 缺zarr；另已单独确认当前环境缺imagecodecs |
| 原版采点/归一化helper接真实r03已标目标 | AST隔离执行未修改的方法；24点组成4个waypoint，经方形投影后得到 `[4,32,3]`，finite；仍仅24个独立点 |
| 无目标点mask | 原helper因对空数组抽样抛ValueError；需要显式跳过或null-goal路径 |
| 640×360中心UV归一化 | 原helper得到 `[0.888889,0.28125]`，应为 `[0.5,0.5]` |
| 原版get_pointcloud反投影 | 使用已有伪标签z组稀疏depth并整数取像素，24点median0.565/max0.873mm量化误差；不验证depth模型质量 |
| 原版FlowPos2dConcat前向 | `[1,4,32,3]` flow和工程零descriptor，输出 `[1,4,32,32]` finite；只改visibility由1到0，最大输出差0；无效UV再置0时最大差1.9975 |

后一个检查证明了默认位置编码的mask语义，不是一个完整robot policy forward；
零descriptor只用于算子检查，没有伪造机器人qpos、gripper或action。

另对先前导出的22窗检查遗漏时变外参产生的几何差：每窗有效末端场景点的
误差中位数平均12.604mm；最大窗口r07 median74.474mm/p9598.459mm。
这比较同一重建点在不同坐标解释下的位置，不能作为tracker/depth准确率。

结果：`outputs/goal-point-wam/egodex-upstream-source-audit-20261013-r2/summary.json`。
此前r1保留；r2增加visibility位置编码检查，未改变原始EgoDex数据。

```bash
/home2/wyy/miniconda3/envs/graspenv/bin/python \
  src/task/goal-point-wam/tools/audit/audit_egodex_upstream_contracts.py \
  --im2flow-root third_party/im2flow2act-egodex-audit-20261013 \
  --egodex-audit outputs/goal-point-wam/video-object-goal-audit-20261013-r1 \
  --output outputs/goal-point-wam/<fresh-engineering-audit-run>
```

输出目录必须是新目录。helper执行使用AST提取未修改原函数；完整dataset正常导入
失败记录始终保留，不能用helper通过来宣布loader已通过。

## 当前采用判断

不优先迁移整个Im2Flow2Act策略：它需要2D/固定相机/机器人action等多项合同变化。
可借鉴目标内query和当前flow/desired plan分离，但在我们现有3D、人手mask监督
模型中使用更合适。若单独做语言条件flow generation，需要另外的数据转换和训练
协议；这不能直接回答非语言goal Actor在EgoDex上的训练收益。
