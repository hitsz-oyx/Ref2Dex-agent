# 远程数据入口与限定库存核对（2026-10-12）

## 结论与范围

用户恢复 Windows 反向 SSH 隧道后，使用已有身份及严格的既有主机键校验，已成功只读登录
`deepcybo-4-4090-48`，账户为 `wbcd`。NAS 正常挂载；目前无需用户重新授权密钥或修改服务器配置。
Windows 隧道窗口和网络需保持在线。此前连接拒绝是本机 `22023` 未监听，恢复后 SSH 退出码为 0。

本次只检查 `/mnt/ugreen_nas/storage/oyx_storage` 的顶层及下一层目录名称，每个目录最多
150 个子项（实际均未触及上限），以及已知 PointWAM/EgoDex 扩展的小 manifest 和一个 NPZ
的数组头。没有递归扫描整个 NAS，没有下载、写入远端文件、启动训练或占用远端 GPU。
该范围内未找到名字含 EPIC、touch、T-Rex/trex、EgoTac、Aether、ObjectForesight 的候选包。
**这不表示 NAS 其他路径或更深层绝对没有这些数据。**

脱敏记录：

- [连接检查及恢复记录](../../../../../tmp/remote-data-access-20261012/connectivity.json)
- [限定库存和样例数组头](../../../../../tmp/remote-data-access-20261012/inventory.json)
- [本地只读检查脚本](../../../../../tmp/remote-data-access-20261012/read_remote_inventory.py)

## 可见资源与数据

NAS 挂载点为 `/mnt/ugreen_nas`；检查时报告总量 71 T、已用 22 T、可用 50 T。
`nvidia-smi` 报告 4 张 RTX 4090，每张 49140 MiB、检查时显存占用 0 MiB。
这些是容量和可见性观察，不意味着已迁移训练、获得额外训练身份或改变全局资源预算。

目标存储中已有 `pointwam-data-expansion-20261011-r1` 与
`egodex-full-part2-20261011-r1`。远端代码入口
`/home/wbcd/workspace/oyx_ws/Ref2Dex-agent` 存在，其下没有 `outputs/`、`data/` 或
`datasets/` 目录。现有历史存储还包括 ARCTIC、GRAB、ContactPose 的命名目录；本次未重新审计其合同。

现有扩展的资格记录为：

```text
/mnt/ugreen_nas/storage/oyx_storage/pointwam-data-expansion-20261011-r1/
  manifests/pointwam-expanded-qualification.json
```

它报告 `COMPLETED`、26 个原始 episode（其中 1 个准备失败）、48 个 qualified clip，
23 fit/25 dev clip、12 fit/13 dev record；工程门槛通过，结果为 `PROMISING`。
**记录同时明确 `training_allowed=false`，不能直接把这批工程产物混入当前训练。**
本次只读取记录，并未独立重跑资格审计或验证其科学有效性。

`egodex-full-part2-20261011-r1/part2.zip.part` 文件为 327,274,733,628 字节；
它仍使用 `.part` 后缀，本次没有验证完整性，不将其视为已经完成且可解包的完整归档。
未干预相关下载进程。

## 已知 clip 合同与接入限制

样例文件：

```text
/mnt/ugreen_nas/storage/oyx_storage/pointwam-data-expansion-20261011-r1/
  processed-groups/group00/
  episode00-assemble_disassemble_furniture_bench_chair-0/
  clip00-f0000/pointwam_clip.npz
```

只读数组头所见文件大小约 4.89 MB：

| 字段 | 形状 | 含义边界 |
| --- | --- | --- |
| `scene_xyz_m` / `scene_valid` | `[31,5848,3]` / `[31,5848]` | 逐点场景轨迹及有效性 |
| `scene_rgb` | `[5848,3]` | 当前点颜色 |
| `hand_xyz_m` / `hand_valid` | `[31,2,10,3]` / `[31,2,10]` | 两手各 10 个语义点 |
| `scene_displacement_m` | `[30,5848,3]` | 场景轨迹监督 |
| `hand_displacement_m` | `[30,2,10,3]` | 手轨迹监督 |
| `source_frame_ids` | `[11]` | 原采样帧索引；不能把 31 个输出帧直接当作 31 次独立原生测量 |
| `timestamps_s` | `[31]` | 输出时间轴，接入前须核对采样和插值定义 |

远端该 episode 的 manifest 明确 `split=official_train`，
`engineering_only=true`、`training_allowed=false`、`strictly_causal_observation=false`，
观察来源为 `offline full-clip teacher reconstruction`。
以实际远端 manifest 为准，不能根据本地工具旧默认值推断它来自 official test。

这批包不是直接满足当前 Goal-Point-WAM 合同的新增源：当前手合同为每手 11 点，并要求
H4 历史和 K24 未来的统一时钟；样例为每手 10 点且存在 31 输出帧/11 原采样索引的区别。
数组清单也没有当前路线所需的目标对象身份、配对对象点及刚体目标 R/t。
此外，整段离线教师重建不自动满足当前观测的因果性要求。

最小接入审计应先在单个原始 episode 上核对：手点语义映射、原生时间与插值、是否可构成
H4/K24 窗口、仅用已观测帧能否生成当前场景输入，以及目标对象/goal 是否有可审计的标签。
这些核对完成前保留工程隔离和 `training_allowed=false`，不为扩大规模而直接混入。
现有 EPIC 本地稀疏手标注不足的问题，本次没有找到更完整的远端来源可以直接解决。

Touch/EPIC 的外部候选及用途见
[数据适配调研](2026-10-12-touch-epic-data-fit.md)。如用户知道其他实际数据目录，
可在那个明确路径继续做小 manifest 审计；当前远程连接本身已恢复，不再需要用户协助排除网络故障。
