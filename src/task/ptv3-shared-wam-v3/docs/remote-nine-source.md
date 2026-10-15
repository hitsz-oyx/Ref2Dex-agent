# 九源数据的远端准备

本地与远端使用同一份已清洗窗口、九源采样配方和 TRAIN-only horizon mean/std。
2026-10-15 用户明确要求先暂停本地 v3，后续考虑远端四卡训练；当前迁移只交付数据和
固定代码、运行 CPU 读取审计。UniDexGrasp++ 的大位移尾部仍待质量检查，**不在迁移脚本中
启动远端训练**。四卡入口、配方准入与资源安排由主任务单独处理，不绕过现有 launcher 的卡数限制。

## 目录与交付范围

远端独立 scope：

```text
/mnt/ugreen_nas/storage/oyx_storage/ptv3-shared-wam-v3-nine-source-20261015-r1/
  staging/                     # 可断点传输的4块归档、portable.json、固定代码包
  code/                        # 已提交版本92f3127c的训练依赖闭包
  training-code/               # NAS交付Git快照，初始9356a834；不在此启动DataLoader
  data/native/                 # 新六源的完整运行依赖与原始manifest副本
  data/recipe/                 # 九源recipe/common horizon stats及原数值副本
  data/import-receipt.json      # 原始recipe与位置变更的明确记录
  data/portable-reader-audit.json
  remote-launch.json
  remote-progress.json
  remote-supervisor.log
  training-code-deployment.json
```

原三源远端已存在，保留原目录并只读复用：

```text
/mnt/ugreen_nas/storage/oyx_storage/goal-point-wam-remote-20261012-r1/repo/outputs/cm-pointflow-effect-pretrain/pointworld-main3-sharedstats-20261009/data-remote
```

其原 source manifest 与索引身份与本地 main3 一致；导入工具复核既有身份，不改写这些
manifest。原三源 TRAIN 窗口为 OakInk2 3,860,393、GRAB 207,179、ARCTIC 310,906。

`code/` 是约 1 MB 的固定代码归档，供本次 CPU 迁移审计使用；NAS 的
`training-code/` 是另一份真实 Git 交付快照。后者 detached 在 `9356a834`，通过约 577 KB 增量 Git bundle 部署，
只读借用旧 v2 repo 的 Git 对象和 PointWorld vendor；原 repo 保持不变。它已经通过
`checkpoint.git_commit()` 与九源数据模块的 CPU 导入检查，不需要伪造 Git 身份或放宽
核心 checkpoint 合同。旧 Git 对象库目前是明确依赖，应保留；如未来要删除旧 repo，
须先为新 clone 补齐其借用对象。

正式训练代码使用短路径 `/home/wbcd/workspace/oyx_ws/wam-v3-20261015/repo`，数据仍位于
NAS。legacy IPC helper 必须在代码 ROOT 下建立 AF_UNIX socket，NAS 根路径过长，最后
socket 路径会超过 Linux 108 字节限制，单改 `TMPDIR` 无法解决。因此 NAS Git 快照不作为
多 worker DataLoader 的执行目录。后续 viewer/四卡入口修订继续以增量 bundle 部署短路径
checkout，再 detached 到已提交代码。

远端 SSH 会话的默认 soft nofile=1024、hard=1048576。正式多卡 launcher 应只对自身进程
提升 soft limit，不修改系统或其他进程；四卡入口和运行前检查由主任务另行验收。

新增部分包含 37,591 条 native payload 和 2,638 个 CAD，共 40,229 个 NPZ、2,457,338,191 字节。
归档加入清洗manifest、原始manifest副本、冻结窗口索引、统计与配方后为 2,689,935,360 字节。
这里迁移的是训练所需的 pose/mesh 产物，不包含原始视频或全部原始下载包。

源 ID 4/5/6/8/9/10 分别为 UniDexGrasp++、TACO、HOT3D、DexYCB、H2O、HOI4D，
native TRAIN 窗口分别 264,439 / 30,568 / 71,070 / 351 / 5,691 / 55,547。
本包保留 HOcap 的工程数据，但 verified TRAIN/DEV 均不使用其未确认时钟的窗口；
H2O official TEST、当前输入体素域拒绝的窗口不进入配方，也不重新计算窗口来绕过这些限制。
旧三源 TEST 索引仅用于核对既有来源身份，CPU 读取面板不读取 TEST 标签。

## 位置绑定与不变量

- NPZ/CAD 去重后使用 `native/payloads/` 相对路径，实际读取不再依赖本地绝对目录。
- `native/origin-manifest.json` 保存原始路径与身份；`source_manifest`、`source_path` 等原始
  provenance 注记继续表示来源，不声明为远端运行依赖。
- 显式重绑九源 `recipe.main3_root`、`recipe.native_root`、内嵌 native engineering recipe 的
  `data_manifest`、共同 stats 的 `recipe` 以及 native `source-moments.input_manifest`。
- 窗口索引字节、大小、纳秒 mtime 保留；每个 payload 的字节、大小和纳秒 mtime 保留。
  NAS 已单独实测支持精确纳秒时间；导入会恢复归档解析可能舍入的时间并复核。
- 共同手/场景 horizon mean/std 不重算，保存导入前 `origin-horizon-numeric.json`，CPU 审计
  逐数组核对。原始/clean engineering 产物的 `training_allowed=false` 保留。
- 用户此前的训练准入只记录在独立九源 recipe 中；产物迁移完成不等于远端训练已启动，
  也不替代本轮质量审查。

如后续只调整窗口筛选、统计或配方，可继续复用现有远端 NPZ 闭包，仅部署新的索引/manifest/
统计视图并保留旧视图；不必再次传输所有轨迹或重新下载原数据。

## 监督与恢复

本地运行回执：`tmp/nine-remote-20261015-r1/launch.json`，监督 PID 1601283；
`progress.json` 和 `supervisor.log` 记录传输、远端导入、CPU 验收各阶段。
每块 `native.part-*.rsync.log` 保存可恢复的传输进度。

四条 SSH 连接实测合计约 1 MB/s，现有隧道总带宽是主要限制；已把代码包由整仓库
249 MB 缩为约 1 MB 的已提交代码闭包。NAS 空间约 48.5 TB；本地和远端均使用 20 GiB
空闲空间底线。仅访问已有账户、密钥与严格主机键记录。

本地监督器会自动执行“断点传输 → 启动独立远端监督器 → 组包 → 新目录导入 → CPU 九源读取
验收”。转入远端阶段后，即使本地 SSH 断开，远端监督器仍独立运行。监督器不启动 GPU 工作。

需要恢复传输时，在项目根目录复用 `launch.json` 中完整 `command`，避免修改 SSH 身份或
重建源数据。`rsync --partial --append-verify` 保留并校验已有分块。不要并行启动第二个
相同 run 的监督器；先核对 launch PID 和远端 `remote-launch.json`。

若远端导入失败，已传归档与部分导入保留；脚本拒绝静默覆盖部分导入目录。先检查日志，
修复原因，再声明一个新的导入目录或 scope；不要删除用户已有数据或 checkpoint。

独立审计可从远端 `code/` 执行：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  /home/wbcd/miniconda3/envs/graspenv/bin/python \
  src/task/ptv3-shared-wam-v3/tools/audit/audit_portable_nine.py \
  --import-root /mnt/ugreen_nas/storage/oyx_storage/ptv3-shared-wam-v3-nine-source-20261015-r1/data
```

远端提前 CPU 预检已通过：Torch 2.0.1+cu118，v3 九源数据模块及其依赖成功导入，
旧三源实际样本通过统一 collate，source ID 0/1/2 正确。新六源最终验收以 scope 内
`data/portable-reader-audit.json` 的 `CPU_READER_PASS` 与 `remote-progress.json` 为准。

本地 `test_portable_import.py` 的合成 roundtrip 已通过，检查位置绑定、原 mean/std、payload
纳秒时间、原始数据 flag 和原 manifest 字节保持不变。最终远端 CPU 审计检查全部 native
文件依赖、TRAIN/DEV 实际读取与混合 collate；不代表远端 CUDA、四卡吞吐或训练收敛已验证。
