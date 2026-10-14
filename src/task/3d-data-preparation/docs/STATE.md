# 原生3D数据接入状态

## 2026-10-14 全量扩产 Decision Note

- 用户明确取消产物磁盘空间上限并要求立即推进；根CAMPAIGN据此更新，六张项目GPU上限与外部只读/身份边界不变。
- 行动：HOT3D全部ground_truth/hand_data/assets约2.70GB；TACO三份几何ZIP约15.69GB，CPU下载与转换并行；复用官方资产启动UniDex新增mesh采集，随后HO-Cap轻量接入。
- 输入：用户三个授权清单只读；TACO/HO-Cap官方公开几何包；UniDex官方TRAIN中排除TEST mesh，保留train/dev按mesh分组。源数据不因未来结果筛选。
- 成本与停止条件：HOT3D下载4线程、TACO手部ZIP8线程独立Range、24小时可恢复期限，单次转换内存有界；UniDex实际每批25对象/200env/两轮（末批更少；原collector最多256env）、GPU4单worker、每批≤10分钟；扩产输出只受实际文件系统剩余≥20GiB约束，不生成视频。既有训练与未知进程不动。
- 结果仍属工程语料，不自动准入训练；缺权限或临时URL失效及时向用户说明。源clock/语义问题保留明确mask与provenance，不伪造标签。

## 扩产实际状态（初始样例之后）

- HOT3D849包已全部完成：2,701,783,954B压缩、9,069,098,557B展开；Aria136+Quest158=294个记录提供手物GT，130个不提供。290段转换为999,865个名义30Hz时间步，4段没有QA通过的原生时段跳过，0实现失败。
- 全序列版保留native/grid timestamp和最近邻残差，容忍≤16ms；重复选中的native帧标invalid。它不是插值后的额外真实观测，也不是正式时间精度准入。原512步pilot仍保留其≤2ms规则。
- UniDex排除已有20mesh和官方TEST后，2046个新mesh分82批，每mesh16条真实执行，正在GPU4持续采集与自动审计。原20保留，总目标2066mesh/33056条；实际完成数看progress，不写成已经全量完成。
- HO-Cap旧Task已有9人64段72944帧，经原作者MANO与标签检查；只读复用成64条DEV工程序列，继续保留外部测试用途。DexYCB已有subject10，100段7282帧直接读取一个固定相机的joint_3d/pose_y，不处理图像、不重复8个视角；遵循官方s1把subject10全部放TRAIN。两者native clock仍明确未核实。
- TACO物体位姿和模型包完成，15.23GB手包并行Range续传；完成后等待中的converter自动按序列读取ZIP，避免再全量解包26.99GB pickle。真实pilot重打包验证全量reader，全部原生数组与pilot一致（1e-7容差）。
- `native3d-live-corpus-20261014-r1`每分钟自动追加完成分片、共享CAD引用，不把已有pilot和full重复纳入。5项合同测试通过，混合来源shared-collate已通过工程检查；没有开始多源模型训练。
- 旧393/400GB全局watcher已核验归属后替换为纯容量/远端训练监控，避免扩数据误停已有训练；相关receipt在项目tmp。
- H2O实际HTTP401 Basic账号要求、HOI4D网盘下载访问受限，已向用户发缺失访问问题；DexTrack官方受限且社区镜像有混合修改lineage，暂不把kinematic references记作执行数据。

实时查看：`python3 src/task/3d-data-preparation/tools/audit/status_geometry.py`。输出与日志在`outputs/3d-data-preparation/`和`tmp/3d-data-bulk-20261014-r1/`。下文初始512MiB/400GB预算是历史快照，已被本节用户授权覆盖。

## 2026-10-14 路线切换 Decision Note

- 决策：按用户授权暂停视频重建，转向已有连续3D手、物体pose/mesh的数据；不改变根MISSION的目标或claim。
- 证据：EgoDex首版每尝试窗约2.63秒且目标语义仍有问题；已有UniDexGrasp++20对象/320条执行的3200窗合同审计通过。
- 行动：核验归属后，用既有SIGUSR1安全边界暂停远端EgoDex下载器及GPU1/3旧重建，保留原始输入和全部证据；不停止已有模型训练。
  在本Task实现统一PoseMeshWindows，先把现有UniDex轨迹以按需方式接入，再取得TACO/HOT3D最小授权几何样例。
- 成本：当前实现/文件统计使用CPU，无新增GPU任务；原生样例下载初始≤512MiB、处理输出≤1GiB、临时输入/缓存≤1GiB。
  全量解包/回放和训练另按容量核验安排，遵守全局400GB及六卡上限，窗口不重复展开存储。
- 停止条件：输入/单位/时间/手与物体坐标不明确先停该reader；没有授权下载清单不代用户申请身份或接受条款；不可读数据/目录/资源边界及时询问。
- 下一步：source/profile明确、多对象当前anchor与缺失标签合同通过后，做真实模型工程接线；正式微调须另核验统计和来源采样。

外部访问：用户已提供HOT3D Aria/Quest/Assets授权清单，首批原生样例下载完成；三个清单保持本机且被gitignore排除。HOI4D分享读取失败、H2O需身份申请，尚不启动。

## 已完成的原生3D工程链路

- 视频下载和两条旧重建已正常STOPPED；保留全部数据，既有模型训练继续。证据：`outputs/3d-data-preparation/route-switch-20261014-r1/remote-inventory-r1.json`。
- `PoseMeshWindows`支持共享CAD、多对象、当前目标物体固定anchor、原生缺失mask、按需H4/K24；成功与失败执行均保留，控制量不混入人手几何标签。
- 现有UniDex20对象/320条执行接入新manifest，无复制原始数据，TRAIN2560/dev640。全部窗口与旧reader逐字段比对，最大差2.98e-8，并进入既有forecasting_collate。
- HOT3D两段授权真数据+完整Assets：下载177,089,578B，解压236,321,352B。各取512个时间步，TRAIN57/dev58窗、最多6对象、左右手；官方QA掩码实际应用。
- Aria选214-1时间流并最近邻对齐30Hz，偏差上限2ms，实测最大1.685ms；Quest最大约1ns。原始/grid时间戳及偏差保留，不插值pose、不把多相机时间戳当额外动作。
- 5项独立合同测试通过，覆盖固定anchor、法线/导数、多对象、未来干预、缺失标签、reset和共享collate。
- TACO一段下载完成：24.03MB成员payload、34.35MB原生输入，221帧双手/两物体生成25个TRAIN窗；新增compact序列和两几何仅99,947B。mixed-r2复用HOT3D后为TRAIN82/dev58，共140个新窗口。TACO30Hz仍是待证实的clock假设，不能正式训练准入。
- 数据继续标记`training_allowed=false`：目标操作语义、人手点profile、跨源normalization/sampler和模型工程接线仍未准入；两段HOT3D的参与者/设备划分仅工程使用。

空间与后续安排见`docs/user/架构.md`。初始下载/缓存边界不自动扩到全量；扩大采集前记录下一条Decision Note并复核全局400GB。
