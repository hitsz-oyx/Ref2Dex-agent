# 原生3D数据接入状态

## HOI4D Windows 取数 Decision Note（2026-10-14）

- 授权/证据：用户明确授权按`docs/user/连接远程windows.md`从`D:\BaiduNetdiskDownload`取数，并已启动反向SSH隧道。已用现有服务器密钥连接oyx@127.0.0.1:2222，四个ZIP大小与官方清单完全一致，共24,189,914,732B。
- 行动：只读SFTP拉取camera/hand/CAD/annotations四包至`hoi4d-native-full-20261014-r1/archives`；Windows源文件保留。2并发、断点partial、严格固定本次SSH host key、完整长度和ZIP目录检查后原子发布；不传用户其他文件。
- 成本/停止条件：CPU/网络、最多24h、20GiB物理余量，STOP文件可停自己的SFTP子进程并保留断点；不会修改Windows系统设置或源数据。
- 初步实测：hand490,584个pickle/1,858个sequence ID，左31,845、右458,739；annotations远端只读目录891,900 objpose、2,973 camera trajectory/action。左右各一帧FK投影与原生kps2D平均误差约0.7px；已实际看到文件名00000与内部frameId1不一致，配对必须另核。
- 下一步：输入先检查hand/objpose/camera轨迹及rigid类别的共同坐标与原生时钟；annotations中的segmentation不自动当pose训练输入。

TACO网络恢复补记：原下载遇到transport断流，保留已完成分片。用真实HTTP故障服务复现了64MiB分片中途断流后重试从头丢失已接收前缀的问题；修复为每次请求不超过8MiB且同一分片重试保留已确认字节。该故障测试先失败后通过，Task共9项测试通过；STOP仅用于核验归属的旧writer安全切换，不丢弃完成分片。

## TACO GPU6 加速 Decision Note（2026-10-14）

- 决策：用户询问增加GPU并行后，启用空闲本地GPU6执行TACO MANO FK；保留GPU4上的UniDex队列，余下约3–4批不为临时并行重排任务。
- 证据：已核实本地5/7和远端0/2是既有训练，UniDex用本地4，共5卡，增加6不超过六卡上限。真实221帧标注及仅用于测速的8倍平铺输入，FK在GPU快3.19/3.55倍；全reader GPU smoke与CPU原生数组差≤7.45e-8m，object/time和mask不变。数据解码和写盘仍在CPU，不能按FK倍数宣称全流程提速。
- 行动与成本：先确认旧TACO converter仍处WAITING_NATIVE_ARCHIVES、UID/cmdline归属正确，再替换为GPU6版本；原生ZIP完成后自动开始，转换前再次检查该卡无其他占用。CPU4线程、单GPU、2h期限、20GiB物理余量，无视频，STOP可在序列边界停止。
- 时间边界：21:13盘点TACO下载约余20分钟；2317序列的全量转换吞吐尚未测，暂估15–40分钟，后续审计约5–15分钟。该预算是工程估算，开始转换后用实际完成速率更新。HOI4D等待用户四包，DexTrack缺官方可读输入，均不计入可承诺完工时间。

## H2O 接入 Decision Note（2026-10-14）

- 决策：用户授权凭证下载后，直接接入pose包已有的原生21点手、物体RT和逐帧cam_to_world；不新增视频估计或MANO拟合。
- 证据：实际subject1包同时含hand_pose（128数，左64/右64）、hand_pose_mano（124数）、obj_pose_rt和cam_pose；CAD米制范围约3–26cm。官方player以cam_to_world乘物体RT，官方pose split按session分开。
- 行动：统一右/左11点，保留原始21点、MANO、相机轨迹和action/verb；仅使用cam4，官方train/val进入工程索引，subject4 test写独立heldout-test-manifest，不进入自动混合索引。原始帧号断点拆段，短于28帧不造窗。
- 成本/停止条件：CPU流式读取每份tar，不完整解包近百万txt；2h期限、20GiB物理余量、STOP文件边界可停。CPU用于纯标签转换，无学习模型；不新占GPU。相机/物体非法旋转或缺失帧不补假标签。
- 限制：论文记载30fps采集，但当前原生包没有逐帧时间戳，clock仍待准入；原生joint profile与MANO重建profile分开。工程转换不改变training_allowed=false。

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
- H2O用户提供的HTTP Basic凭证已实际验证：六包均HTTP200，合计326,238,749B；已启动4线程可恢复下载（1h期限、20GiB物理余量），凭证只存本机mode0600文件，不进入Git或日志。手/物体pose和CAD下载后再核验坐标、时钟、split；下载成功不等于训练准入。HOI4D仍需网盘下载访问，DexTrack官方受限且社区镜像有混合修改lineage，暂不把kinematic references记作执行数据。

H2O新增实测：138个TRAIN/DEV原始take已变为524个连续片段、67,358帧，8个CAD；22个TRAIN帧所在连续段短于28帧而跳过。全量6,871窗口接口审计通过（TRAIN5,691/dev1,180），真实原生21点与官方MANO约定spotcheck最大差8.14e-8m；8项合同测试通过。静态图与冻结审计位于`h2o-interface-snapshot-20261014-r1`，多源冻结接口检查也通过；H2O六包现已全部完成，含subject4 test的总转换为184个take/744个连续段/90,749帧，0失败；test220段/23,391帧独立保留，未进入多源索引或训练。

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
