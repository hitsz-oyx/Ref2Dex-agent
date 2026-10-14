---
schema: ref2dex.probe.v2
probe_id: P-20261014-unidexgrasp-local-admission
experiment_id: P-20261014-unidexgrasp-local-admission
date: 2026-10-14
task: video-data-preparation
branch: main
git_commit: 255322371f7aa5bec6466cc55e67a7e2f90b5680
claim_id: C1
hypothesis_family: HF-unidexgrasp-rigid-admission
probe_index_in_family: 1
seed_pool: probe
seeds: []
decision_changed_if_positive: use native rigid geometry and explicit ShadowHand FK, then assess rollout acquisition
decision_changed_if_negative: resolve asset scale, pose or kinematic semantics before expansion
status: PROMISING
run_id: unidexgrasp-local-subset-20261014-r1
---

# UniDexGrasp / ++ 刚体样例接入检查

Result: PROMISING（静态几何接入）。四个真实抓取完成ShadowHand FK与尺度mesh审计，++ checkpoint可读；现有动态训练准入仍UNCLEAR，未采集rollout。
Decision: 优先用++状态策略做短仿真采集以获得真实连续轨迹；先解决兼容Isaac Gym运行环境和保真输入，不由静态终态插值生成训练数据。

Class: Decision. 服务于Shared-WAM刚体手物监督来源的选择。
区分“原生刚体mesh/抓取标签可便宜解析”与“格式/尺度/手FK仍是主要blocker”，同时确认
发布文件是否提供真实连续手物轨迹。最便宜方法是一个官方示例物体、四个静态抓取，
读取原始数据与尺度约定，生成真实FK/mesh预览；静态标签不能通过复制或插值伪造future24。

## Decision Note / execution budget

- 用户明确要求先试UniDexGrasp和++；沿用main，不建分支，不修改既有训练和外部资产。
- 官方原仓库自带Car样例；使用固定上游提交`36c9bfcf7cedc987dd15e2384c548e51d5aadc21`。
  ++源码另在`third_party/UniDexGrasp2`，仅核验它复用的数据与策略接口。
- 有界获取≤64MiB必要样例/源码，总下载≤10分钟，每次网络请求≤30秒；
  本轮审计输出≤32MiB，静态FK/绘图CPU≤2分钟，无GPU训练/仿真。
  CPU用于少量参数几何与文件审计；如果需要策略推理/仿真，重新检查GPU预算，另记录运行。
- 输出`outputs/video-data-preparation/unidexgrasp-local-subset-20261014-r1/`，
  scratch `tmp/unidexgrasp-source-audit-20261014/`。两个第三方仓库先以blob:none/no-checkout克隆，
  不自动获取大模型或全量数据；当前GitHub API限流，改用固定SHA的raw文件。
- 格式/坐标不明、手与物体尺度严重不符、下载超限即停止扩量；保留失败记录。

## Limitations / future evidence

不以一个物体验证大规模质量/训练收益。已发布静态抓取、目标姿态和模型checkpoint与连续
rollout是不同产物；如无时钟和序列，本轮不生成完整Shared-WAM训练样本。
ShadowHand与当前Inspire的控制action不同，语义手点映射不意味着能混用Chunk-R action标签。
完整轨迹需实际策略仿真、记录hand/object/action/时间，再按30Hz构造H4+future24。

## 实际输入与CPU几何检查

- 使用官方原仓库固定SHA的小样例：一个Car、四个NPZ抓取、五种已缩放mesh/URDF和MJCF，
  必要输入/源码18文件573,016字节（约0.55MiB）；首次raw请求遇到SSL EOF，保留文件后有界重试成功。
  下载manifest及一次性fetch脚本在声明的output/scratch内。没有下载全量tar或修改外部模型。
- 每个抓取文件只有`qpos`（28个命名参数）、`scale`、`plane`；没有time、frame、action序列。
  四个文件是独立抓取而非相邻帧。实际尺度倒数12.5/8.333334/12.5/6.666667，对应
  .08/.12/.08/.15，URDF mesh scale=1；不再二次缩放。物体最长尺寸约15.0/22.5/15.0/28.1cm。
- 原始wrist-free MJCF包含22个finger DOF；保留body/joint/inertia/site原参数，去除仅供视觉/
  actuator用的asset/default引用，用已安装MuJoCo做静态FK（无物理step、无GPU）。
  手root的`sxyz`/`xyz`Euler与translation作用于同一物体坐标系，没有拟合额外对齐。
- 沿用11槽角色，但landmark显式为palm origin、thhub、四个knuckle和五个XML fingertip site；
  这是ShadowHand profile，不宣称等同人类解剖或Inspire link原点。XML tip offset与policy中
  20mm distal offset不同，后续不能悄悄混用。左手全invalid。
- 四个FK均有限，重建指尖到物体mesh的unsigned距离约5.5–12.8mm，只是几何接近关系，
  不能据此证明表面无穿透、真实接触或抓取稳定。三个抓取无关节越界；00003的THJ0为
  +0.001498rad，超原范围上限0，保留原值与告警，不自动clamp再声称原样验证通过。
- CPU audit1.22秒，生成`static-grasps.npz`与`review-static-grasps.jpg`；静态profile不含H4/future24，
  `training_allowed=false`，完整动态窗口数0。初步目视没有明显整体尺度错位，未请求用户批准
  抓取稳定性或将该图升级为数据质量Validation。

## ++策略回放准备与当前运行缺口

按用户提出的policy回放方向，另获取官方++固定SHA
`6c9e710b04769bf5d9765e3233820101cb3d718f`的state checkpoint（17,220,395字节）
及最小loader代码、五份官方`pc_feat_*.npy`，总输入仍小于64MiB。CPU仅反序列化/严格加载
参数及查验feature，不运行神经网络推理；GPU使用0。

- checkpoint为有限tensor的state dict，state actor为300→1024→1024→512→512→24，hidden ELU，
  actor参数严格加载成功，五份形状编码均64维/有限。没有用零向量代替官方形状编码。
- 应采用++的状态观察合同：236:300为0.1×官方pc_feat；其target相关207:236置零。
  保留官方obs/action clamp及24D动作解释（base力/力矩与18个actuator目标），不能当成22DOF角度序列。
  原项目goal-conditioned观察与++状态checkpoint不可混搭。详细来源见[研究笔记](../../research/2026-10-14-unidexgrasp-admission.md)。
- 只读检查远端现有pointwam-expansion环境为Python3.10且没有isaacgym、gym、transforms3d。
  本地graspenv3.8已有这些组件，但本次检查时本地六张GPU已有任务，未额外分配或停止进程。
  现有Isaac Gym native bindings为Python3.6–3.8，不能在远端3.10环境里仅加PYTHONPATH冒充兼容。
  尚未新建环境、安装包或运行policy，checkpoint结构加载不证明回放成功。

证据：`download-manifest.json`、`audit-summary.json`、`checkpoint-audit.json`、
`plus/manifest.json`、`remote-runtime-preflight.json`。下一最小采集为一个物体、少量env、短rollout，
用兼容环境与实际GPU资源另记录run_id/预算；必须采到实际timestamp、q/root、物体SE(3)、
action及reset边界后才能构造30Hz窗口。future query用当前选定的mesh局部点经真实object pose传播，
不得跨reset拼接，也不得用最终抓取姿态插值替代执行轨迹。
