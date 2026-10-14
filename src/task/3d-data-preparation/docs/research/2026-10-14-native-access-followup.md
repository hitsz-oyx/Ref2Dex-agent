# HOI4D / H2O / DexTrack 原生几何访问补查

时间：2026-10-14，11:18–11:26 UTC 左右。承接 [geometry-storage-survey](2026-10-14-geometry-storage-survey.md)。范围仅公开来源、响应头、分享目录和 reader 源码；没有下载数据大包、注册身份、接受协议、联系作者、训练或仿真。

**新增实证：HOI4D 的官方百度备份可匿名读取几何包文件大小，hand 425 MB、CAD 1.51 GB、annotations 22.25 GB，四个相关包共 24.19 GB 压缩。** 这解决了上一份报告的压缩输入大小未知，但尚未取得数据下载直链或 ZIP directory，因此解压大小、实际 hand 覆盖和 compact 大小仍未知。H2O 确认是真正的个人申请/HTTP basic-auth 门槛。DexTrack 的 OneDrive 本轮仍不能匿名读取包元数据，保留 unknown。

## 可行动的访问结论

| 来源 | 已取得 | 未取得 | 下一步所需用户动作 |
|---|---|---|---|
| HOI4D | 官方百度分享匿名目录与精确 ZIP bytes | 数据直链、ZIP directory、完整可用标注范围 | 使用现有百度网盘/浏览器访问官网备份，只提供 hand/CAD/annotations/camera 四个几何包的现有路径或可访问下载方式；无需 RGB/D |
| H2O | 当前注册表单与 pose 下载 401 challenge | 7 天凭据、各 pose 包 Content-Length | 本人填姓名、email、affiliation并同意官方条款，取得邮件凭据；已有已授权拷贝则只需提供路径 |
| DexTrack | 官方 references / assets 分享 URL 与读取字段 | 文件清单、压缩/解压总量、样例内容 | 用户浏览器确认 OneDrive 分享可开后，提供 reference ZIP 和目标 assets 文件列表/大小或已授权本地路径；不必先下载全套 assets/checkpoints |

这里的用户动作是缺少的实际入口或身份授权，不是要求批准研究设计。root 可以继续已有 TACO / HO-Cap / UniDexGrasp++ 路线，同时等待这些入口，不需把所有来源等齐才推进。

## HOI4D：官方百度备份仍能列出文件

官网可正常访问：[HOI4D official downloads](https://hoi4d.github.io/)。它目前所链的 [百度备份](https://pan.baidu.com/s/1ZeX8SzU-gnJ-wpO9ZukulQ?pwd=7gfs) 提取码是 **7gfs**。本次使用官方公开提取码访问分享，没有登录百度账号，也没有创建长期身份；服务返回匿名 `share/verify errno=0`，`share/list errno=0`。

`/HOI4D` 下本次实际目录元数据：

| 文件 | 精确压缩 bytes | 十进制大小 | fs_id（公开分享中的文件标识） |
|---|---:|---:|---:|
| HOI4D_Hand_pose.zip | 425,303,390 | 425.30 MB | 471900079579407 |
| HOI4D_CAD_Model_for_release.zip | 1,512,101,743 | 1.51 GB | 671427074898886 |
| HOI4D_annotations.zip | 22,252,507,455 | 22.25 GB | 40327043893606 |
| camera_params.zip | 2,144 | 2.14 KB | 185659639904150 |
| HOI4D_release.zip（不列入本路线下载） | 23,269,777,121 | 23.27 GB | 234114062591967 |
| DEPTH/（目录，不列入本路线下载） | 目录 size=0，不能解释成内容 0 B | 未盘点 | 303476297881464 |

四个几何相关 ZIP 共 **24,189,914,732 B / 24.19 GB / 22.53 GiB**。该值来自官方所链存储服务的文件元数据，不是实下载包后的 `du`，也不是可训练数组总量。annotations 可能含大量 2Dseg/3Dseg，不能把 22.25 GB 全当 pose；拿到可 Range 的 ZIP URL 或本地 ZIP 后应仅提取 `objpose`、需要的 `action` 及 camera trajectory 等成员。[官方 annotation 结构](https://github.com/leolyliu/HOI4D-Instructions#data-organization)

**camera_params.zip 只有约 2 KB，不能假设它含所有移动相机的逐帧轨迹。** 官方 instructions 的 `3Dseg/output.log` 才描述相机轨迹读取入口。纯几何 adapter 仍须提取这条坐标转换链；无需保留同目录的整个 raw/label point cloud。[官方 camera pose 说明](https://github.com/leolyliu/HOI4D-Instructions#3d-static-scene-panoptic-segmentation)

元数据可执行复现方法（仅列目录，不请求数据内容）：

```python
import http.cookiejar, json, urllib.parse, urllib.request

share = "https://pan.baidu.com/s/1ZeX8SzU-gnJ-wpO9ZukulQ?pwd=7gfs"
headers = {"User-Agent": "Mozilla/5.0", "Referer": share}
opener = urllib.request.build_opener(
    urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
)
opener.open(urllib.request.Request(share, headers=headers), timeout=15).read(200000)
verify = urllib.request.Request(
    "https://pan.baidu.com/share/verify?surl=ZeX8SzU-gnJ-wpO9ZukulQ",
    data=urllib.parse.urlencode({"pwd": "7gfs", "vcode": "", "vcode_str": ""}).encode(),
    headers=headers,
)
with opener.open(verify, timeout=15) as response:
    assert json.load(response)["errno"] == 0
query = urllib.parse.urlencode({
    "uk": "1054684671", "shareid": "38955042415", "dir": "/HOI4D",
    "order": "other", "desc": "1", "showempty": "0", "web": "1",
    "channel": "chunlei", "clienttype": "0", "num": "100", "page": "1",
})
request = urllib.request.Request("https://pan.baidu.com/share/list?" + query, headers=headers)
with opener.open(request, timeout=15) as response:
    listing = json.load(response)
assert listing["errno"] == 0
for item in listing["list"]:
    print(item["server_filename"], item["size"], item["isdir"])
```

这是一次公开服务的实测方法，不是百度承诺长期稳定的 API。若返回验证码或登录要求，应由用户使用现有浏览器处理，不绕过验证。anonymous sharing 页本轮没有可用 `sign/timestamp` download fields；未完成匿名下载直链核验，不能把匿名目录成功写成数据下载已打通。

OneDrive：[CAD](https://1drv.ms/u/c/12e5c3dbeffd0594/EZQF_e_bw-UggBIsAQAAAAAB5gBbXuK7eDzXQmzDS5W09g?e=DdkFTE)、[hand](https://1drv.ms/u/c/12e5c3dbeffd0594/EZQF_e_bw-UggBIrAQAAAAABHyHqDhs5CpJAkWTAQoGWxQ?e=ALFMCD)、[annotations](https://1drv.ms/u/c/12e5c3dbeffd0594/EZQF_e_bw-UggBIuAQAAAAABELgpzxDxCz58Qov76-wpvw?e=qhyY0h)。本轮直接 GET 403；标准匿名 shares API 401 `unauthenticated`；一次经本机现有代理的 CAD GET 也 403。没有得到 content metadata。这些响应不足以证明官方分享失效或用户本人没权限，浏览器与脚本访问结果可能不同。

来源审计线索：官方仓库公开 issues 有 hand coverage 不完整、objpose JSON `frameId` 与文件名不一致、camera1 缺失的用户报告；它们**不是作者确认事实**，但为真实 pilot 明确了最小审查项：framewise hand/object intersection、取时间键规则、相机坐标转换、缺失帧阻断。不要用公开总 sequence 数直接外推可训练规模。[issue #31](https://github.com/leolyliu/HOI4D-Instructions/issues/31)、[issue #21](https://github.com/leolyliu/HOI4D-Instructions/issues/21)

## H2O：需要用户本人取得当前凭据

[ETH 官方申请入口](https://h2odataset.ethz.ch/) 本轮 HTTP 200，页面列 First Name、Last Name、Email、Affiliation，以及 `I Agree`；网站明确邮件发送访问凭据，有效 **7 天**。条款限定学术用途、禁止商用和转给第三方；使用已有数据的授权仍要与该来源对应。

本轮对 `https://h2odataset.ethz.ch/data/dataset/subject1_pose_v1_1.tar.gz` 的未认证请求实际得到：

```text
HTTP 401
WWW-Authenticate: Basic realm="Please register to access content"
```

因此这是实际身份/条款门槛，不能靠公开 URL 或网络代理解决；我没有提交表单。用户不必手工下载图像：获得授权凭据后由 root 有界请求 `subject{1..4}_pose_v1_1.tar.gz`、`object.zip`、`label_split.zip` 的 Content-Length，先取一个 subject 的几何 pilot。[官方用途与凭据说明](https://h2odataset.ethz.ch/)

约 78 MB/subject、5.6 MB mesh、0.3 MB split 的数字只见于下游 NVIDIA adapter 文档，本轮没有 ETH 已认证响应头支撑，所以保留**约 0.32 GB 的未核实压缩估算**，不能作为已测完整 native input 大小。MANO 模型如仓库已有相应授权拷贝则沿用，不重复申请。[NVIDIA H2O geometry-only adapter 说明](https://github.com/nvidia-isaac/video_to_data/blob/main/robotic_grounding/docs/H2O_SETUP.md)

## DexTrack：references 可用于几何候选，发布大小仍未知

官方 README 当前直接列：[retargeted data](https://1drv.ms/f/c/c746413ba7b58f04/EmTCrn1XSShNn34d3mTr7b4BBk1W4yqJxjY1y3WN0KRm3A?e=dxXnuT)、[assets part 1](https://1drv.ms/u/c/c746413ba7b58f04/EVEQ8nNWsnROk5swNPIcqR8Bfq6eb716MFJTbkMneJEvew?e=1JbI7k)、[part 2](https://1drv.ms/u/c/c746413ba7b58f04/Ed7YUar7_0lAvf9B8YQCZC0BEuaSr_5oSO65qR18RwBNvw?e=QC48Nn)、[part 3](https://1drv.ms/u/c/c746413ba7b58f04/EWEaowIuf7BMrL2NMSBXzM4BoXxjsc9HUp7wm36aLjdC5A?e=ccINfB)。数据目录目标是 GRAB_Tracking_PK_reduced_300、GRAB_Tracking_PK_LEAP_OFFSET...、modified_kinematics...、TACO_Tracking_PK_reduced；mesh assets 与 checkpoints 分开。[官方 Data](https://github.com/Meowuu7/DexTrack#data)

本轮 retargeted data 短链 GET 403，第一组 asset 的匿名 shares API 401。没有获得可核验 bytes/成员目录。官方 README 未给总大小；HF 搜索与公开 issue 未核实到作者发布的替代几何包，因此不给“实际约多少 GB”的伪精确数字。[官方 artifacts 请求 issue](https://github.com/Meowuu7/DexTrack/issues/1)

补充可执行替代线索：公开 HF API 检索得到社区派生仓库 `liang12121/dextrack-wuji-mjlab-assets`，它**不是原作者官方 release**。其 `isaacgymenvs/data/` tree 包含上述 GRAB/TACO 原名目录，以及多套 WUJI 新增数据。可逐文件取一条 reference 和唯一 CAD，不必下载整个训练环境。社区仓库的 `data_tars/isaacgymenvs_data.tar` 元数据是 **3,283,640,320 B / 3.28 GB**，`data_tars/assets.tar` 是 **10,393,579,520 B / 10.39 GB**；总 **13,677,219,840 B**，这是该混合镜像的 TAR 大小，不是官方 ZIP 大小或我们需要的纯几何总量。使用前仍需核继承许可、修改差异、原始 GRAB/TACO lineage、embodiment；本轮未下载或准入。[社区包 tree metadata](https://huggingface.co/api/datasets/liang12121/dextrack-wuji-mjlab-assets/tree/main/data_tars?recursive=false&limit=1000)、[社区 reference 目录 metadata](https://huggingface.co/api/datasets/liang12121/dextrack-wuji-mjlab-assets/tree/main/isaacgymenvs/data?recursive=false&limit=1000)

该社区 mirror 的 `GRAB_Tracking_PK_reduced_300/data/` 第一页 1000 个 NPY 合计 35,333,000 B；TACO 相应目录第一页 1000 个文件合计 48,696,163 B。两页都有 next-link，**这不是全量总数/大小**。例如 `passive_active_info_ori_grab_s10_airplane_fly_1_nf_300.npy` 是 35,333 B，适合有界 reader pilot。TACO 示例文件名含 `ori_grab_s2_phone_call_1_interped_taco_...`，显示它可能是跨 GRAB/TACO 的 modified reference，不能按原生 TACO sequence 独立划分来源。[GRAB 文件 metadata](https://huggingface.co/api/datasets/liang12121/dextrack-wuji-mjlab-assets/tree/main/isaacgymenvs/data/GRAB_Tracking_PK_reduced_300/data?recursive=false&limit=1000)、[TACO 文件 metadata](https://huggingface.co/api/datasets/liang12121/dextrack-wuji-mjlab-assets/tree/main/isaacgymenvs/data/TACO_Tracking_PK_reduced/data?recursive=false&limit=1000)

源码确实区分两条数据链：`_load_mocap_info` 从 `robot_delta_states_weights_np`、`object_transl`、`object_rot_quat` 读取 reference；`_load_optimized_traj` 另读 `shadow_hand_dof_pos`/`object_pose` 或 `shadow_hand_dof_tars`。这证明官方实现能够消费参考和执行记录两种形式，但不能证明公开压缩包包含全部物理控制日志。[官方 Allegro loader](https://github.com/Meowuu7/DexTrack/blob/main/isaacgymenvs/tasks/allegro_hand_tracking_generalist.py)

最小入库应只取一条 reference、对应 object mesh/scale、对应 hand URDF，以及实际时间/坐标/关节映射，随后从 FK 产生手点；无需为几何 reader 安装整个 RL 环境或启动策略。读取时还要核手根 Euler/quaternion 与关节重排、ground/table offset、downsample 设置。retargeted kinematic targets 不等于曾实际执行的 action，不能直接监督我们 Inspire 动作头。GRAB/TACO 来源与原人手数据必须共享 split lineage。

仅作 compact 工程预算：若 22 维 robot root+joint state、7 维 object pose 用 float32，裸数组为 **116 B/step**；再保存手语义点、时间戳和 masks 可按约 **0.15–0.5 GB/百万时间步**，外加每个唯一 CAD 一份。这不是 DexTrack 原 ZIP 的估算：原包可能含 dense vertices、重复样本或仿真多环境记录，未知前不能将这个数组预算替代下载成本。

## 当前决策

HOI4D 现已可具体告诉用户需要哪四个文件以及压缩体积；下一步应拿可读 ZIP/已授权路径，审计 pose-only 成员与 coverage，先 rigid pilot。H2O 等个人凭据。DexTrack 等公开包目录/已有拷贝；因为它和 GRAB/TACO 同源，而且已可继续人手几何来源与 UniDexGrasp++ rollout，不必为了接一个 retargeted 候选长时间阻塞扩产。
