# EgoDex 下载来源与下一步吞吐诊断（2026-10-13）

当前没有找到已验证优于 Apple CDN 的完整替代来源。不能据约 7.5 MB/s 的观测直接
认定 Apple 配置了限速。本文是来源研究和最小诊断建议，不启动下载、不修改生产配置。

## 官方来源与镜像边界

Apple 当前 README 只给出 `ml-site.cdn-apple.com/datasets/egodex/` 下的
`part1.zip` 至 `part5.zip`、`test.zip`、`extra.zip`，未公布第二个下载域名或云对象存储
入口。官方研究页也指向论文与仓库，没有另列国内镜像。应继续将这些官方对象作为原始
数据身份基准。[Apple 仓库](https://github.com/apple-aiml-research/ml-egodex)、
[Apple 研究页](https://machinelearning.apple.com/research/egodex-learning-dexterous-manipulation)。

已有实测与内容核对保存在当前
[实验卡](../experiments/probes/P-20261012-goal-point-wam-video-points.md)：
ModelScope `YuhengZhao/EgoDex` 是 TRAIN-only，所核对成员可以匹配官方成员，但远端
吞吐约 6.56 MB/s，尚不能代替全量七个对象。HF `zhenyuxie-zhzh/egodex` 的上传者
声称保存了官方 ZIP 的字节分块，并提供分块清单；这是上传者声明，不是本任务完成的
全量等价审计。已有匿名 HF/镜像访问 401，原因未确定。值得注意的是其当前布局是
`chunks/<archive>.zip/<archive>.zip.part*`，不能用不存在的顶层 `test.zip` 路径来判断
整个镜像不可用；如再次检查，应先读公开树和分块清单。
[镜像上传者数据卡](https://huggingface.co/datasets/zhenyuxie-zhzh/egodex)。

## 本轮已测方向：不要重复扩大无效搜索

下表来自 root 当前隔离吞吐探针回报，正式产物路径由 root 在实验卡中补充；这里不将
单次顺序测量包装成因果结论。每连接 8 MiB Range，成功并不等于稳定生产吞吐。

| 隔离条件 | 聚合吞吐 MB/s | 当前判断 |
| --- | ---: | --- |
| requests 并发 8 / 16 / 32 / 再测 8 | 3.55 / 6.83 / 7.24 / 6.03 | 32 未显示大幅提升，时间变化明显 |
| curl 固定 IP 17.253.61.133 / .139 | 4.64 / 6.55 | 未超过此前生产约 7.5 |
| 总 8 连接混合 part2/3/4/5 | 6.32 | 跨 ZIP 未显示明显收益 |

Google DNS 返回 .148/.154、Cloudflare 返回 .143/.133，均在本轮观测的
17.253.61 段。地址差异本身不能证明不同物理 CDN 节点或不同国际路径。
`curl --resolve` 可以只覆盖单次连接的域名解析，无需改系统 DNS，但这些单次结果
没有支持生产固定某个 IP。[curl 官方手册](https://curl.se/docs/manpage.html#--resolve)。

## 仍可能改变决策的低成本检查

1. **Range URL 的查询参数对照。** 当前
   `src/task/cm-pointflow-effect-pretrain/tools/run/ref5_data_expansion/download_egodex.py`
   的 `RemoteZip.read` 为每个请求加 `?ref5_range=<start>-<end>`，注释说明用于规避
   中间缓存返回错误 Range。HTTP 缓存规范将方法和目标 URI 作为缓存 key 的基本组成，
   因此不同查询参数可能阻碍共享缓存复用；Apple 是否按这个方式缓存，尚无证据。
   用同一对象、相同并发、多个不同区间做 query / bare URL / query 对照，保留精确
   `206`、`Content-Range`、长度与现有 ETag 检查，并重复一个区间检查一致性。bare URL
   一旦返回错误区间即停止，不直接删除生产参数。只有正确且重复出现实质收益时才考虑
   实现改动。[RFC 9111 §2](https://www.rfc-editor.org/rfc/rfc9111.html#section-2)。
2. **网络路径与传输协议。** 本机没有公网 IPv6 默认路由；客户端 TCP 接收乱序计数较高，但该计数不能直接证明丢包或 CDN 限速。高重传与吞吐下降若在
   同时段相关，可支持继续查路由；重传低不能单独证明 CDN 限速。HTTP/2 可在既有 curl
   构建支持时做短对照，但必须记录实际协商版本，不能仅凭 `--http2` 参数判断使用了
   HTTP/2。不为这个探针安装系统软件或更改网络配置。
   [curl 协议选项](https://curl.se/docs/manpage.html#--http2)。
3. **镜像可访问性与分块路径。** 若官方路径与协议检查均无收益，再对 HF 分块清单中
   的一个真实公开对象做短 Range 检查。可读且实测速率更好后，才讨论跨分块映射与
   已完成成员复用；遇到 401 不自动申请新账号，也不将它解释成已确定的权限根因。

Range 规范允许服务端忽略 Range，因此探针必须拒绝 `200` 或错误区间，避免意外下载
整个数百 GB 对象。若使用 `If-Range`，其验证失败会转成完整响应；这里仍应在读取正文前
拒绝非精确 `206`。[RFC 9110 §14](https://www.rfc-editor.org/rfc/rfc9110.html#section-14)、
[If-Range 语义](https://www.rfc-editor.org/rfc/rfc9110.html#section-13.1.5)。

当前决策：维持已可用的官方来源；优先完成 query 缓存与现有网络探针，暂不投入完整
镜像迁移、无界提高并发或完整 ZIP 的额外复制。所有建议均是待测工程假设，不能据此
宣称找到提速方案。

## 追加：公开托管搜索与可访问性实查

2026-10-13 本地已授权代理、无 token、仅 API/README/HEAD；没有启动数据下载。
`https://huggingface.co/api/datasets?search=egodex&limit=100` 返回 200，列出约 40 个
公开仓库。因此先前某个 EgoDex 镜像的 401 不代表所有 HF 数据集均不可访问。
API 的元数据可读、HEAD 成功和远端正文吞吐是三个不同检查，本节未测吞吐。

| 新来源 | 本轮公开访问 | 内容与完整性边界 |
| --- | --- | --- |
| [Wehere/EgoDex](https://huggingface.co/datasets/Wehere/EgoDex) | API/tree 200；`add_remove_lid.tar` HEAD 200，4,841,021,440 bytes，支持 bytes Range | 根目录 118 个任务 TAR，总约 578.05 GB；无 README，不能确定保留原始 MP4/HDF5、是否重编码、覆盖哪几个官方 split |
| [UCBProject/EgoDex](https://huggingface.co/datasets/UCBProject/EgoDex) | API/tree 200；`test/add_remove_lid/0.mp4` HEAD 200，1,472,629 bytes | 根目录只有 test，含原始命名 MP4/HDF5 及 HaWoR 输出；不是全量来源，适合 HF 网络对照 |
| [griffinlabs/EgoDex-LeRobot-v3.0](https://huggingface.co/datasets/griffinlabs/EgoDex-LeRobot-v3.0) | API/README 200 | 上传者声明 train/test 转成 LeRobot，约 1.47 TB；明确丢弃 confidence；未宣称提供 extra，需适配与语义审查 |
| [yxgao/egodex-wds](https://huggingface.co/datasets/yxgao/egodex-wds) | API/README 200，749 个文件 | WebDataset，包含视频、transform/confidence NPY；数据卡有 original/target_resolution，原始分辨率与 split 覆盖未核实，不能当原始镜像 |
| [liang12121/dreamzero-egodex-raw](https://huggingface.co/datasets/liang12121/dreamzero-egodex-raw) | API/tree 200 | 根只有 index/manifests；索引 JSON 321 MB，JSONL 285 MB；名称 raw 不意味着托管原始视频，尚未解析索引，不下载大索引 |
| [D1ckyTwist3r/EgoDex](https://huggingface.co/datasets/D1ckyTwist3r/EgoDex) | API 200，无 README | 17,231 文件，Parquet/LeRobot 布局；转换数据，完整性未知 |

两个可立即交给 root 做有上限远端网络对照的真实 URL：

```text
https://huggingface.co/datasets/Wehere/EgoDex/resolve/main/add_remove_lid.tar
https://huggingface.co/datasets/UCBProject/EgoDex/resolve/main/test/add_remove_lid/0.mp4
```

Wehere 文件树证据来自公开
[tree API](https://huggingface.co/api/datasets/Wehere/EgoDex/tree/main?limit=200)，
UCB 根目录证据来自
[root API](https://huggingface.co/api/datasets/UCBProject/EgoDex/tree/main?limit=100)。
精简响应保存在项目 `tmp/video-training-20261012-takeover/mirror-discovery/` 下的
`Wehere-tree-20261013.json`、`UCB-root-20261013.json`、`Liang-root-20261013.json`。
API siblings 对大型仓库返回恰好 100,000 行时视为可能截断，不用于声称完整统计。

ModelScope 另外找到的 `luyitas/egosim_egodex_egovid_full`，匿名 repo/tree 返回
403（无权访问），metadata 返回 404；不能确认不存在还是隐藏。搜索索引称为 5,000
EgoDex clips 的 16 fps/720p 子集，但无法从当前可读一级来源核实，不推荐替换。
YuhengZhao 的公开 repo/tree 仍为 200。通用 ModelScope datasets 查询返回 500，
没有据此声称已穷尽平台所有仓库。

Apple 旧 issues 页面可读的 #1/#3/#4 分别讨论 LeRobot 转换、数据更新、坐标约定，
未见其他托管地址。当前仓库页面 issues 为 0，旧仓库 GitHub API 因代理出口公开
限额返回 403，评论未完整可读；不能声称全部历史 issues 已审查。

新增决策顺序：root 先比较多个独立站点下载，区分整体出口问题和 Apple 路径问题；
若 HF 正文实测更快，再优先审计 Wehere TAR 中少量可映射的原始配对及 split；如果
它是删减或重编码数据，保留其网络诊断价值，但不把它投入现有原始数据重建队列。

root 后续远端实查：Wehere TAR 在官方 HF 和 hf-mirror 路径均返回精确 `0–1023`
的 `206`，最终托管 `us.aws.cdn.hf.co`；UCB 示例视频也返回 `206`。仍未据此宣称速度
更快。不同站点短探针中，Apple 3.38→复测 6.62 MB/s、PyPI 2.63、Anaconda 5.97、
上海交大 Ubuntu 镜像 34.84（8 连接、每请求 4 MiB）。这些短测已排除“所有互联网
下载都同样慢”的说法，但国内结果还需长一点的探针排除 burst；不能区分国际出口和
各国外 CDN 路径的影响。结果产物见后续独立互联网来源对照记录。

Wehere 覆盖进一步只读核对：现有远端官方 `part2.json` inventory 的 13 个 HDF5
任务目录，均在其 118 个 TAR 任务名中出现。此处仅证明任务名覆盖；episode 数量、
MP4 编码、置信度、session 字段与官方 split 归属未核实。根目录没有 test/extra/part
标识，当前不能确认全量。HF API 将它自动标记 `format:webdataset`，但平台可能仅按
TAR 扩展名推断，不能把这个标签单独作为转换证明。

## 追加：原始配对候选与 HF 区域选择边界

继续查公开 HF API 和上传者数据卡，没有发现第二个可匿名读取的全量官方 ZIP 镜像。
以下结果是有意义的排除，不能把同名仓库数量当成可用镜像数量：

| 候选 | 匿名 API | 对当前原始下载的限制 |
| --- | --- | --- |
| [szang18/egodex_train_eval](https://huggingface.co/datasets/szang18/egodex_train_eval) | 200，259 个文件 | 有 extra/train/test 命名的 HDF5/MP4，但最多约 129 对的小抽样，无全量证据 |
| [licesma/egodex_sub](https://huggingface.co/datasets/licesma/egodex_sub) | 200，23 个文件 | 约 11 对小抽样，无全量证据 |
| [jungwook2358/egodex-hdf5](https://huggingface.co/datasets/jungwook2358/egodex-hdf5) | 200，2 个文件 | 仅 `basic_pick_place_hdf5.tar`，没有对应原始视频全集 |
| [RayYoh/egodex_hdf5](https://huggingface.co/datasets/RayYoh/egodex_hdf5) | 200，7 个文件 | 没有原始 MP4/HDF5 配对树，不能替代本任务全量原始获取 |
| [abraranwar/egodex_rfm](https://huggingface.co/datasets/abraranwar/egodex_rfm)、[jesbu1/egodex_rfm](https://huggingface.co/datasets/jesbu1/egodex_rfm) | 200；大量 `trajectory_*.mp4`，Parquet reward 数据卡 | 未见 HDF5，任务/episode 映射需审计；本轮没有确认 MP4 是否原样、重编码或裁剪，不作为原始配对镜像 |

`szang18/egodex_train_eval_part1` 只有 `.gitattributes`，其他 LeRobot/WDS 来源沿用
前面的转换边界。公开搜索仍仅找到先前 ModelScope YuhengZhao 的原始 TRAIN 转存；
平台通用搜索 API 未成功，不能声称全平台不存在其他原始仓库。

HF 官方 [Storage Regions](https://huggingface.co/docs/hub/main/storage-regions)
当前列 US/EU，Asia-Pacific 为 Coming soon。区域由拥有仓库的 Team/Enterprise
组织配置，普通仓库默认 US；这是存储位置设置，没有公布给下载者把别人的 US 仓库
切到亚洲的参数。不能将开通账户或替换 URL hostname 视为已有授权内的提速方案。

官方 [下载文档](https://huggingface.co/docs/hub/main/datasets-downloading)
列出 `us.aws.cdn.hf.co` 和 `us.gcp.cdn.hf.co`。HF 人员在
[下载重定向公告](https://discuss.huggingface.co/t/how-to-get-a-list-of-all-huggingface-download-redirections-to-whitelist/30486?page=2)
说明 GCP 域名适用于来自 Google Cloud Platform 的流量。不同实际云出口可能改变
服务端重定向，但没有文档支持下载者通过参数强制云区域。现有签名 URL 不应随意
替换域名；获取一个 GCP 出口或另购机器是新的资源/身份问题，本文不启动这些操作。

官方 [hf_xet 下载说明](https://huggingface.co/docs/huggingface_hub/en/guides/download#faster-downloads)
提供的是 CAS/xorb 分块并行与去重，不承诺改变跨境网络路径；这里没有可读全量原始
HF 仓库可用于实测，故不安装 SDK 或开始复制数据。当前更有价值的待决策信息仍是
root 的实际出口与已知原始来源吞吐，而非再列处理后同名数据集。

## 后续实际结果

- 同一八区间四轮 bare/query/query/bare 对照为4.52/3.55/3.85/3.66 MB/s；
  全部精确206、长度和 ETag一致，逐区间SHA256相同。没有稳定的大幅缓存收益，
  因此保留生产中的 range-specific query。
- Google DNS 带不同 ECS 的观测又得到17.253.87、17.253.95、17.253.5网段；
  不修改系统解析，只通过保持域名/TLS的curl resolve测试。四连接聚合分别
  2.54/3.15/4.63 MB/s，同轮默认17.253.61地址6.60 MB/s。未支持固定其他网段。
- 较早对 `zhenyuxie-zhzh/egodex` 的 HF 与 hf-mirror tree API匿名访问均返回401；这不是顶层ZIP
  不存在的404，仍无法取得公开分块清单。不使用新身份或私有token。
- 国内Ubuntu镜像对照未取得有效吞吐：阿里入口降级HTTP跳转后连接被拒绝，
  对跳转目标使用HTTPS也连接超时；清华ISO返回403。因此不能据此推断服务器
  公网总带宽上限，7–8 MB/s仍可能来自网络路径、CDN策略或其他共享瓶颈。
- 当前请求直接停止小样本训练，已保存并核验step31338 checkpoint；原始数据下载
  和GPU3重建继续。详细停止记录、吞吐证据目录及生产决策见
  [本轮运行记录](2026-10-13-egodex-stop-and-download-probes.md)。

HTTP协议补充：固定同一IP、八连接的HTTP1.1/HTTP2/HTTP2/HTTP1.1对照为
4.042/4.82/5.502/7.206 MB/s；实际http_version分别1.1/2/2/1.1且区间和ETag一致，
未支持为HTTP2重写生产传输。该结果也显示顺序实验的明显时间波动。

## 当前决策更新：独立互联网来源实测

已完成更大国内对照：上海交大Ubuntu镜像8连接、总256MiB读流测得67.53 MB/s，
同轮Apple8连接64MiB为6.86 MB/s。它反驳“远端所有互联网下载均只有7–8 MB/s”，
不能把某个来源的短测速当作服务器长期公网额定带宽。PyPI2.63与Anaconda5.97
也明显低于该国内源，说明现象不局限于EgoDex单个文件；数据源/CDN与所走网络
路径是当前更有价值的诊断方向，尚不能证明Apple限速或全部国际流量受限。

Wehere真实TAR8连接64MiB测得3.38 MB/s，经hf-mirror入口4.76 MB/s，二者
最终均到`us.aws.cdn.hf.co`。TAR前8个512-byte headers中发现
`add_remove_lid/actions/0_0.pt`、`0_1.pt`，即至少包含已经处理的动作产物；
并非仅根据平台format标签推测。但未遍历全部内容，不能断言没有原始MP4/HDF5。
当前没有速度或原始完整性证据支持切换生产源。

ModelScope现有原始repacked候选的实际跳转hostname为
`cdn-lfs-cn-1.modelscope.cn`，不能仅因域名在中国就假定一定沿国内快路径。
详细网络对照、后续ModelScope复测及状态见
[互联网与EgoDex来源对照](2026-10-13-egodex-internet-source-comparison.md)。

ModelScope实际CDN补测八连接32MiB为3.53 MB/s，紧接的交大32MiB对照25.63；
当前不能用“国际出口普遍受限”解释所有慢源，也不能保证国内平台更快。
能确认的是独立国内来源明显超过EgoDex当前链路，远端整体统一低速假设已排除。

追加路线诊断：本地代理峰值未能稳定复现，SSH中转慢，完整读流没有稳定收益；
HTTP/3校准通过但Apple当前连接超时，ModelScope基线403不可判定。生产继续
官方直连，证据和结束快照见[追加路线诊断](2026-10-13-egodex-route-followup.md)。


## 国内托管搜索更新

从ModelScope真实前端核实的公开dolphin接口使用Query参数，返回18个EgoDex候选，
发现此前网页索引遗漏的jinzhou5042原始part1–4重打包及scarick TEST。part2抽查的
MP4与官方SHA256相同，但HDF5内参变成(0,3)，且缺which_llm_description属性，
不能原样接入现有重建。RoboFlywheel1.956TB实际是LeRobot v2.1。新两个国内源
短测速2.57/2.74 MB/s，同轮官方3.59，无稳定速度优势；仍未确认七split国内原始
全集。非ModelScope平台公开搜索和所有新内容证据见
[国内托管搜索与内容核验](2026-10-13-egodex-domestic-hosting-secondary-search.md)。
