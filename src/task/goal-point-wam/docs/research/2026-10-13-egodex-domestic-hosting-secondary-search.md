# EgoDex 国内原始托管搜索与内容核验（2026-10-13）

本轮通过 ModelScope 当前前端的真实搜索接口找到此前网页搜索遗漏的候选，但仍未
找到经核实覆盖官方全部七个 split、保留当前原始 MP4/HDF5 且更快的国内来源。
其中 jinzhou5042 的四个 part 有原始配对布局，但抽查 HDF5 的内参为空；scarick
提供大小及前缀匹配的 TEST 候选；RoboFlywheel 是 LeRobot 转换数据。当前维持官方
生产下载和 GPU3 重建，训练保持用户要求的停止状态。

没有联系发布者、登录或创建账号，没有下载完整归档；只有有字节数上限的样本读取
和吞吐探针。非 ModelScope 平台调查的阴性结果不能概括为“国内没有候选”。

完整性门槛是 Apple 发布的 `part1..5/test/extra` 七个对象，或有可审查 split/episode
覆盖且保留原始 MP4/HDF5 的重打包。代码镜像、非视频监督、转换后数据及近似同名
数据集不满足这个门槛。
[官方原始发布说明](https://github.com/apple-aiml-research/ml-egodex)。

## 平台一级入口核对

以下访问使用本地已授权代理、匿名请求。POST 仅用于网站本身的只读列表搜索，接口
和参数从平台当前公开前端源码核对，没有猜测或绕过身份控制。

| 平台 | 实际一级证据 | 当前边界 |
| --- | --- | --- |
| OpenDataLab | 公开搜索返回 HTTP 200、`code=0`、`list=[]`、`total=0` | 关键词 egodex 的公开数据集列表没有命中 |
| OpenDataLab 自动建议 | HTTP 200，仅返回 `EgoDexter` | 名称不同，不能当作 EgoDex 原始镜像 |
| OpenXLab | 当前 `openxlab.org.cn` 本地代理访问 TLS 失败；定向搜索未见可核对原始入口 | 未完成该平台内部列表审查，不能据此认定没有数据 |
| WiseModel | 公开数据集搜索 HTTP 200、`code=0`、`list=[]`、`allCount=0` | 关键词 egodex 的公开数据集列表没有命中 |
| Gitee | 官方搜索 HTTP 200，仅命中 `mirrors_apple/ml-egodex` | 实际 README 下载链接仍指向 Apple CDN，是代码镜像，不是国内数据存储 |
| AtomGit AI | 首页、datasets 页面、search 页面可读；关键词搜索页面未给出可审查 EgoDex 文件列表 | 没有得到内部搜索 API 的完整结果，不能声称平台零命中 |

OpenDataLab 的实际请求：

```text
POST https://opendatalab.com/datasets/api/v3/datasets/list
{"keywords":"egodex","pageNo":1,"pageSize":20,"sort":"all"}
GET https://opendatalab.com/datasets/api/v3/search/autoSuggest?keywords=egodex
```

接口依据是当前公开
[datasetsV3 前端源码](https://webpub.shlab.tech/dps/opendatalab-web/xlab_v5.2103/assets/datasetsV3-b4eca022.js)
及其首页搜索组件。自动建议不是完整列表搜索，本文分别记录两者。

WiseModel 的实际请求：

```text
POST https://wisemodel.cn/project/api/v1/queryprojectlist
{"project_name_zh":"egodex","project_category_id":0,
 "baseReq":{"page":0,"pageSize":30},"project_type":"D"}
```

接口、名称筛选字段和数据集类型 D 来自当前公开
[前端源码](https://wisemodel.cn/assets/index-BH6Cdn3v.js)。网站入口：
[WiseModel datasets](https://wisemodel.cn/datasets)。

Gitee 的公开
[搜索响应](https://so.gitee.com/v1/search/widget/wong1slagnlmzwvsu5ya?q=egodex&from=0&size=20)
只有一个项目；[项目 README](https://gitee.com/mirrors_apple/ml-egodex/raw/main/README.md)
可直接匿名读取，仍引用 `ml-site.cdn-apple.com` 的官方 ZIP。

AtomGit 本轮实际访问的是
[datasets 搜索页](https://ai.atomgit.com/datasets?keyword=egodex) 和
[搜索页](https://ai.atomgit.com/search?keyword=egodex)。页面可读仅说明网站可访问，
不是数据对象可读或完整性的证据。

## 国内发布团队、研究组与网盘覆盖

平台定向公开搜索覆盖 `openxlab.org.cn`、`opendatalab.com`、`wisemodel.cn`、
`atomgit.com`、`gitee.com`；公开网盘关键词覆盖百度网盘、夸克、阿里云盘，以及
`pan.baidu.com`、`pan.quark.cn`、`aliyundrive.com`、`alipan.com`。另查 `edu.cn`
的 EgoDex 下载记录。这些检索没有产生可回到发布者并审查七个 split 的公开下载链接。
搜索引擎没有结果不是一级托管缺失证明，因此不把它升级成“没有国内镜像”的结论。

实际读到的国内团队一级记录也没有新增原始镜像：

- [小鹏 DIAL](https://github.com/xpeng-robotics/DIAL) 的 EgoDex 数据准备步骤要求从
  Apple 仓库下载原始数据，然后转换 LeRobot；没有另列国内原始 URL。
- [无极 MINT 中文 README](https://github.com/wuji-technology/wuji-ego-mint/blob/main/README_ZH.md)
  明确其公开 EgoDex 等数据是处理后的**非视频**监督，不是原始视频/HDF5 全量镜像。

没有采用商业“EgoDex 规格”宣传页、论文聚合页或网盘转载标题作为镜像证明。也没有
将 EgoDemo/EgoSuite、EgoDexter、EgoDex-R 等独立数据集误记成 Apple 原始全集。

## 给执行端的结论

非 ModelScope 调查没有新增可以直接测速的国内原始数据对象 URL，只有可复查的平台
列表搜索和代码 README URL。ModelScope 后续新增候选由 root 检查实际对象、内容与
吞吐，本报告不将页面存在或接近官方的文件大小当成完整原始数据等价证明。

可复查的匿名响应和有限页面/前端源码保存在项目
`tmp/egodex-domestic-hosting-20261013/`，其中
`opendatalab-search.json`、`wisemodel-search.json` 保存请求与零命中响应，
`so.gitee.com-search-response.txt` 保存 Gitee 搜索结果。文件只作诊断证据，不提交
大型页面副本；本报告保存本轮来源发现和核验结果。

## 后续 ModelScope 实际搜索突破与限定核实

root 从真实前端发现公开
[dolphin 搜索 API](https://modelscope.cn/api/v1/dolphin/datasets?PageNumber=1&PageSize=100&Query=egodex&Sort=default)，
本轮 HTTP 200、TotalCount=18，不是此前失败的通用 datasets 端点。精简搜索、
metadata/tree 已保存在同一诊断目录。这使国内候选搜索得到实质进展：

- `jinzhou5042/egodex_part1..4`：各有 7 个大 TAR.GZ，大小接近相应官方 ZIP 的
  数百 GB 规模。名称有 `ego4d_part*.tar.gz`，不能仅因文件名或大小断言其身份；
  root 负责读取压缩包前缀、核实原始 MP4/HDF5 布局与实测吞吐。
- `scarick/egodex-test-full`：tree 的顶层 `test.zip` 长度 17,304,529,397 bytes，
  精确符合现有官方 TEST 对象大小；root 负责实际 Range、目录与成员身份核对。
- `RoboFlywheel/EgoDex`：虽然 metadata 写 829h、存储约 1.956 TB，但下面的小文件
  核查已确认是转换布局，不能当七个原始对象的直接替代。

### RoboFlywheel 已确认转换

匿名 [meta/info.json](https://modelscope.cn/api/v1/datasets/RoboFlywheel/EgoDex/repo?Revision=master&FilePath=meta/info.json)
HTTP 200，约 63 KB；字段为：

```text
codebase_version: v2.1
total_episodes: 286727
total_frames: 73555307
fps: 30
splits: {"train":"0:286727"}
data_path: data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet
video_path: videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4
```

注释存成 Parquet，位姿特征例如 `observation.state.hip` 为 pos+rot6d 的 9 维；不是
原始 HDF5 schema。meta 树另有 provenance/raw_annotations JSONL，这些名称不能
证明视频或所有原始字段原样保留。当前 split 只有一个 train episode 区间，无法按
info.json 恢复官方七个 split。没有下载大 JSONL 或视频，也没有宣称 confidence 已
丢失（本轮未做对应字段审查）。证据为 `RoboFlywheel-info.txt` 和
`RoboFlywheel-meta-tree.txt`。

### jinzhou5042 的实际公开列表

从当前 ModelScope 前端确认的公开
[用户 datasets API](https://modelscope.cn/api/v1/dolphin/profile/jinzhou5042/datasets?UserName=jinzhou5042&PageNumber=1&PageSize=100)
返回 HTTP 200，内层 TotalCount=5，完整公开列表为：

```text
hgg
egodex_part4
egodex_part3
egodex_part2
egodex_part1
```

未出现 part5/test/extra 库；猜测路径 `egodex_part5`、`egodex_extra` 也返回 404。
这一结论仅限该用户当前匿名可见的列表，不排除私有仓库或其他账号。
`hgg` 的 metadata 只有默认说明，tree 只有 `hgg.tar.gz`（22,108,312,738 bytes）
及 README/gitattributes，身份仍未知，不把它归为 EgoDex。可供 root 做有上限前缀
核对的 [hgg 对象入口](https://modelscope.cn/api/v1/datasets/jinzhou5042/hgg/repo?Revision=master&FilePath=hgg.tar.gz)；
本代理没有读取这个大对象正文。列表与 metadata/tree 分别存为
`jinzhou5042-public-datasets.json`、`jinzhou-hgg-meta.json`、`jinzhou-hgg-tree.json`。

当前执行意义：已有国内原始候选值得进一步内容与吞吐核查；暂未确认覆盖全部七个
官方 split，也未确认新候选更快。非 ModelScope 阴性搜索不影响继续验证这些候选。


## root 实际原始内容核验与吞吐（本轮最终结果）

### ModelScope 搜索的可复现性

实际 GET 请求使用 `Query=egodex`，HTTP200、TotalCount=18、返回18行；大写
`Query=EgoDex` 返回相同18个身份。参数来自当前公开
[datasets 前端](https://g.alicdn.com/sail-web/maas/2.13.164/p__datasets__index.async.js)
与 [公共服务模块](https://g.alicdn.com/sail-web/maas/2.13.164/umi.js)。此前误用的
Name/Search/Keyword参数虽返回200，却给出未筛选热门列表，不能当关键词搜索
证据；旧通用 datasets 接口500也不能当零命中。

| 国内候选 | 本轮一级/正文证据 | 全量原始替代判断 |
| --- | --- | --- |
| [YuhengZhao/EgoDex](https://modelscope.cn/datasets/YuhengZhao/EgoDex) | 当前README声明Part1–5完整TRAIN重打包，根35个ZIP及repacked_zips；此前成员抽查见已有来源报告 | 尚未审计所有TRAIN覆盖；无test/extra，单独不是七split全集 |
| [jinzhou5042/egodex_part1](https://modelscope.cn/datasets/jinzhou5042/egodex_part1) 至part4 | 各7个TAR.GZ；四个首包均可精确206读取，解压前缀有同名MP4/HDF5；part2样本视频SHA256相同 | HDF5样本不符合当前原始合同；无公开part5/test/extra，不接入生产 |
| [scarick/egodex-test-full](https://modelscope.cn/datasets/scarick/egodex-test-full) | `test.zip`精确206，长度17,304,529,397，前1 KiB与Apple SHA256相同 | TEST候选，未完成全文件等价校验，无TRAIN/EXTRA |
| [RoboFlywheel/EgoDex](https://modelscope.cn/datasets/RoboFlywheel/EgoDex) | 约1.956TB但info.json为LeRobot v2.1、Parquet及重命名MP4，只有train区间 | 转换数据；大体积不能证明原始全集 |
| LLidabao/Egodex_selected | 根part1–5目录，但平台总存储约40.59GB及metadata.csv | 已选子集，无官方全集覆盖证明 |
| hliang/dreamzero-egodex-pretrain-no-videos | 根data/meta、MANIFEST、没有videos目录 | 非原始视频配对来源 |
| Voxel51/egodex-test-multimodal | FiftyOne文件metadata/samples及data目录 | TEST多模态发布，不能覆盖全部TRAIN/EXTRA |
| macrodata的8个egodex库 | 根仅JSONL、dataset_infos、README和gitattributes，JSONL约3KB–3.3MB | 调试/处理记录，没有原始大文件全集 |

原始内容不是根据tar文件名中的ego4d推断：远端四个包的前1MiB实际压缩正文可
解出以下成员头：part1 `add_remove_lid/0.hdf5`和`0.mp4`；part2
`assemble_disassemble_furniture_bench_chair/0.hdf5`和`0.mp4`；part3
`gather_roll_dice/0.hdf5`和`0.mp4`；part4 `pick_place_food/0.hdf5`和`0.mp4`。
解压输出有4MiB上限，只核对首批成员，不据此宣称所有成员完整。

### part2 单配对的关键不等价

追加读取part2首包前4MiB，完整提取第一个HDF5及MP4，仅在内存中比较已下载的
官方对应文件 `part2/assemble_disassemble_furniture_bench_chair/0`：

- MP4为1,855,837bytes，镜像/官方SHA256均为
  `46e74910c0f0e638b142b6b2de7fe34797b250afd278a5ac07fb74ad9a597379`。
- HDF5镜像706,448bytes、官方708,608bytes，SHA256不同。
- 数据集路径集合相同，但镜像`camera/intrinsic`形状为`(0,3)`，官方为`(3,3)`；
  镜像根属性缺少`which_llm_description`。其余当前比较的数据集/属性相同。

这个样本足以否定“该包所有HDF5与当前官方原样等价”的假设，但不足以推断全部
样本都坏。可能涉及不同历史发布或上传过程；本轮没有核实原因，不将它归因于
网络损坏或NAS权限。空内参无法直接支持当前重建合同，不通过静默补值掩盖差异。

另对用户公开列表中的唯一额外`hgg`读取64KiB压缩前缀，实际成员为
`hgg/HJ_MINLO_Pt-200ToInf_0.root`，不是发现一个遗漏的EgoDex原始part。只核对了
前缀，不对整个hgg做内容结论。

### 远端速度与实际动作

固定八连接，每连接4MiB，拒绝非精确206、错误Content-Range和截断正文；本任务
producer在身份和进程出生时间核验后短暂停止，finally恢复，独立180秒救援机制
保留；GPU3 consumer持续运行。只读探针不改变源或缓存合同。

| 来源 | 总成功正文 | 聚合MB/s | 请求完整性 |
| --- | ---: | ---: | --- |
| jinzhou part1首包 | 32MiB | 2.572 | 8/8、精确206 |
| scarick TEST | 32MiB | 2.740 | 8/8、精确206 |
| 同轮Apple TEST | 32MiB | 3.589 | 8/8、精确206 |

三个最终host分别是两个`cdn-lfs-cn-1.modelscope.cn`和官方
`ml-site.cdn-apple.com`。短测有明显时间波动，不能当长期带宽限额；它没有支持
切换新来源。TAR.GZ还需顺序解压，现有ZIP成员定位不能原样套用；既没有速度优势
又出现HDF5合同差异，当前不投入全量迁移适配。

结束快照：producer399615 RUNNING、part2完成93,991/95,125、下载失败尝试121
与预取重试32未增长；consumer360908完成3,626条、接纳29,119窗口、历史失败
17,479未增长；停止的训练supervisor378066不存在。快照不证明全部数据质量通过。

### 新增证据位置

本地同一诊断目录 `tmp/egodex-domestic-hosting-20261013/`：

- `modelscope-search-query.json`、`modelscope-search-query-uppercase.json`：实际关键词搜索。
- `*-tree.json`、`*-meta.json`：各候选公开根目录及说明。
- `remote-candidates.jsonl`：四个tar和TEST初步读取；`part2-hdf-comparison.jsonl`、
  `part2-video-hdf-comparison.jsonl`：单配对数据合同/字节比较。
- `remote-speed.jsonl`：隔离吞吐及producer恢复；`remote-hgg.jsonl`：hgg前缀排除。
- `final-production-status.json`：结束时远端状态。

远端本任务NAS根目录的`domestic-raw-candidates-1791861401`、
`domestic-hdf-contract-1791861544`、`domestic-hdf-contract-1791861584`、
`domestic-mirror-speed-1791861545`、`domestic-raw-candidates-1791861634`
均保存0644结果JSON，不包含临时签名URL或身份秘密。原始数据和checkpoint没有修改。
