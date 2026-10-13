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

## 后续实际结果

- 同一八区间四轮 bare/query/query/bare 对照为4.52/3.55/3.85/3.66 MB/s；
  全部精确206、长度和 ETag一致，逐区间SHA256相同。没有稳定的大幅缓存收益，
  因此保留生产中的 range-specific query。
- Google DNS 带不同 ECS 的观测又得到17.253.87、17.253.95、17.253.5网段；
  不修改系统解析，只通过保持域名/TLS的curl resolve测试。四连接聚合分别
  2.54/3.15/4.63 MB/s，同轮默认17.253.61地址6.60 MB/s。未支持固定其他网段。
- 本轮对 HF 与 hf-mirror 的真实仓库 tree API匿名访问均返回401；这不是顶层ZIP
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
