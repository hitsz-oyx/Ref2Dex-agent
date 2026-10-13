# EgoDex 追加传输路线诊断（2026-10-13）

用户要求继续寻找更快方式。本轮工程诊断的决策是：是否存在足以替换现有官方直连的
稳定快路径。结论：当前已测路线没有提供这种证据，继续现有八 worker、32 MiB
物理顺序预取，不重新启动用户已停止的训练。单次峰值不能作为生产配置选择依据。

## 探针范围和执行边界

只做有字节数、连接数和时间上限的探针。远端原始产物仍留在授权 NAS；没有复制
完整 ZIP、本地没有共享 NAS 挂载。部分隔离吞吐实验用已有身份核验和超时恢复
机制暂停本任务 producer，结束后恢复；GPU3 consumer 不暂停。没有修改系统 DNS、
代理控制器或生产 Python 环境，没有申请新身份或付费出口。

HTTP/3 诊断使用 curl_cffi 0.13.0，依赖仅安装在本任务 NAS 独立
`http3-probe-deps-1791860157/packages`；安装及缓存均有独立目录。使用 TLS 证书
校验、强制 V3ONLY、不允许降级，正文上限 1 KiB。Cloudflare 校准证明当前客户端
和远端可以实际协商 HTTP/3，不能用下载源失败推断所有 UDP 出口不可用。

## 实际结果

| 方向 | 测量 | 对下一步的影响 |
| --- | --- | --- |
| 本地 Apple 直连 | 八连接，每连接 8 MiB，两轮仅 4/8、5/8 请求完成，25秒超时预算 | 未形成可用快路径 |
| 本地已授权代理 | 首轮 7/8 完成；下一轮 8/8 完成、12.67 MB/s；追加三轮分别 7/8、7/8、2/8 完成 | 峰值不稳定，不能切换生产 |
| 本地到远端 SSH | 随机 16 MiB，双方 SHA256 一致；含连接建立16.87秒，0.994 MB/s | 中转链路本身慢；本地下载快也不能保证交付快 |
| 八条独立 SSH 并发 | 至少一条超过45秒，聚合探针没有完整结果 | 无效吞吐测量，不能当成0 MB/s或宣称所有连接失败 |
| Apple Range / 无 Range 读流 | 四连接，各读8 MiB；range/stream/stream/range 为2.660/3.263/5.000/6.569 MB/s | 时间波动明显，无完整读流稳定收益 |
| ModelScope Range / 无 Range 读流 | 相同限制；6.508/6.605/6.379/6.114 MB/s | 没有足以重写传输的收益 |
| Apple HTTP/3 | HTTP/2精确206、1 KiB成功；强制HTTP/3约5秒连接超时 | 当前官方地址和网络路径下不可用 |
| ModelScope HTTP/3 | HTTP/2基线403；HTTP/3连接错误 | 基线未通过，协议比较不可判定 |
| HTTP/3校准 | Cloudflare HTTP/2和HTTP/3均200，实际协商版本均核实 | 排除诊断客户端完全不支持HTTP/3 |

本地超时组的“已完成字节/批次耗时”会遗漏部分已传字节，也包含慢请求尾部等待，
不能解释为持续 worker 的精确平均速度或带宽上限。Apple/ModelScope 的 range 与
完整流实验只比较相同前缀，SHA256 均一致；无 Range 返回200后只读8 MiB即关闭，
没有保存完整归档。上述结果仍是短时间工程探针，不代表长期额定速度。

ModelScope 403 只说明该 curl_cffi 请求未成功；此前 requests 已能读取对应公开
数据。没有据此把数据集标成私有，也没有将它归因于 NAS 文件权限。HTTP/3 信息
和强制版本语义见 [curl HTTP/3 文档](https://github.com/curl/curl/blob/master/docs/HTTP3.md)
与 [curl_cffi 高级用法](https://curl-cffi.readthedocs.io/en/latest/advanced.html)。

## 来源搜索与生产决策

新查原始配对候选最多约129对、11对的小抽样；其他候选只有 HDF5 或转换后的
动作/视频。没有找到经过速度和原始完整性核对、可替代全量七个官方 ZIP 的来源。
具体来源及 HF 存储区域选择边界见
[来源研究](2026-10-13-egodex-download-source-options.md)。

此前国内交大源256 MiB探针67.53 MB/s证明这台机器并非所有互联网下载统一受限
在7 MB/s；Apple、HF和ModelScope实际路径都慢，不能据此证明Apple故意限速。
继续增加并发、固定已测IP、换查询参数、换HTTP/2、本地SSH中转，均缺乏稳定
收益证据。更有价值的后续条件是实际不同公网出口，或真正托管原始完整数据的国内
镜像；更换主机、购买出口或建立新长期身份不在本轮探针内。

## 结束时生产快照

UTC时间戳1791860572，远端 acquisition 检查时间1791860552：

- Producer399615 RUNNING，八workers、32 MiB预取，提交f2ee8cea。
- part2完成90,978 / 95,125原始配对；官方全量共七个归档，当前仍在第一个part2。
- network_bytes319,116,239,619；包含前期慢下载和本轮隔离暂停的历史平均6.75 MB/s，
  不等于当前瞬时速度。raw_verified_bytes360,733,129,227含已复用数据，二者不可混用。
- 下载failed_packets121、prefetch_retries32与上一轮相同，不能把失败尝试数当唯一坏样本数。
- Consumer360908 RECONSTRUCTING，完成3,551条、接纳28,406窗口；processed_bytes
  144,780,151,674。历史failed17,479包含尝试/重试等记录，本轮没有增长；它仍须
  在恢复完整训练前做数据状态核对，不能仅凭持续运行宣称整个数据质量已验证。
- 停止的训练supervisor/torchrun/三rank PID均不存在，唯一GPU计算进程为consumer，
  占用约9,764 MiB；没有重启训练。

## 证据与复现入口

本地工程证据目录 `tmp/egodex-routes-20261013/`（不纳入Git）：

- `local-probe-1791859480/result.json`、`local-probe-1791859668/result.json`：本地路线。
- `ssh-throughput-1791859787/result.json`：单SSH。并发探针因TimeoutExpired退出，
  未保存完整聚合JSON；只保留诊断脚本和会话错误，不能据此计算吞吐。
- `remote_range_stream_and_status.json`：Range/完整流比较。
- `http3_and_final_status.json`：协议结果、生产快照、训练PID核对。
- 诊断脚本 `range_vs_stream.py`、`http3_probe.py`、`ssh_parallel.py`。

远端同一任务根目录：`download-range-vs-stream-1791859592/result.json`、
`download-http3-probe-1791860317/result.json`。后者最初读取历史owned receipt遇到
PermissionError，只对该确切诊断receipt恢复0644后重跑；NAS权限变化根因仍未确认，
没有对源数据或整个共享目录批量修改权限。证据不包含临时签名URL或代理秘密。
