# EgoDex来源与互联网下载瓶颈对照（2026-10-13）

## 问题与判别

用户要求先区分EgoDex来源慢与整个远端互联网下载都慢，再寻找更快原始来源。
这是Blocker/工程诊断，不涉及训练或科学claim。训练保持STOPPED/step31338；
全量原始下载producer399615与GPU3重建consumer360908继续。

已有最小复现是八连接Apple有限精确Range（约3–7 MB/s）。
本轮可证伪假设：如果是固定服务器全网下载上限，独立国内来源不应显著超过它；
如果只Apple有问题，其他多个国外来源应普遍明显更快；如果特定CDN/网络路径
有关，不同来源会呈现明显差异。对照不把HTTP403/连接失败计为有效低吞吐。

## 协议

同一服务器/requests环境、无代理、相同1或8线程、每请求新的Session，
精确206/Content-Range/长度检查；同一轮同一对象的ETag及size要一致。
正文只流入内存后丢弃，无磁盘写入、ZIP解压或模型推理。
每连接4MiB，8连接32MiB；后续HF/Apple每连接8MiB，国内长区间每连接32MiB。
部分连接重复同一范围，统计实际传输总字节而非不同成员覆盖量，
本测试不将任何非EgoDex内容存成数据集。不同payload长度的结果分别报告。

每轮仅短暂停已核验UID/cmdline/output/PID出生时间的本任务producer，
独立看门狗180s自动恢复，finally也SIGCONT；GPU3不暂停。
原始截止/存储授权不变。探针有连接、读超时及每请求25/50秒正文预算；
顺序测量与时变仍为限制，未声称正式因果识别某个CDN策略。

## 结果：不是所有互联网下载都慢

| 来源 | 1连接4MiB MB/s | 8连接32MiB MB/s |
| --- | ---: | ---: |
| Apple EgoDex首次 | 0.68 | 3.38 |
| PyPI NumPy wheel | 1.31 | 2.63 |
| Anaconda官方Miniconda文件 | 3.13 | 5.97 |
| 上海交大Ubuntu镜像 | 5.05 | 34.84 |
| Apple EgoDex末尾复测 | 3.11 | 6.62 |

后续同一顺序组：

| 来源 | 8连接总字节 | MB/s |
| --- | ---: | ---: |
| HF Wehere/EgoDex任务TAR | 64MiB | 3.38 |
| hf-mirror同一Wehere任务TAR | 64MiB | 4.76 |
| 上海交大较大区间对照 | 256MiB | 67.53 |
| Apple末尾同时间段复测 | 64MiB | 6.86 |

国内源较大区间268435456bytes/3.975s，非仅测首页、HEAD或首字节。
同轮Apple67108864bytes/9.785s。已排除“这台服务器所有互联网下载都只能
7–8 MB/s”的解释。实际可见至少约540Mbps短区间接收吞吐，不能当服务器
长期合同带宽，更不能承诺任何大文件一直达到这个速度。

PyPI/Anaconda也低于国内对照，说明不局限于Apple这个文件。
当前更支持数据源/CDN及其网络路径差异；国际出口、路径拥塞、站点客户端策略
仍未区分。国内平台域名不等于已证明数据正文一定沿更快的国内路径。

ModelScope补充：复用已解析的实际CDN URL（不打印签名参数），八连接32MiB
测得3.53 MB/s；随后上海交大八连接32MiB为25.63 MB/s。
正文hostname为`cdn-lfs-cn-1.modelscope.cn`，它也慢，因此不能简单把
国外域名本身当作根因，尚无理由承诺换到国内平台一定更快。

## EgoDex来源搜索与真实可读性

[来源索引](2026-10-13-egodex-download-source-options.md)记录原始一级资料。
本轮新找到HF `Wehere/EgoDex`，118任务TAR，当前文件树约578.05GB；
无需新账号，远端HF与hf-mirror精确Range均可读，最终均`us.aws.cdn.hf.co`。
因此hf-mirror在此只是入口镜像，未提供国内正文副本。

有上限读TAR前8个header，包含PAX header、目录以及
`add_remove_lid/actions/0_0.pt`（2381bytes）、`0_1.pt`（3981bytes）。
至少包含处理后的动作产物；是否同时保留原始配对、split、confidence等未知。
目前速度也更慢，因此不迁移既有原始下载/重建数据合同。

UCBProject/EgoDex公开例test/add_remove_lid/0.mp4精确Range成功，根目录
仅test；LeRobot/WDS候选属于需适配审查的转换内容，不作为全量原始替代。
先前zhenyuxie特定repo的401不能泛化成整个HF不可访问。

## 决策

继续保留官方producer，训练不恢复。当前不值得进一步仅扩大同一路线并发；
找真正不同托管/网络路径，并同时检查实际正文hostname、吞吐与原始数据完整性。
不因为发现快的Ubuntu镜像就假定那里托管EgoDex，也不因HF任务名称覆盖就
假定原始完整性。尚未找到可以直接上线且显著更快的全量原始替代源。

## 可追溯产物

远端根目录
`/mnt/ugreen_nas/storage/oyx_storage/goal-point-wam-remote-20261012-r1/`下：

- `download-independent-sources-1791858261/result.json`
- `download-hf-and-long-control-1791858364/result.json`
- `download-modelscope-control-1791858622/result.json`

本地[throughput_evidence.json](../../../../../../tmp/egodex-source-vs-network-20261013/throughput_evidence.json)
含实际请求尺寸、用时、错误和生产恢复状态。throwaway harness位于同一tmp目录，
不修改生产源码，不创建新branch或身份。

ModelScope对照及最终恢复状态见
[final_modelscope_and_status.json](../../../../../../tmp/egodex-source-vs-network-20261013/final_modelscope_and_status.json)。

生产恢复核对：下载PID399615仍RUNNING、
/proc状态S，completed 85745（前次84800）；
GPU3消费者360908仍RECONSTRUCTING，
completed 3432、admitted windows 27276。
未迁移来源或恢复训练。
