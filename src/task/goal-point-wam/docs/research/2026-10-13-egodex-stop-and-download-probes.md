# 停止小样本EgoDex训练与下载吞吐诊断（2026-10-13）

## 用户指令与停止结果

用户明确要求停止当前三卡50k训练，因为冻结EgoDex训练样本过少；继续研究
下载加速。该指令撤销当前fit继续跑满50k的要求，不撤销全量原始下载和GPU3重建。
冻结303 TRAIN窗口/18sessions的r2训练已在31338更新停止，不将新的重建数据
临时混入这个run，不以该停止判断视频预训练方法无效。

确认supervisor378066的完整命令、工作目录与当前UID后发送SIGUSR1，由已有
launcher给三rank发送保存退出信号；supervisor与torchrun378458及
rank378520/378521/378522均消失。stage与progress均STOPPED。
通过训练环境CPU读取最新checkpoint，确认checkpoint_kind
`goal-point-wam.point-supervised.v1`与step31338；latest.pt大小629882244bytes。
三张训练卡释放，所有已有日志/验证记录保留。停止属于用户意图，不是训练报错。

远端run:
`/mnt/ugreen_nas/storage/oyx_storage/goal-point-wam-remote-20261012-r1/video-points-ddp50k-20261012-r2`

停止之前的decision receipt为`user-stop-1791856790.json`。GPU3消费者360908和
原始下载399615未被训练停止操作触碰。当前源代码固定f2ee8cea，下载仍为
八worker/32MiB物理连续Range，CRC/SHA及显式0644发布不变。

## 决策问题与诊断边界

Blocker/工程诊断：能否在同一远端、既有来源和权限下显著超过既有约7.45 MB/s
长时吞吐，从而缩短全量数据获取时间？廉价方法是隔离有限HTTP Range吞吐，
排除NAS写入/ZIP解压后依次改变并发、对象分流、DNS目的地址、查询参数、协议。
若重复改善且内容身份不变才替换生产传输；阴性则保留当前可用下载器，
不投入无收益的大并发改造、完整ZIP重复复制或新账号。

每组临时SIGSTOP已核验属于本任务的下载producer399615，测试正文流只入内存
或/dev/null，GPU3重建继续；独立看门狗按PID出生时间核验自动SIGCONT，
finally也恢复，原始7日截止与2.5TB raw/8TB processed/10TB NASfree边界不变。
测试暂停可能使已有HTTP连接触发超时/预取fallback，不能将这部分fallback
独立解释成网络质量恶化。最终必须核对producer恢复运行且完成数增长。

## 结果

| 同轮有限条件 | 聚合MB/s | 判断 |
| --- | ---: | --- |
| requests8/16/32/再测8；每连接8MiB | 3.55/6.83/7.24/6.03 | 32对16增益6.1%，8自身时变显著 |
| curl8连接固定17.253.61.133/.139 | 4.64/6.55 | 未支持永久固定IP |
| 总8连接分part2/3/4/5 | 6.32 | 未突破已有长时吞吐 |
| bare/query/query/bare；8连接 | 4.52/3.55/3.85/3.66 | 全区间SHA相同；缓存收益不稳定 |
| ECS返回87/95/5网段；4连接 | 2.54/3.15/4.63 | 同轮默认61网段6.60更好 |
| 同IP HTTP1.1/2/2/1.1；8连接 | 4.042/4.82/5.502/7.206 | 实际版本已核验，无重复大幅收益 |

requests并发和query对照拒绝非精确206与长度错误；ETag需同一对象保持，
query对照还验证相同区间SHA256一致。curl区域对照保存精确Content-Range、
ETag及实际地址。结果均为短时顺序诊断，不是稳定上线提速效果。

没有公网IPv6默认路由，不能仅因DNS有AAAA就启用IPv6。
HF与hf-mirror仓库tree API均401，未能取得真实分块清单；没有解释成确定的
账户权限原因。国内独立源对照均不可读/连不通，不能证明服务器总公网限速。

## 当前决策与后续

保留当前生产下载源、32MiB连续范围与八worker。当前证据指向传输链路的
吞吐限制/波动，而非GPU训练算力或bulk NAS写入；仍不能区分上游网络、
CDN针对本客户端策略与共享公网限制。没有发现重复可用的大幅提速方法，
不声称已经修复速度问题。

进一步能改变决策的证据是另一条真实可用、已测更快的网络路线或完整原始源；
连接数32、目前试过的DNS网段与跨ZIP分流已经不值得继续扩大。
官方来源与HTTP规范见[来源研究](2026-10-13-egodex-download-source-options.md)。

## 产物

远端证据根目录为
`/mnt/ugreen_nas/storage/oyx_storage/goal-point-wam-remote-20261012-r1/`，子目录：

- `download-parallel-probe-1791856873`
- `download-cdn-probe-1791856969`
- `download-query-probe-1791857135`
- `download-cdn-region-probe-1791857234`
- `download-domestic-control-1791857344`
- `download-http-probe-1791857469`

每个result.json使用原子发布与显式0644。本地聚合receipt为
[tmp/egodex-speed-20261013/remote_evidence.json](../../../../../../tmp/egodex-speed-20261013/remote_evidence.json)。
有限诊断脚本在同一tmp目录，仅作为throwaway harness，未改产品代码。

最终恢复核对：producer399615仍RUNNING，/proc状态S；两次状态快照
完成pairs从82226增至82332，约8.38 MB/s
短区间吞吐（状态30s更新且大Range批次计数，不能当稳定平均）。消费者360908
仍RECONSTRUCTING，累计3353 completed/26563 admitted windows；
GPU进程查询仅有360908，训练STOPPED/step31338。
最终快照见[final_status.json](../../../../../../tmp/egodex-speed-20261013/final_status.json)。
