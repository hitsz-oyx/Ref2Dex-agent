# NAS 文件权限诊断（2026-10-13）

结论边界：文件权限是已复现的直接故障原因；**旧文件为什么出现000仍为
NOT_CONFIRMED**。当前证据支持继续查 NAS/NFS 创建和权限处理，也保留外部权限修改
的可能性。普通创建返回777与排他创建返回664，是可重复的差异；它本身不是000的复现。
不把社区案例、ctime/mtime或暂时没有活跃chmod进程当成根因证明。

## 本机观测

远端是Linux7.0.0-38-generic，客户端账户wbcd UID/GID1006；挂载是NFSv3、AUTH_SYS，
服务器为192.168.110.40。新建NAS文件显示UID1005/GID10，本地磁盘显示1006/1006。
现有本地身份可以读取NAS显式600文件，因此“UID不一致”单独不足以解释故障；
具体Squash规则没有NAS端配置证据。

旧HDF5的普通open实际报PermissionError。一个未修复的本任务Git alternates文件，
通过`statx(AT_STATX_FORCE_SYNC)`仍报告000且不可读；只执行chmod0644后，同一inode、
大小和mtime保持不变，立即可读。服务端刷新后仍看到000，说明至少这个样本不是
普通stat显示了陈旧零权限那么简单。没有捕获其最初创建时的NFS报文，也没有NAS
权限审计，因此不能确认是创建时零权限还是后来的修改。

同一进程、umask0002、open请求0666下的对照：

| 创建方式 | NAS强制刷新后的权限 | 本地磁盘权限 | 读取 |
| --- | --- | --- | --- |
| 普通O_CREAT/O_TRUNC | 777 | 664 | 成功 |
| O_CREAT/O_EXCL | 664 | 664 | 成功 |
| 普通创建后chmod0644 | 644 | 644 | 成功 |
| 普通创建后fchmod0644再关闭 | 644 | 644 | 成功 |

关闭后、5s及70s检查一致。昨天的普通创建、JSON、NPY、NPZ对照过约10h仍为777，
显式600/644/664样本也可读。新下载器最初一批的6个HDF5样本在8.85--9.39h后仍为
644且可读。无法支持“所有普通文件会在属性缓存过期后变000”的解释。

10种单变量操作（覆盖打开、ftruncate零/同大小/缩小/扩展、utime、rename、替换已有
目标、chmod后覆盖和重复覆盖）均未自然产生000。另在独立目录、旧原始文件实际
父目录与旧训练诊断目录下，以8线程创建45个range写入、JSON原子发布、NumPy和HDF5
样本，初次及70s服务端刷新均可读；36个777、9个显式644。这说明目前尚无稳定的
自然触发器，不能把普通创建、重命名、文件类型或并发单独判为故障原因。

通过NFS读取所查父目录和文件的`system.posix_acl_access/default`均返回ENODATA；
这不证明NAS没有其他权限系统或私有ACL。相关生产代码未见主动chmod000、异常umask、
ZIP外部权限恢复或chown操作。历史根因仍有NAS策略/固件、NFS客户端与服务端交互、
其他客户端或进程修改等未区分候选。

## 规范和外部材料

[NFSv3 RFC1813 CREATE](https://www.rfc-editor.org/rfc/rfc1813.html#section-3.3.8)
规定EXCLUSIVE创建后由客户端SETATTR设置正确属性。
[Linux7.0 nfs3proc.c](https://raw.githubusercontent.com/torvalds/linux/v7.0/fs/nfs/nfs3proc.c)
也明确在该路径做post-create SETATTR。因此排他创建和显式chmod都有额外设置属性
这一步，能解释两类路径的差异来源，但尚无本机报文证明哪个CREATE处理出了问题。

[statx手册](https://www.man7.org/linux/man-pages/man2/statx.2.html)
说明FORCE_SYNC强制同步服务端属性；诊断不要求关闭全局NFS缓存或重新挂载。
[绿联官方文件权限文档](https://support.ugnas.com/detail/article/zh-CN/142)
说明AUTH_SYS的UID/GID与Squash映射影响访问；本机具体规则仍未读取。

[UGOS1.16.0.0089的一手社区故障报告](https://www.reddit.com/r/UgreenNASync/comments/1tyc7c2/bug_report_diagnosis_ugos_pro_11600089_silently/)
提供私有ugacl与POSIX权限不一致的排查线索。它主要影响共享根目录，本机主要是
大量文件；NAS固件版本也未知，不能认定是同一bug，不能据此修改或复制NAS私有ACL。

下一项能区分根因的外部证据是NAS端固件/升级时间、实际NFS导出和Squash规则、
受影响路径的服务端POSIX/ugacl属性、权限变更审计。当前身份不能读取这些NAS端资料，
客户端kernel journal也无读取权限。用户不清楚近期是否改过NAS规则。未使用sudo、
安装系统包、修改NAS配置、换用户身份或重新挂载。

## 已落实的恢复与复现

新下载器在原始payload和JSON receipt原子发布前显式chmod0644。旧批恢复不再只靠
一次stat筛选000：对10792对源文件和receipt共32376个文件显式chmod0644，逐对做
小量读取验证，全部成功。9540个PermissionError失败receipt保留后重新入队；重试
排队不等于已重建，历史失败计数保留。NAS起因仍未知，新链路长期样本正常不能替代
旧失败补跑。现有三卡fit、GPU3消费者和固定训练输入均未重启或改动。

可复用CPU诊断入口：[probe_nas_permissions.py](../../tools/audit/probe_nas_permissions.py)。
只创建新的诊断目录，不改已有文件；对真实访问失败返回非零。示例（在远端执行，两个
目录必须尚不存在，其父目录必须已有；结果留在各自项目工作区）：

```bash
python probe_nas_permissions.py --nas-output /path/to/owned/nas/new-probe --local-output /path/to/owned/project/tmp/new-probe --observe-seconds 70
```

远端证据根目录：`/mnt/ugreen_nas/storage/oyx_storage/goal-point-wam-remote-20261012-r1/`。
相关子目录：

- `permission-probe-1791816960/`：初始syscall及隔夜对照。
- `permission-statx-probe-1791853351/`：本地/NAS、普通/排他/显式模式强制刷新对照。
- `permission-metadata-probe-1791853584/`：单变量截断、时间和重命名对照。
- `permission-same-parent-1791853782/`：45文件、同父目录与并发对照。
- `permission-mode-repair-control-1791853850/`：真实故障文件000→644的控制。
- `full-old-raw-recovery-1791852682/result.json`：旧批全量显式恢复与重试receipt。

`permission-old-force-1791853457/`采样时被恢复任务抢先修复，不能用它推断未修复旧
HDF5在强制刷新后的权限。无重复全量hash或NAS全树扫描。

复用工具在远端`permission-replay-1791854209/`进行了零等待工程复核：10个本地/NAS
样本均可读，创建方式差异仍一致；CLI和py_compile检查通过。45个同父目录诊断样本
已归档至`permission-same-parent-1791853782/archived-samples-*`；读取历史结果中的旧
路径时按该目录的`relocations.json`映射，原始数据与训练目录不保留这些假载荷。
