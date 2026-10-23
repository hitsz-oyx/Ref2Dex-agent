"""Human report for the complete prepared catalog and its immutable partitions."""
import argparse
from collections import Counter
import csv
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import prepare_full_rigid_data as p
from publish_full_rigid_report import table,coverage_plot


def read_cov(path):
    with Path(path).open() as f:return {(r['phase'],r['geometry_class'],int(r['source_id'])):int(r['windows']) for r in csv.DictReader(f)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['catalog','historical-report','output']:parser.add_argument('--'+key,type=Path,required=True)
    a=parser.parse_args();catalog=a.catalog.resolve();out=a.output.resolve();m=p.read(catalog/'manifest.json')
    if m['status']!='COMPLETED' or out.parent!=(p.TASK/'docs/audit').resolve():raise ValueError('completed task-owned catalog/report required')
    historical=Path(m['historical_manifest']['path']).parent;h=p.read(historical/'manifest.json');run=historical.parent
    support=catalog/'report-support';support.mkdir(exist_ok=False)
    old=read_cov(historical/'coverage-old-6000.csv');before=read_cov(catalog/'coverage-train-before.csv');after=read_cov(catalog/'coverage-train-after.csv')
    oldzero=sum(v==0 for (q,g,s),v in old.items() if q!='unknown' and g!='unknown')
    zero=m['coverage']['train']['after']['zero_requested_cells']
    coverage_plot([old,before,after],['TRAIN old6000','Prepared full before dedup','Prepared full after dedup'],support/'train-coverage.png')
    coverage_plot([read_cov(historical/'coverage-old-dev.csv'),read_cov(catalog/'coverage-dev-before.csv'),read_cov(catalog/'coverage-dev-after.csv')],
                  ['DEV old3356','Full DEV before dedup','Full DEV after dedup'],support/'dev-coverage.png')
    rel='../../../../../'+str(catalog.relative_to(p.ROOT));link=lambda file,label=None:'['+(label or file)+']('+rel+'/'+file+')'
    gaps=[]
    for split in ['train','dev']:
        split_before=read_cov(catalog/('coverage-'+split+'-before.csv'))
        split_after=read_cov(catalog/('coverage-'+split+'-after.csv'))
        for (phase,gc,sid),n in split_after.items():
            if phase=='unknown' or gc=='unknown' or n:continue
            if split_before[(phase,gc,sid)]:reason='removed_as_near_duplicate_of_actual_retained_content'
            elif not sum(v for (q,g,s),v in split_before.items() if s==sid):reason='source_absent_from_split'
            elif not sum(v for (q,g,s),v in split_before.items() if g==gc and s==sid):reason='geometry_class_absent_in_admitted_source'
            else:reason='phase_not_observed_in_admitted_source_geometry'
            gaps.append(dict(split=split,phase=phase,geometry_class=gc,source_id=sid,windows=0,reason=reason))
    with (support/'zero-gaps.csv').open('x') as f:
        writer=csv.DictWriter(f,fieldnames=['split','phase','geometry_class','source_id','windows','reason']);writer.writeheader();writer.writerows(gaps)
    names={0:'OakInk2',1:'GRAB',2:'ARCTIC rigid slots',4:'Shadow replay',5:'TACO',6:'HOT3D',8:'DexYCB',9:'H2O',10:'HOI4D'}
    sources=[];old_sources=Counter()
    for (q,g,s),n in old.items():old_sources[str(s)]+=n
    for sid in p.SOURCES:
        key=str(sid);sources.append([str(sid)+' '+names[sid],old_sources[key],m['source_distribution']['train']['before'].get(key,0),m['source_distribution']['train']['after'].get(key,0),
            m['source_distribution']['train']['after'].get(key,0)-old_sources[key],m['source_distribution']['dev']['before'].get(key,0),m['source_distribution']['dev']['after'].get(key,0)])
    dist=[]
    for gc in p.CLASSES:
        oldn=sum(n for (q,g,s),n in old.items() if g==gc)
        dist.append([gc,oldn,m['geometry_distribution']['train']['before'].get(gc,0),m['geometry_distribution']['train']['after'].get(gc,0),m['geometry_distribution']['dev']['after'].get(gc,0)])
    phase_rows=[]
    for phase in p.PHASES:
        oldn=sum(n for (q,g,s),n in old.items() if q==phase)
        phase_rows.append([phase,oldn,m['phase_distribution']['train']['before'].get(phase,0),m['phase_distribution']['train']['after'].get(phase,0),m['phase_distribution']['dev']['after'].get(phase,0)])
    sha=p.digest(catalog/'manifest.json');free=p.shutil.disk_usage(catalog).free;f=m['fresh'];pr=m['provenance']
    text=f'''# 全量数据准备总报告 — brief47 / 2026-10-23

{table(['观测','含义','决定'],[
[f'108格中TRAIN零格：旧6k {oldzero} → 全量准入 {m["coverage"]["train"]["before"]["zero_requested_cells"]} → 去重后 {zero}','原冻结九源全量恢复后，覆盖缺口显著减少（报具体格数，不推断模型收益）','可进入全量数据重训/续训排期；本轮0updates，不自动启动'],
[f'仍有{zero}个0格；source8 DexYCB DEV=0','本地准入数据仍存在特定来源/几何/相位缺口','逐格列清，由机主决定补采集/补标注/开发集设计，不用source4冒充其他源'],
[f'当前磁盘{free/2**30:.3f}GiB；硬停止线30GiB','本轮保留原产物、分段完成并满足存储边界','没有删历史outputs；触线先停止，不自行清理'],
[f'goal认证 {pr["certified_goal_before"]:,}；缺证 {pr["uncertifiable_before"]:,}','历史池UNKNOWN与新已认证分区分别统计','缺证不算非法；不以目标label哈希/mtime冒充reset前证明']])}

**全量定义与边界。** 原冻结39526条九源计划、1335条本地legacy母序列和旧6k中的1064条
active TRAIN母序列按真实原始payload URI去重计数，不能把这些相交集合直接相加。
原计划外唯一非TEST legacy母序列用ID39526补齐，历史主池共39527条完成准入判定；
202条原协议TEST母序列逐条排除进入TRAIN/DEV，不作质量非法判定。另有633条派生view
（含Inspire与qualitative MANO）逐条列拒绝理由，不增加母序列数、不借human标签给robot。

并行任务刚完成的96条新刚体记录已经只读独立复核，作为独立的**已认证TRAIN候选分区**
纳入准备总目录。因此本次准备母序列合计 **{m['mother_decisions']:,}**，加202条TEST形成
**{m['total_mother_census']:,}**条母序列审计全集。旧6k、原512锚点、两个原分区manifest、
生产TRAIN、模型、normalizer和训练配置均未修改；新总目录`training_allowed=false`。
本轮只做存量转换、复核与编目，没有新采集、物理回放或训练；96条的采集属于独立
P-20261023-rigid-source-bound-pilot，其原始记录与证书保留原目录。

## 全量manifest与分布

{table(['split','准入窗口前','去重后','删除'],[[s,m['windows_before'][s],m['windows_after'][s],m['windows_before'][s]-m['windows_after'][s]] for s in ['train','dev']])}

每行均为`(sequence_id,slot,tick)`。历史ID0..39526不重排；新96条为39527..39622，
通过原candidate-manifest的`sequence_ordinal`映射，**不误用plan.jobs顺序**。
原pending6000序列已全部尝试且完成转换准入，不是只准备6000窗口；精确对账与逐条
receipt/拒绝原因在[历史主池报告]({a.historical_report.name})。

{table(['source','旧6k TRAIN','全量TRAIN前','全量TRAIN后','vs6k净增','DEV前','DEV后'],sources)}

{table(['几何类','旧6k TRAIN','全量TRAIN前','全量TRAIN后','DEV后'],dist)}

{table(['phase','旧6k TRAIN','全量TRAIN前','全量TRAIN后','DEV后'],phase_rows)}

几何类别由冻结PCA extent规则决定，非物体名称；phase是原proximity协议，不是实测force或
指令性接触。approach包括“已知clear且K24内没有CONTACT”的状态，不保证有定向靠近动作。
source0 dense slot扩展含背景物体样本，窗口量不能直接当成独立有效交互数量。
仅保留刚体SE3/已有独立slot表示，不增加关节、可变形或关系型目标字段。

## 去重与覆盖缺口

物理谓词沿原规则：形状extent5%、完整28帧anchored joint21/query32坐标2mm、物体旋转1度、
相同phase/validity和逐帧16part接触比例2%；actual retained representative，不用传递闭包。
跨母序列不按物体名称或source身份设阈值。候选为同序列24tick内最后保留代表，或跨序列
相同固定bin内容签名；跨bin和不同canonical CAD朝向会保留部分近邻，不宣称穷尽所有近重复。
旧61.23%是“同sequence/slot起点相隔<24帧”的时间重叠率，历史全量前/后为
{h['dedup']['temporal_near_repeat_percent_before']:.6f}%/{h['dedup']['temporal_near_repeat_percent_after']:.6f}%，
不与内容删除率混称同一指标。

历史池删除 **{m['dedup']['historical_removed']:,}**；新已认证分区
**{f['before']:,}→{f['after']:,}**，删除 **{f['removed']:,}**。
对于整个准备目录，还要求真实模型输入的goal-validity mask一致；有效goal的32个delta也须在2mm内。
未知条件和独立已知任务指令是不同输入，不能因物理状态相近而把合法指令样本并入unknown。
这个规则与物体身份无关，不改物理阈值。跨分区物理签名匹配并通过完整物理谓词的窗口
**{m['dedup']['cross_partition_physical_matches']:,}**，条件完全相同数
**{m['dedup']['cross_partition_condition_identical']:,}**；两种计数分别保存，不伪装成无重复。

完整108格（另列unknown）在{link('coverage-train-before.csv','TRAIN前')}、
{link('coverage-train-after.csv','TRAIN后')}、{link('coverage-dev-before.csv','DEV前')}、
{link('coverage-dev-after.csv','DEV后')}。可视化：{link('report-support/train-coverage.png','TRAIN热图')}、
{link('report-support/dev-coverage.png','DEV热图')}。
{link('report-support/zero-gaps.csv','全部0格及原因')}逐项列出；当前零格原因
`{dict(Counter(r['reason'] for r in gaps))}`。新source4记录不能填source8/10的空格。
后续补数据要按更新后的全量缺口决定，不能仅沿旧6k的source4缺口自动扩采。

TRAIN剩余0格逐项如下；DEV有{m['coverage']['dev']['after']['zero_requested_cells']}个0格，
也逐项包含在上述带split字段的CSV中。source8 DEV整源缺失单列为split缺口，不假装源数据没有这些相位。

{table(['source','phase','几何类','窗口数','原因'],[[str(r['source_id'])+' '+names[r['source_id']],r['phase'],r['geometry_class'],0,r['reason']] for r in gaps if r['split']=='train'])}

## 来源认证、哈希和消费合同

历史池去重前 **{pr['uncertifiable_before']:,}**条窗口保持`source_role=UNKNOWN`、
`available_before_reset=null`、reset/source时间和独立source hash为空；label pose SHA与
filesystem mtime仅作回溯。完整native字段census还单列goal/动作标签候选，不能泛称原始数据
没有goal，也不能由reset后录制的goal快照自动给完整TaskSpec/来源时序发证。

新分区去重前 **{pr['certified_goal_before']:,}**、去重后 **{pr['certified_goal_after']:,}**条有
`source_role=explicit_task_instruction`、`available_before_reset=true`、实际同clock纳秒时间和
intent-artifact SHA。重验全部96条保存窗口、reset前source时间/hash、每次原生reset恢复、
实际21点与保存21点、778表面/16part、接触重算抽帧、全部event重算和原生60→30Hz时钟，
结果PASS；没有凭另一个报告数字直接发证。

所有准入窗口的角色/时间/hash在总manifest的`window_manifest_parts`中逐行落盘；新分区
{link('fresh-windows.jsonl','逐窗证据')}和{link('fresh-sequences.jsonl','原始ordinal/来源映射')}可独立重读。
goal认证**不替event意图发证**：event条件认证仍0。
消费旧池必须用显式UNKNOWN goal，不能回退自身t+24 endpoint；消费新分区必须调用现有
`fresh_rigid.reset_bound_sample`并核对原instruction/reset。两者event条件都保持UNKNOWN。
这些是新目录的必要读取合同，不激活任何生产训练入口。
未来若同时改变数据量与启用新合法goal，必须另列条件对照，不能把收益仅归因数据量。

## 与旧版本逐项差异与复现

历史主池旧6000 TRAIN行全部保留在全量去重前；新分区全为新增ID。
完整TRAIN旧行membership mask与前/后新增tuples已在总目录重导出：
{link('train-membership-in-old6000.npy','membership')}、
{link('train-added-vs6000-before.npy','新增前')}、{link('train-added-vs6000-after.npy','新增后')}。
旧版本DEV为3356行，其逐项缺失/保留/新增与source/phase/几何差异在历史主池manifest的
`versus_6000.dev_*`，新分区不改变DEV。全量索引
{link('train-all.npy','TRAIN前')}、{link('train-deduplicated.npy','TRAIN后')}、
{link('dev-all.npy','DEV前')}、{link('dev-deduplicated.npy','DEV后')}可直接按ID映射重读。

总manifest **SHA256 `{sha}`**：{link('manifest.json','manifest')}，{link('manifest.sha256','SHA文件')}。
其`files`记录每个索引/账本的SHA和字节；两分区manifest路径与SHA也固定在内。
所有转换命令、按批receipt、预算v1–v7、环境与随机种子见历史主池报告和run目录；
新分区Probe seed198–203在原计划/每批simulation-config中，本轮编目不抽新seed。
已知原6k/512面板不重抽；未做预测性能验证或训练。
固定转换182331193978 B已逐receipt与实际文件求和一致，旧96.5GiB仅是6000条sequence批次估算。
目录/inode、实际file blocks、当前storage stop和详细预算见九源独立readiness报告；
任何显示GiB字段与bytes不一致时按bytes重算，历史预检不覆盖。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \\
 /home2/wyy/miniconda3/envs/graspenv/bin/python \\
 src/task/ptv3-shared-wam-v6/tools/audit/build_full_prepared_catalog.py \\
 --historical {historical.relative_to(p.ROOT)} \\
 --fresh outputs/ptv3-shared-wam-v6/rigid-source-bound-pilot-20261023-r2 \\
 --check {Path(m['independent_fresh_check']['path']).relative_to(p.ROOT)} \\
 --output {catalog.relative_to(p.ROOT)}
# 重跑必须使用新目录；所有旧产物保持不变。
```
'''
    out.write_text(text)
    p.write(support/'report-receipt.json',dict(manifest_sha256=sha,report_sha256=p.digest(out),zero_gaps=gaps,
        free_bytes_at_report=free,git_commit=p.subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),training_updates=0))
    print(dict(report=str(out),manifest_sha256=sha,zero_cells=zero,certified_goal_before=pr['certified_goal_before']))


if __name__=='__main__':main()
