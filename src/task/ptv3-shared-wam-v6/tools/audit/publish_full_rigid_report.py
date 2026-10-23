"""Render the completed full preparation report from immutable numeric artifacts."""
import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import prepare_full_rigid_data as p


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,row))+' |' for row in rows])



def coverage_plot(matrices,labels,path):
    os.environ.setdefault('MPLCONFIGDIR',str(p.ROOT/'tmp/matplotlib-full-rigid'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    phases=['approach','close','contact','hold'];classes=['flat','elongated','compact']
    rows=[(q,g) for q in phases for g in classes]
    data=[np.array([[m[(q,g,s)] for s in p.SOURCES] for q,g in rows]) for m in matrices]
    cmap=plt.get_cmap('cividis').copy();cmap.set_bad('#eeeeee')
    vmax=max(1,float(np.log10(max(int(d.max()) for d in data)+1)))
    fig,axes=plt.subplots(1,3,figsize=(18,7),constrained_layout=True)
    for ax,d,label in zip(axes,data,labels):
        values=np.ma.masked_where(d==0,np.log10(d+1));im=ax.imshow(values,aspect='auto',cmap=cmap,vmin=0,vmax=vmax)
        ax.set_title(label);ax.set_xticks(range(9));ax.set_xticklabels(p.SOURCES)
        ax.set_yticks(range(12));ax.set_yticklabels([q+' / '+g for q,g in rows],fontsize=8)
        ax.set_xlabel('source_id')
        for i in range(12):
            for j in range(9):
                n=int(d[i,j]);t=(f'{n/1e6:.2f}M' if n>=1e6 else f'{n/1000:.1f}k' if n>=1000 else str(n))
                color='black' if n==0 or np.log10(n+1)>vmax*.6 else 'white'
                ax.text(j,i,t,ha='center',va='center',fontsize=7,color=color)
    fig.colorbar(im,ax=axes,label='log10(window count + 1)',shrink=.8)
    fig.suptitle('Gray = zero coverage; unknown phase/geometry reported separately',fontsize=11)
    fig.savefig(path,dpi=170);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--summary',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    run=a.run.resolve();summary=a.summary.resolve();out=a.output.resolve()
    if out.parent!=(p.TASK/'docs/audit').resolve():raise ValueError('task-owned report required')
    m=p.read(summary/'manifest.json');status=p.read(summary/'progress.json')
    if status['status']!='COMPLETED':raise ValueError('completed full manifest required')
    budget=p.read(run/'storage-budget-v7.json');legacy=m['legacy'];diff=m['versus_6000'];environment=p.read(run/'environment-record.json')
    source_names={0:'OakInk2',1:'GRAB',2:'ARCTIC rigid slots',4:'Shadow replay',5:'TACO',6:'HOT3D',8:'DexYCB',9:'H2O',10:'HOI4D'}
    def load_cov(path):
        with path.open() as f:return {(r['phase'],r['geometry_class'],int(r['source_id'])):int(r['windows']) for r in csv.DictReader(f)}
    old=load_cov(summary/'coverage-old-6000.csv');before=load_cov(summary/'coverage-train-before.csv');after=load_cov(summary/'coverage-train-after.csv')
    old_zeros=sum(v==0 for (phase,gc,sid),v in old.items() if phase!='unknown' and gc!='unknown')
    bzero=m['coverage']['train']['before']['zero_requested_cells'];azero=m['coverage']['train']['after']['zero_requested_cells']
    support=run/'report-support';support.mkdir(exist_ok=False)
    gaps=[]
    for (phase,gc,sid),count in after.items():
        if phase=='unknown' or gc=='unknown' or count:continue
        if before[(phase,gc,sid)]:reason='removed_as_near_duplicate_of_retained_content'
        elif not sum(v for (q,g,s),v in before.items() if g==gc and s==sid):reason='geometry_class_absent_from_admitted_source'
        elif not sum(v for (q,g,s),v in before.items() if g==gc and s==sid and q!='unknown'):reason='phase_annotations_all_unknown_for_source_geometry'
        else:reason='phase_not_observed_for_source_geometry'
        gaps.append(dict(phase=phase,geometry_class=gc,source_id=sid,windows_after=0,windows_before=before[(phase,gc,sid)],reason=reason))
    with (support/'coverage-gaps-explained.csv').open('x') as f:
        w=csv.DictWriter(f,fieldnames=['phase','geometry_class','source_id','windows_after','windows_before','reason']);w.writeheader();w.writerows(gaps)
    coverage_plot([old,before,after],['TRAIN old 6000','TRAIN full before dedup','TRAIN full after dedup'],support/'coverage-train.png')
    dev_matrices=[load_cov(summary/name) for name in ['coverage-old-dev.csv','coverage-dev-before.csv','coverage-dev-after.csv']]
    coverage_plot(dev_matrices,['DEV old version','DEV full before dedup','DEV full after dedup'],support/'coverage-dev.png')
    minimum=min(m['disk_free_after_bytes'],p.shutil.disk_usage(run).free)
    receipts=[p.read_receipt(path) for path in sorted((run/'receipts').glob('*.json'))]
    minimum=min([minimum]+[r['disk_free_bytes'] for r in receipts if 'disk_free_bytes' in r])
    # Include intermediate journal/derived-index peaks, not just final free
    # space after SQLite has closed its transaction journal.
    for log in run.glob('full-manifest-*.log'):
        for line in log.open():
            try: observation=json.loads(line)
            except (ValueError,TypeError): continue
            if isinstance(observation,dict) and 'disk_free_bytes' in observation:
                minimum=min(minimum,observation['disk_free_bytes'])
    counts=Counter(r['split'] for r in receipts if r['status']=='ADMITTED')
    sha=p.digest(summary/'manifest.json');rel='../../../../../'+str(run.relative_to(p.ROOT))
    link=lambda name,label=None:'['+(label or name)+']('+rel+'/'+name+')'
    source_rows=[]
    source_delta={int(r['category']):r for r in diff['source_delta']}
    for sid in p.SOURCES:
        d=source_delta[sid];st=m['sequence_status_by_source'][str(sid)]
        source_rows.append([str(sid)+' '+source_names[sid],st.get('ADMITTED',0),st.get('REJECTED',0),d['old6000'],d['full_before'],d['full_after'],d['difference_after'],m['source_distribution']['dev']['before'].get(str(sid),0),m['source_distribution']['dev']['after'].get(str(sid),0)])
    class_rows=[[r['category'],r['old6000'],r['full_before'],r['full_after'],r['difference_after'],m['geometry_distribution']['dev']['before'].get(r['category'],0),m['geometry_distribution']['dev']['after'].get(r['category'],0)] for r in diff['geometry_delta']]
    dev_source_rows=[[r['category'],r['old6000'],r['full_before'],r['full_after'],r['difference_after']] for r in diff['dev_source_delta']]
    dev_geometry_rows=[[r['category'],r['old6000'],r['full_before'],r['full_after'],r['difference_after']] for r in diff['dev_geometry_delta']]
    dev_phase_rows=[[r['category'],r['old6000'],r['full_before'],r['full_after'],r['difference_after']] for r in diff['dev_phase_delta']]
    phase_rows=[[r['category'],r['old6000'],r['full_before'],r['full_after'],r['difference_after']] for r in diff['phase_delta']]
    opening=table(['观测','含义','决定'],[
        [f'全量TRAIN准入后的0覆盖格：旧6k {old_zeros}/108 → 全量{bzero}/108；去重后{azero}/108','按固定刚体合同测得的数据覆盖变化','可纳入全量数据重训/续训排期；本轮不训练，条件合同门槛保留'],
        [f'去重后仍有{azero}个0格，原因逐格列出','区分几何未覆盖、相位未观测、相位标签unknown和去重影响','由机主决定补采集或补标注；不按物体身份放宽准入'],
        [f'最低记录磁盘余量{minimum/2**30:.3f}GiB；硬停止线30GiB','本轮存储边界是否持续满足可直接核对','低于预测安全线即停；本轮未删除历史outputs'],
        [f'可认证独立条件窗口0/{sum(m["windows_before"].values()):,}','有目标标签不等于reset前独立指令证据','UNKNOWN/null逐窗保留；不能把缺证算非法或伪造事前可得']])
    text=f'''# 全量刚体数据准备 — brief47 / 2026-10-23

{opening}

本报告中的“全量”包含全部 **1,335条本地legacy母序列**、旧6k覆盖的
**{diff['baseline_mothers']:,}条active TRAIN母序列**所在的**1,177条缓存**，以及九source
冻结计划的**{m['frozen_recipe_mothers']:,}条母序列**。同母序列不重复相加。母序列按source_id+原始完整payload URI计数；source4同名相对路径位于
不同采集根，不能只按basename合并。完整URI逐条见mother-identities.jsonl。
唯一原计划外的本地非TEST母序列用append-only ID39526补齐，实际全量准入判定
**{m['total_prepared_mother_decisions']:,}条TRAIN/DEV母序列**；另有202条官方TEST母序列
逐条排除进入TRAIN/DEV；这是保留原协议TEST划分，不是按物体身份新设质量阈值或判数据非法。合并去重后的母序列审计全集为39,729条，派生view不增加母序列数（含Inspire与qualitative MANO）。

没有6,000窗口上限，也没有source/phase配额；对已有刚体slot逐tick使用同一H4/K24
合同，packed source保留原target_slot，source0/1/2检查已注册slot。没有训练、换模型、
新增关节/可变形/关系型目标字段或删除历史outputs。ARCTIC只沿用独立刚体pose slot，
不将多部件整体称为单刚体。

## 全量序列与窗口

母序列准入状态：`{json.dumps(m['sequence_status'],ensure_ascii=False)}`。
原pending **6,000条序列批次**：`{json.dumps(m['pending6000'],ensure_ascii=False)}`。
legacy1335状态：`{json.dumps(legacy['mother_status'])}`；两根派生view共
`{sum(legacy['views_by_root'].values())}`条，全部独立列出原因，不借用human parent标签。
完整判定在{link('full-manifest-r1/sequence-verdicts.jsonl','序列判定')}、
{link('full-manifest-r1/legacy-mother-verdicts.jsonl','legacy母序列')}和
{link('full-manifest-r1/legacy-view-verdicts.jsonl','legacy view')}。

{table(['split','准入窗口（去重前）','去重后','删除'],[[s,m['windows_before'][s],m['windows_after'][s],m['windows_before'][s]-m['windows_after'][s]] for s in ['train','dev']])}

每个索引行都是`(sequence_id, slot, tick)`；原39526个ID沿用冻结计划顺序，追加ID不重排
旧ID。旧6k使用的缓存ID通过稳定`sequence_key/slot/tick`精确join。
全量TRAIN母序列去重前/后为 **{diff['train_mothers_before']:,}/{diff['train_mothers_after']:,}**。

{table(['source','准入母序列','拒绝母序列','旧6k TRAIN','全量TRAIN前','全量TRAIN后','vs6k净增','DEV前','DEV后'],source_rows)}

几何类别来自固定CAD surface的PCA extents：flat=min/mid≤0.2，elongated=mid/max≤0.4，
其余compact；无有效厚度为unknown，不按物体名称分类或设准入阈值。

{table(['几何类别','旧6k TRAIN','全量TRAIN前','全量TRAIN后','vs6k净增','DEV前','DEV后'],class_rows)}

{table(['phase','旧6k TRAIN','全量TRAIN前','全量TRAIN后','vs6k净增'],phase_rows)}

phase仍是原冻结proximity事件协议，不是实测force contact或独立任务意图。
旧6k版本的DEV是原有 **{diff['dev_baseline_windows']:,}** 行，不把“6k TRAIN”误记成“6k DEV”。

{table(['DEV source','旧版本DEV','全量DEV前','全量DEV后','净增'],dev_source_rows)}

{table(['DEV几何','旧版本DEV','全量DEV前','全量DEV后','净增'],dev_geometry_rows)}

{table(['DEV phase','旧版本DEV','全量DEV前','全量DEV后','净增'],dev_phase_rows)}

## 去重、覆盖与逐项差异

去重前 **{m['dedup']['before']:,}**，去重后 **{m['dedup']['after']:,}**，删除
**{m['dedup']['before']-m['dedup']['after']:,}**。规则：{m['dedup']['rule']}。
每个删除项直接满足与实际保留代表的完整谓词，不用传递闭包。跨TRAIN/DEV重复优先保留DEV，
TRAIN跨集删除 **{m['dedup']['cross_split_train_removed']:,}**。
不按object/source身份限制是否可成为重复。保守fixed-bin会保留跨bin边界的近邻；不同CAD的
canonical朝向可能漏检，不宣称穷尽所有近重复或CAD等价。

旧6k的时间起点近重复比例为61.23%；全量同口径去重前/后为
**{m['dedup']['temporal_near_repeat_percent_before']:.6f}%/{m['dedup']['temporal_near_repeat_percent_after']:.6f}%**。
它统计同sequence/slot起点相隔<24帧，和本次内容去重删除率是不同指标；密集tick增加会改变
时间近重复比例，不能把两者混为“无重复”。

覆盖矩阵为4phase×3数值几何类×9source=108格，另保留unknown行/列。
TRAIN/DEV各有去重前后全表：{link('full-manifest-r1/coverage-train-before.csv','TRAIN前')}、
{link('full-manifest-r1/coverage-train-after.csv','TRAIN后')}、
{link('full-manifest-r1/coverage-dev-before.csv','DEV前')}、
{link('full-manifest-r1/coverage-dev-after.csv','DEV后')}。
{link('report-support/coverage-gaps-explained.csv','零格逐项原因')}：
`{json.dumps(dict(Counter(g['reason'] for g in gaps)))}`。
可视化：{link('report-support/coverage-train.png','TRAIN覆盖热图')}、{link('report-support/coverage-dev.png','DEV覆盖热图')}。
“0”表示当前合同下没有准入样本；原本有样本但被内容去重移到其他保留代表的格子
不算原始数据结构性缺失。仅有unknown相位的格子先列标注缺口，不伪装成相位阳性。

旧6k的 **{diff['baseline_rows_present_before_dedup']:,}/{diff['baseline_windows']:,}**行仍在
全量去重前，去重后保留 **{diff['baseline_rows_present_after_dedup']:,}**行。
未包含的旧行逐项写在manifest的`versus_6000.baseline_rows_missing`，不是静默丢弃。
新增TRAIN行数前/后为 **{diff['train_added_before_dedup']:,}/{diff['train_added_after_dedup']:,}**；
{link('full-manifest-r1/train-added-vs6000-before.npy','新增tuples前')}和
{link('full-manifest-r1/train-added-vs6000-after.npy','新增tuples后')}给出全部逐项差集；
{link('full-manifest-r1/train-membership-in-old6000.npy','旧6k membership mask')}、
{link('full-manifest-r1/dedup-pairs.jsonl','删除代表逐项账本')}可回溯每一行。
DEV旧行去重前/后保留 **{diff['dev_baseline_rows_present_before_dedup']:,}/{diff['dev_baseline_rows_present_after_dedup']:,}**；
DEV新增前/后为 **{diff['dev_added_before_dedup']:,}/{diff['dev_added_after_dedup']:,}**。
对应{link('full-manifest-r1/dev-added-vs-old-version-before.npy','DEV新增tuples前')}、
{link('full-manifest-r1/dev-added-vs-old-version-after.npy','DEV新增tuples后')}、
{link('full-manifest-r1/dev-membership-in-old-version.npy','DEV旧行mask')}均已导出；
未纳入的旧DEV行在`versus_6000.dev_baseline_rows_missing`逐项列出。
旧curated manifest和固定512 TRAIN锚点保持原样。没有重新拟合normalizer或修改训练配方。全量准入窗口另逐条核对30Hz时钟/segment，
不允许H4/K24跨不连续边界；source0全部568条frame_id核对0断点。

## 来源认证与实际缺口

每个去重前准入窗口在TRAIN/DEV JSONL中都有`source_role=UNKNOWN`、
`available_before_reset=null`、reset/source时间戳null、独立source hash null，以及当前目标
label pose的SHA256和filesystem mtime。可认证 **0**，不可认证
**{m['provenance']['uncertifiable']:,}**；去重后不可认证 **{sum(m['windows_after'].values()):,}**。
缺少reset时间、独立事前TaskSpec/意图来源、角色声明和reset前可得性证明。
label hash/mtime只能回溯当前录制目标，不能证明事前条件可用；不从未来轨迹反推合法计划。
数据目标监督准备完成不等于旧goal/event条件缓存获准训练，原HOLD门槛保留。
原生条件候选并未忽略：完整payload字段审计记录于
{link('full-manifest-r1/native-source-evidence.jsonl','原生字段与候选证据')}。
候选母序列/窗口为`{json.dumps(m['provenance']['native_candidate_mothers'])}` /
`{json.dumps(m['provenance']['native_candidate_windows'])}`。
Shadow原始`goal[T,3]`是reset后录制的task参数快照；原生goal或动作语义标签存在，不自动等于
完整TaskSpec、原执行/reset/来源绑定及reset前可得性证据。上述候选不升级为合法证书，
也不能泛称“原始数据没有goal字段”。

两根legacy派生view的拒绝原因：`{json.dumps(legacy['view_rejection_reasons'])}`。
几何pool或q/wrist存在不等于joint21/part16/annotation validity/contact/event语义已绑定，
不把human parent的MANO或contact贴给Inspire。完整文件缺口在上述逐view JSONL。
source2已从本地官方raw恢复295条；6条既有缓存对照pose误差0、手部最大误差
1.1920928955078125e-7m。source0腾盘造成的562条路径失效通过NAS原始备份恢复，
实测输入660,748,339字节、每条metadata/shape与原计划核对、562个packet全部READY。
这些是恢复已有数据，不是新增采集，也不记为物理结构性缺失。

## 磁盘、运行、复现

{table(['项目','字节'],[['新转换固定文件精确序列化',budget['conversion_fixed_exact_bytes']],['固定文件按FS block取整',budget['conversion_fixed_allocated_file_bytes']],['完整峰值上界（含manifest/12条最大partial/目录余量）',budget['total_peak_upper_bound_bytes']],['启动盘点可用',budget['filesystem_free_before_bytes']],['峰值后可用下界',budget['free_after_peak_lower_bound_bytes']],['最低实际记录可用',minimum],['硬停止线',30*2**30]])}

完整预算{link('storage-budget-v7.json')}；分段发布可续，receipt完成后才视为已处理。
每批64条，续跑校验window shard SHA与hand/contact/event metadata identity；已完成序列不重算，
batch receipt带attempt ID，运行manifest按attempt留档。调整输入/并发时只在序列边界STOP，
原STOP/receipt保留并命名归档，没有杀未知进程或删历史outputs。预算版本保留，没有用新数字
覆盖历史预检。环境为graspenv/Python{environment['python'].split()[0]}、Torch{environment['packages']['torch']} / CUDA{environment['cuda']}、
NumPy{environment['packages']['numpy']}；完整版本在environment-record.json。MANO使用GPU7，
KD-tree/审计最多12个CPU worker，全量去重8个CPU worker。GPU4/5/6上的其他作业未触碰。
seed：manifest/dedup登记值170，本次确定性排序/去重不消耗RNG；已有Shadow表面采样20261019。
geometry registry不重采样。没有随机抽样、优化器更新。

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\
 /home2/wyy/miniconda3/envs/graspenv/bin/python \\
 src/task/ptv3-shared-wam-v6/tools/audit/prepare_full_rigid_data.py \\
 --mode execute --workers 12 --device cuda:7 \\
 --output outputs/ptv3-shared-wam-v6/full-rigid-20261023-brief47-r1
# 上述同命令可幂等续跑；inventory、原始恢复清单及additional-local-jobs都在run目录。
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\
 /home2/wyy/miniconda3/envs/graspenv/bin/python \\
 src/task/ptv3-shared-wam-v6/tools/audit/summarize_full_rigid_data.py \\
 --workers 8 --run outputs/ptv3-shared-wam-v6/full-rigid-20261023-brief47-r1 \\
 --output outputs/ptv3-shared-wam-v6/full-rigid-20261023-brief47-r1/full-manifest-r1
# summary必须使用新目录；既有完成manifest不覆盖。
```

所有实际恢复/转换命令和环境都在run-manifest*.json、native-recovery-budget.json、
source0-copy-receipt*.json和remaining-legacy-budget.json；逐序列receipt在`receipts/`，
逐批在`batches/`。主manifest：{link('full-manifest-r1/manifest.json')}。
**manifest SHA256：`{sha}`**；文件级SHA与字节数在其`files`表，
{link('full-manifest-r1/manifest.sha256','SHA文件')}可直接校验。

工程核验：刚体准入/精确NPY字节公式/磁盘floor及serial/spawn并行全量manifest定向检查通过，
最新实测数量和日志见{link('verification-summary.json','核验清单')}；
真实9source的18个首尾窗口通过实际sample_window reader、形状与有限值检查，
{link('actual-reader-smoke.json','reader receipt')}明确保留legacy future-goal事实、不当成部署认证。
仓库`tools/verify.py --changed`另有腾盘后历史cm-pointflow outputs链接缺失，未为了过检查
恢复空文件或修改无关历史文档。没有训练、物理回放或预测性能改善结论。
'''
    text=text.replace('full-manifest-r1',str(summary.relative_to(run)))
    out.write_text(text)
    p.write(support/'report-receipt.json',dict(status='COMPLETED',manifest_sha256=sha,report_sha256=p.digest(out),report=str(out),
        coverage_visual_sha256={name:p.digest(support/name) for name in ['coverage-train.png','coverage-dev.png']},
        minimum_recorded_free_bytes=minimum,coverage_zero_reasons=dict(Counter(g['reason'] for g in gaps)),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),training_updates=0))
    print(json.dumps(dict(report=str(out),manifest_sha256=sha,windows_before=m['windows_before'],windows_after=m['windows_after'],minimum_recorded_free_gib=minimum/2**30)),flush=True)


if __name__=='__main__':main()
