"""Read-only accounting bounds for UNKNOWN/voxel harm; no rollout generation."""
import argparse
import json
import sys
from pathlib import Path

TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]
OUTPUT=ROOT/'outputs/ptv3-shared-wam-v6/negative-result-audit-20261022-r1'


def count_bounds(draws,unknown_draws,ended,voxel):
    """Injection: each ended episode consumes one distinct initialization draw.

    This is a set-cardinality bound, NOT an estimator of episode frequency.
    D-E unmatched draws may include startup replacement, live episodes and tail.
    """
    if not (0<=unknown_draws<=draws and 0<=voxel<=ended<=draws):raise ValueError('inconsistent cardinalities')
    lower=max(0,unknown_draws-(draws-ended));upper=min(unknown_draws,ended)
    union_lower=max(voxel,lower);union_upper=min(ended,voxel+upper)
    return dict(ended_unknown=[lower,upper],unknown_voxel_overlap=[max(0,lower+voxel-ended),min(upper,voxel)],
        unknown_or_voxel=[union_lower,union_upper],remaining_defined_terminal_nonvoxel=[ended-union_upper,ended-union_lower],
        unmatched_initialization_draws=draws-ended,
        formulas=dict(unknown_lower='max(0, Udraw-(D-E))',unknown_upper='min(Udraw,E)',
            union_lower='max(V,Ulower)',union_upper='min(E,V+Uupper)',remaining='E - union, in reversed bound order'))


def transition_bounds(total,ended,count_range,minimum=3,maximum=1800):
    """Transitions INSIDE ended affected episodes; not causal avoidable waste.

    Upper bound reserves >=minimum transitions for every other ended episode.
    Transitions in ongoing episodes and any useful learning inside these episodes
    prevent interpreting the interval as wasted PPO updates or wall-clock time.
    """
    lo,hi=count_range
    if not (0<=lo<=hi<=ended and minimum*ended<=total and minimum<=maximum):raise ValueError('inconsistent budget')
    lower=minimum*lo;upper=min(maximum*hi,total-minimum*(ended-hi))
    return dict(transitions=[lower,upper],fraction_of_updated_transitions=[lower/total,upper/total],
        minimum_controls_per_ended_episode=minimum,maximum_controls_per_ended_episode=maximum,
        formulas=dict(lower='minimum * episode_lower',upper='min(maximum*episode_upper, T-minimum*(E-episode_upper))'),
        definition='observed transitions in ended affected episodes; not all exposure including unfinished episodes, not useful-gradient loss or causal recoverable budget')


def source_line(path,needle):
    lines=[i for i,text in enumerate(path.read_text().splitlines(),1) if needle in text]
    if not lines:raise ValueError('missing evidence '+str(path)+': '+needle)
    return dict(file=str(path),lines=lines,needle=needle)


def prefix_command_pool(reach):
    """Count admitted command candidates; tau-valid is not task-active/include."""
    cases=reach['cases'];mode_counts={mode:sum(c['prefix_commands'] for c in cases if c['terminal_mode']==mode)
        for mode in ['HOLD','RELEASE','UNKNOWN']};total=sum(mode_counts.values())
    bc=ROOT/'outputs/wam-rl-policy-v3/inspire-scheduled686-bc-20261020-r4/manifest.json'
    manifest=json.loads(bc.read_text());cache=Path(manifest['tau_cache']);layout=json.loads((cache/'layout.json').read_text())
    lookup={c['sequence']:c for c in cases};train=[r for r in layout['records'] if r['split']=='train'];cursor=0
    for row in train:
        c=lookup[row['sequence']]
        assert row['source']==c['source'] and row['commands']==c['prefix_commands']
        assert row['lo']==cursor and row['hi']==cursor+row['commands'];cursor=row['hi']
    assert len(train)==len(cases) and cursor==total
    import numpy as np
    valid=np.load(cache/'train-valid.npy',mmap_mode='r');assert valid.shape==(total,)
    unknown_tau_valid=sum(int(valid[r['lo']:r['hi']].sum()) for r in train if lookup[r['sequence']]['terminal_mode']=='UNKNOWN')
    return dict(admitted_sequences=len(cases),candidate_commands=total,candidate_commands_by_mode=mode_counts,
        unknown_candidate_command_fraction=mode_counts['UNKNOWN']/total,layout=str(cache/'layout.json'),
        row_alignment='All501 TRAIN sequence/source identities and contiguous lo/hi ranges equal the actual admitted prefix command counts.',
        tau_valid_array=str(cache/'train-valid.npy'),tau_valid_candidate_commands=int(valid.sum()),unknown_tau_valid_candidate_commands=unknown_tau_valid,
        bc_manifest=str(bc),bc_manifest_task_active_train_rows=manifest['train_rows'],bc_manifest_tau_valid_train_rows=manifest['train_valid_rows'],
        task_active_include_by_row='UNAVAILABLE: no persisted row-aligned task-active/include array in the actual scheduled cache; train-valid.npy is tau validity only.',
        actual_sampled_unknown_minibatch_frequency='UNKNOWN: sampler row identities/visit counts were not saved; startup_half is not uniform and inherited BC is a separate history.',
        scope='Exact admitted candidate pool and row-aligned tau-valid subset only, not actual BC task-active UNKNOWN rows, sampled batches, optimizer exposure or PPO transition fraction.')


def audit(source=OUTPUT):
    reach=json.loads((source/'task-reachability.json').read_text());history=reach['history'];run=Path(reach['historical_run'])
    natives=[json.loads(Path(r['source']).read_text()) for r in history['actual_motion_initializations']]
    ranks=[json.loads((run/('rank-%d/manifest.json'%rank)).read_text()) for rank in range(3)]
    total_draws=sum(sum(n['motion_visits']) for n in natives);unknown_draws=history['actual_motion_initialization_totals']['UNKNOWN']
    assert total_draws==sum(history['actual_motion_initialization_totals'].values())
    for native in natives:assert sum(native['motion_visits'])==native['indexed_resets']+native['envs']
    snapshots=history['curves'][0]['snapshots'];last=snapshots['1575'];ended=last['completed_episodes'];voxel=last['workspace_failures']
    for rank in history['curves']:
        for iteration in ['1570','1575']:
            assert all(rank['snapshots'][iteration][key]==snapshots[iteration][key] for key in ['completed_episodes','workspace_failures','successes'])
    assert all(r['resume_iteration']==231 and r['iteration']==1575 and r['reset_probe'] is False for r in ranks)
    envs=sum(r['envs_per_rank'] for r in ranks);new_updates=ranks[0]['iteration']-ranks[0]['resume_iteration']
    steps_per_update=ranks[0]['config']['learn']['nsteps'];updated=new_updates*steps_per_update*envs
    native_transitions=sum(n['steps']*n['envs'] for n in natives)
    bounds=count_bounds(total_draws,unknown_draws,ended,voxel)
    frozen=Path(reach['frozen_task']);code=frozen/'src/wam_rl_policy_v3';host=frozen/'tools/run/inspire_physics_host.py'
    official=run/'rank-0/runtime-official/dexgrasp/algorithms/rl/ppo/ppo.py'
    evidence=[source_line(host,'task.reset(ids)'),source_line(host,"audit['indexed_resets']+=len(selected)"),
        source_line(host,"audit['motion_visits']"),source_line(code/'measured_official_env.py',"self.rpc('reset'"),
        source_line(code/'measured_official_env.py','self.completed_episodes+=1'),source_line(code/'measured_official_env.py','len(self.histories[i])==4'),
        source_line(code/'measured_official_env.py','done[i]=decision.terminated or self.workspace[i]'),
        source_line(code/'task_scheduler.py','episode_timeout_s:float=60.'),
        source_line(code/'task_scheduler.py','acquire_timeout_s:float=15.'),
        source_line(code/'goal_scheduler.py','stall_timeout_s: float = 8.'),
        source_line(official,'current_obs = self.vec_env.reset()')]
    voxel_report=json.loads((source/'voxel-harm-diagnostics.json').read_text())
    result=dict(status='COMPLETED',device='cpu',python=sys.executable,historical_git_commit=reach['historical_git_commit'],
        actual_run=str(run),task_counts=history['terminal_modes'],prefix_candidate_pool=prefix_command_pool(reach),initialization_draw_counts=history['actual_motion_initialization_totals'],
        draws_are_not_ended_episodes=True,nominal_unknown_draw_weight=history['nominal_env_object_weighted_unknown'],
        observed_unknown_draw_fraction=unknown_draws/total_draws,
        iteration1570=dict(**snapshots['1570'],voxel_fraction=snapshots['1570']['workspace_failures']/snapshots['1570']['completed_episodes']),
        final_logged=dict(**last,voxel_fraction=voxel/ended),
        initialization_episode_injection=dict(status='VERIFIED_CODE_AND_FINAL_COUNT_ACCOUNTING',draws=total_draws,ended=ended,
            assumptions=['Every ended task episode is initialized once by native startup/reset motion sampling and cannot end twice without reset.',
                'All final motion_visits are monotonically accumulated over this same fresh-episode r3 host lifetime; source sequence identity binds to the same501 task modes.',
                'D-E is unmatched draw cardinality only; no claim its members are all live episodes or that final draw frequencies equal iteration1570 frequencies.'],
            missing='Per-ended-episode sequence and duration were not persisted; exact UNKNOWN ended count and UNKNOWN×voxel overlap remain UNKNOWN.'),
        episode_bounds=bounds,
        transition_ledger=dict(updated_control_transitions=updated,new_updates=new_updates,resume_iteration=231,
            actual_native_control_transitions=native_transitions,unupdated_tail_transitions=native_transitions-updated,
            tail_caveat='Final native reset total exceeds last logged reset count by30; outcomes of that unlogged tail are UNKNOWN. Reported0 applies to last logged53347 only.'),
        transitions_inside_ended_affected_episodes=dict(unknown=transition_bounds(updated,ended,bounds['ended_unknown']),
            voxel=transition_bounds(updated,ended,[voxel,voxel]),unknown_or_voxel=transition_bounds(updated,ended,bounds['unknown_or_voxel'])),
        bounds_scope=dict(minimum3='Fresh H4 has one reset state; first voxel check is control3. With reset_probe false, success0 and acquire15s/path-stall8s/episode60s, no other observed terminal can occur at controls1 or2.',
            maximum1800='TaskController episode timeout60s at30Hz; first terminal resets immediately. This is a conservative upper bound even for earlier gate exits.',
            budget_loss='Exact sample allocation, useful-gradient loss, avoidable compute/time and counterfactual benefit are not identified. Associated-episode steps are not automatically wasted.',
            causal_voxel_harm='Bounds on falsely rejected episodes or success improvement after gate removal: UNKNOWN; the separate P0 points are already metres from object geometry.'),
        after_exclusion=dict(observed_successes=0,remaining_ended_denominator_range=bounds['remaining_defined_terminal_nonvoxel'],
            allowed='Among ended episodes of this changing v4 training policy with defined terminal modes and no voxel exit, no complete TaskController success was recorded; denominator is bounded, not reconstructed.',
            forbidden=['Do not claim counterfactual training without UNKNOWN/voxel would still fail.',
                'Do not claim0/53347 independent evaluation, v6 failure, zero grasp/lift, or a binomial confidence bound from dependent training episodes.',
                'Do not convert52.2410% ended-episode exits into52.2410% lost environment transitions or recovered budget.']),
        p0_separate_protocol_comparison=voxel_report['protocol_comparison'],
        historical27792_first_points='UNRECOVERABLE: historical r3 did not save terminal point clouds, hand states, episode sequence or terminal durations.',
        evidence=evidence,impact_cards=[
            dict(defect='UNKNOWN admitted without success branch',sample_scope='64/501 tasks,9314/56449 initialization draws',episode_scope=bounds['ended_unknown'],
                budget_scope='See unknown associated-transition bound; not a direct waste estimate',
                card='src/task/wam-rl-policy-v3/docs/experiments/probes/P-20261020-inspire-measured-official-ppo.md',
                retract='All tasks had a valid full-task success opportunity; UNKNOWN explains all0; draw fraction equals ended-episode fraction',
                downgrade='0 successes retained only as descriptive training observation with invalid/uncertified opportunity mix',annotate='Mode counts, ended UNKNOWN bounds and absent ended-task telemetry'),
            dict(defect='PTv3 voxel gate called workspace and immediately terminates',sample_scope='Per-control historical bad points absent',episode_scope=[voxel,voxel],
                budget_scope='See voxel associated-transition bound;52.2410% is episode fraction only',
                card='src/task/wam-rl-policy-v3/docs/experiments/probes/P-20261020-inspire-measured-official-ppo.md',
                retract='Mechanical workspace failures; all gate exits are false kills; v6 PPO failed',
                downgrade='No unique method/generator attribution without qualified executable GT and protocol controls',annotate='1570=52.2661%;1575=52.2410%; retain raw counters and missing first-point data'),
            dict(defect='Different P0 evaluation gate/censoring scope',sample_scope='Existing separate P0 flow16 fixed rollouts,4768 H4 states;2 environments have voxel exits',
                episode_scope='N/A: original P0 did not terminate/reset episodes at these flags',
                budget_scope='Same saved trace:4800 vs4744 retained full controls;2316 vs2316 primary controls; actual physical ledger remains4800',
                card='src/task/ptv3-shared-wam-v6/docs/experiments/probes/P-20261022-wam-v6-executor-qualification.md',
                retract='Use these two P0 points as the historical27792-point distribution; assume admission-only is safe model inference',
                downgrade='Observational protocol comparison only, no gate-removed dynamic result',annotate='Both violations beyond reference prefixes; same-trace full task0/16 under both censorings'),
            dict(defect='Unfinished P1 and different selected object distribution',sample_scope='A/B partial seed191, C not started',episode_scope=None,
                budget_scope='Historical501-task weights and bounds are not transferred to P1',
                card='src/task/ptv3-shared-wam-v6/docs/experiments/probes/P-20261022-wam-v6-samples-efficiency-3arm.md',
                retract='Three-arm no-benefit or C-failure conclusion',downgrade='HOLD, incomplete method comparison',annotate='Keep stopped checkpoints/actual cost; no new runs authorized')])
    return result


def render(r):
    e=r['episode_bounds'];t=r['transition_ledger'];p=r['p0_separate_protocol_comparison']
    lines=['# UNKNOWN / voxel 负结果损害边界（2026-10-22，CPU）','',
        '只读已有JSON、冻结代码和P0状态；没有新仿真、训练、GPU操作，也未扩大grid或删除gate。',
        '本报告区分“受影响回合消耗的交互”与“可避免的浪费”：后者的因果数值未被识别。','',
        '## 已核计数与集合界','',
        '- 实际r3使用WAM v4，冻结`'+r['historical_git_commit']+'`。iteration1570：27792/53174 = **'+format(100*r['iteration1570']['voxel_fraction'],'.6f')+'%**；最终1575：27869/53347 = **'+format(100*r['final_logged']['voxel_fraction'],'.6f')+'%**。三个rank重复同一全局reduce，不求和。',
        '- 实际501份task：HOLD416 / RELEASE21 / UNKNOWN64。最终初始化draw56449，其中UNKNOWN9314（16.49985%）；draw不是结束回合。名义权重16.95252%也不是实际结束比例。',
        '- 同501份安全prefix合计**'+str(r['prefix_candidate_pool']['candidate_commands'])+' commands**，UNKNOWN **'+str(r['prefix_candidate_pool']['candidate_commands_by_mode']['UNKNOWN'])+' / '+str(r['prefix_candidate_pool']['candidate_commands'])+' = '+format(100*r['prefix_candidate_pool']['unknown_candidate_command_fraction'],'.6f')+'%**。HOLD49514、RELEASE4754。这是BC准入候选池，不是实际minibatch污染率。',
        '- 实际Scheduled686 cache的layout已逐序列/source/lo-hi核对501/501，train-valid.npy长度56418；它只表示tau有效，不是task-active/include。BC manifest实际task-active train_rows56036；未保存逐行include或sampler访问，UNKNOWN实际BC采样占比仍UNKNOWN，不根据startup_half猜测。',
        '- 冻结host采样+PPO eager reset保证：每个已结束回合对应一次不同初始化。D=56449，E=53347，未匹配draw的容量D−E=3102；它含startup替换/在途/日志尾部，不能指定成员。',
        '- 因而UNKNOWN结束回合U ∈ **'+str(e['ended_unknown'])+'**，公式max(0,9314−3102) ≤ U ≤9314。这是集合界，**没有把16.5%当作episode比例**。',
        '- UNKNOWN×voxel重叠 ∈ '+str(e['unknown_voxel_overlap'])+'；二者并集 ∈ **'+str(e['unknown_or_voxel'])+'**，不能相加计损。',
        '- 同时排除UNKNOWN与voxel后，剩余结束回合N ∈ **'+str(e['remaining_defined_terminal_nonvoxel'])+'**。只能写“该变化中的v4训练策略，在定义终态且非voxel退出的结束回合里，仍未观测到完整TaskController成功；确切分母不可恢复”。',
        '- 不能写“修复后二者仍0成功”、0/53347独立评估、v6失败、抓取0，或对相关训练回合套独立二项置信界。去gate会改变未来动作/reset/采样，反事实成功UNKNOWN。',
        '- 计数表明UNKNOWN结束回合不能覆盖所有失败回合；不排除UNKNOWN训练数据通过共享参数影响其余任务。该间接学习损害也不能由集合界识别。','',
        '## 交互预算：宽界也是结论','',
        '- r3从累计iteration231恢复，新完成1344次更新：**'+str(t['updated_control_transitions'])+'**个control transitions。native实际**'+str(t['actual_native_control_transitions'])+'**，多'+str(t['unupdated_tail_transitions'])+'个已采未更新交互；尾部30次reset的结果未记录，不把最后日志0延伸到它。',
        '- 以下仅界定“已结束受影响回合内部消耗的transition”，不是有用梯度损失、可回收步数或GPU小时。历史未存每回合length/sequence，不能估计精确损耗。','',
        '| 范围 | 回合数界 | 该类已结束回合内部transition界 | 占已更新预算界 |',
        '|---|---:|---:|---:|']
    for name,count in [('unknown',e['ended_unknown']),('voxel',[r['final_logged']['workspace_failures']]*2),('unknown_or_voxel',e['unknown_or_voxel'])]:
        b=r['transitions_inside_ended_affected_episodes'][name]
        lines.append('| '+name+' | '+str(count)+' | '+str(b['transitions'])+' | '+str([round(100*x,6) for x in b['fraction_of_updated_transitions']])+'% |')
    lines += ['', '界的前提：每个已结束失败回合至少3controls（H4首检查在3，reset_probe关闭，其余超时≥8s），最多60s×30Hz=1800；其他已结束回合也至少3controls。',
        '下界m×n_low；上界min(1800×n_high,T−m×(E−n_high))。完整任务UNKNOWN结构性无成功分支，但其局部接近/接触reward和学习信号不必无用。voxel可能拦住真实发散，假阳性与因果损害下界不能由计数推出。','',
        '## A3：既有P0状态的协议比较（单列，不能归因历史27792）','',
        '- 来源`'+p['source']+'`，16环境×ticks3..300 = **4768**个H4状态；V4 float32原gate与已存V6累计标记全部一致（只认证这些状态，不宣称float32/float64边界通用等价）。',
        '- 首个H4准入16/16通过；逐tick有2/16越界，原始不合法状态37/4768。admission-only保留4800controls，逐tick截断保留4744，差56；主要安全prefix仍**2316 vs2316，差0**。',
        '- 相同已存轨迹按两协议截断，完整任务成功都是0/16；这不是去gate后的动态重跑。原物理成本仍4800，不把56当实际省下的训练步。','',
        '| sequence / tick | 首点 | 轴/下界余量 | 距物体中心 / 最近目标surface | world xyz |',
        '|---|---|---|---|---|']
    for d in p['first_invalid_details']:
        g=d['geometry_relation'];h=d['hand_identity']
        lines.append('| '+d['sequence']+' / '+str(d['tick'])+' ('+str(round(d['seconds'],4))+'s) | '+h['slot_name']+'/'+h['canonical_joint_name']+' | '+d['axis']+' / '+str(round(d['lower_boundary_clearance_m'],6))+'m | '+str(round(g['object_anchor_distance_m'],6))+' / '+str(round(g['nearest_included_scene_surface_m'],6))+'m | '+str([round(v,6) for v in d['world_xyz']])+' |')
    lines += ['', '两首点都是right wrist（gate点512）；物手已分离约2.2m，不能称这两例是健康抓取的误杀。两例发生在各自prefix104/141之后。原27792条首点/手slot/episode时长不可恢复。',
        'admission-only在后续仍会遭遇模型表示域外输入；安全fallback、mask和行为改变未定义，不能把离线“继续计分”当可执行训练方案。本次不更改grid或gate。','',
        '## 缺陷 → 影响范围 → 卡片 → 修订','',
        '| 缺陷 | sample / episode / budget范围 | 具体卡 | 撤销 / 降级 / 加注 |','|---|---|---|---|']
    for c in r['impact_cards']:
        lines.append('| '+c['defect']+' | '+c['sample_scope']+'; episodes='+str(c['episode_scope'])+'; '+c['budget_scope']+' | `'+c['card']+'` | 撤销：'+c['retract']+'；降级：'+c['downgrade']+'；加注：'+c['annotate']+' |')
    lines += ['', '此表是证据范围建议；不覆盖旧数字、不将修复倒灌旧run，不自动授权新实验。P0及P1各自分布/运行身份不能套用r3的UNKNOWN频率。','', '## 可核路径与命令','',
        '- 输入：`'+str(OUTPUT/'task-reachability.json')+'`、`'+str(OUTPUT/'voxel-harm-diagnostics.json')+'`及其绑定的实际rank/native manifests。',
        '- 命令：`CUDA_VISIBLE_DEVICES=\'\' PYTHONDONTWRITEBYTECODE=1 /home2/wyy/miniconda3/envs/graspenv/bin/python src/task/ptv3-shared-wam-v6/tools/audit/audit_harm_budget_cpu.py`。',
        '- 机器产物：`'+str(OUTPUT/'harm-budget.json')+'`；逐首点诊断另见`voxel-harm-diagnostics.json`。','', '冻结证据：']
    for item in r['evidence']:lines.append('- `'+item['file']+':'+','.join(map(str,item['lines']))+'` — `'+item['needle']+'`')
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,default=OUTPUT)
    parser.add_argument('--output',type=Path,default=OUTPUT/'harm-budget.json')
    parser.add_argument('--markdown',type=Path,default=TASK/'docs/audit/negative-result-harm-20261022.md');args=parser.parse_args()
    result=audit(args.source);args.output.parent.mkdir(parents=True,exist_ok=True);args.markdown.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');args.markdown.write_text(render(result))
    print(json.dumps(dict(status=result['status'],episode_bounds=result['episode_bounds'],transitions=result['transitions_inside_ended_affected_episodes'],output=str(args.output))))


if __name__=='__main__':main()
