"""Append actual later evidence to preserved v11; no new scientific inference."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PAPER=ROOT/'paper';BASE=ROOT/'src/task/CmResidual/research/contact_response/output'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    destination=PAPER/'manuscript-v12.tex';td=PAPER/'tables-v12';fd=PAPER/'figures-v12'
    assert not any(p.exists() for p in (destination,td,fd,PAPER/'source_manifest-v12.json'))
    inputs={};records={};provenance={};archived={}
    for package in [Path('/home2/wyy/tmp/ref2dex-contact-response-20261002-v11')]+[Path('/home2/wyy/tmp/ref2dex-contact-response-20261002-supplement-r'+str(i)) for i in range(1,20)]:
        m=json.loads((package/'delivery.json').read_text());assert m['run_status']=='COMPLETED'
        for original,target in m['original_source_mapping'].items():
            relative=target if isinstance(target,str) else target['path'];archived[original]=m['files_sha256'][relative]
    def load(name):
        p=BASE/name/'results.json';r=json.loads(p.read_text());assert r['run_status']=='COMPLETED'
        if name=='P-20261003-contrast-acquisition-r1':
            m=json.loads((p.parent/'run_manifest.json').read_text());assert m['run_status']=='COMPLETED' and m['inputs_unchanged']
            assert sha(p)==sha(p.parent/'fit/results.json')==m['input_sha256'][str(p.parent/'fit/results.json')]
        else:assert sha(p)==archived[str(p.resolve())],p
        inputs[str(p.resolve())]=sha(p);records[name]=r
        return r
    td.mkdir();fd.mkdir()
    def table(name,headers,rows):
        spec=dict(headers=headers,rows=[[str(v) for v in row] for row in rows]);provenance[name]=spec
        path=td/(name+'.tex');lines=[r'\begin{tabular}{l'+'r'*(len(headers)-1)+'}',r'\toprule',' & '.join(headers)+r'\\',r'\midrule']
        lines+=[' & '.join(row)+r'\\' for row in spec['rows']];lines+=[r'\bottomrule',r'\end{tabular}'];path.write_text('\n'.join(lines)+'\n')
    f=lambda x:format(x,'.6f')
    policy=[]
    for label,run in [('Deterministic successor','P-20261002-option-model-policy-r2'),('Observed successor distribution','P-20261002-empirical-successor-policy-r1'),('Physical encoder to Q','P-20261002-physical-encoder-critic-r1')]:
        r=load(run)['physical105_counts_per384'];policy.append([label,r['p0'],r['cm'],r['dynamics_off'],r['direct_q']])
    budget=load('P-20261003-budgeted-physical-critic-r2')
    for label,key,q in [('Cold-Q budget block','blockA_counts_per384','cold_q'),('Equal-interaction budget block','blockB_counts_per384','equal_budget_q')]:
        r=budget[key];policy.append([label,r['p0'],r['cm'],r['dynamics_off'],r[q]])
    table('later-policy',['Recipe /384','P0','Cm','Off','Task Q'],policy)
    collision=load('P-20261003-inspire-filter-impact-r1');table('collision',['Physics /192','P0','Cm','Off','Task Q'],[['Before ownership fix',*collision['before_pooled_per192']],['After ownership fix',*collision['after_pooled_per192']]])
    scale=load('P-20261003-cm-scale-cross-hand-r1');reports=scale['reports']
    rows=[[str(n),f(reports['mano_'+str(n)]['mano_eval']['parent_epe_mm']),f(reports['inspire_'+str(n)]['inspire_eval']['parent_epe_mm'])] for n in (512,2048,7168)]
    rows+=[['256-label adaptation',f(reports['adapt_mano_7168']['inspire_eval']['parent_epe_mm']),f(reports['adapt_scratch']['inspire_eval']['parent_epe_mm'])]]
    table('scale',['Training windows / adaptation','MANO / prior','Inspire / scratch'],rows)
    gran=load('P-20261003-cm-granularity-r1');conditions=('64_mean','256_mean','64_detail','256_detail')
    gv={h:[gran['reports'][h+'_'+c][h+'_eval']['parent_epe_mm'] for c in conditions] for h in ('mano','inspire')}
    table('granularity',['Context / neighbor aggregation','MANO EPE(mm)','Inspire EPE(mm)'],[[c.replace('_',' / '),f(gv['mano'][i]),f(gv['inspire'][i])] for i,c in enumerate(conditions)])
    execution=load('P-20261003-surface-execution-input-r1');cal=load('P-20261003-surface-calibration-r2')
    transfer=[]
    for h in ('mano','inspire'):
        for mode in ('oracle','action_velocity'):transfer.append([h+' / '+mode.replace('_',' '),f(execution['prior'][h][mode]['parent_epe_mm'])])
    for name in ('pretrained','scratch','shuffled'):transfer.append(['Calibrated / '+name,f(cal['reports'][name]['parent_epe_mm'])])
    transfer.append(['Object-flow persistence',f(cal['baselines']['persistence']['parent_epe_mm'])]);table('transfer',['Forecast variant','EPE(mm)'],transfer)
    cap=load('P-20261003-rigid-transport-capacity-r1');learn=load('P-20261003-rigid-coupling-learnability-r1');anch=load('P-20261003-state-anchored-transport-r2')
    table('transport',['Different model families','Full EPE(mm)','Matched state(mm)'],[[label,f(r['reports'][a]['episode_epe_mm']),f(r['reports']['state_only']['episode_epe_mm'])] for label,r,a in [('Outcome-fitted segment oracle',cap,'causal'),('Learned coefficient / score',learn,'full'),('State-anchored convex correction',anch,'full')]])
    rot=load('P-20261003-rotational-clearance-adequacy-r1');table('rotation',['Split','Windows / episodes','Flips(\%)','Rotation95th(mm)'],[[s,str(rot['summaries'][s]['windows'])+' / '+str(rot['summaries'][s]['episodes']),f(100*rot['summaries'][s]['episode_flip_rate']),f(1000*rot['summaries'][s]['weighted_rotation_p95_m'])] for s in ('train','held','all')])
    acq=load('P-20261003-contrast-acquisition-r1');table('acquisition',['Contrast minus comparator','Risk(mm$^2$)','One-sided95\%upper'],[[c,f(acq['relative_risk_mm2'][c]['point']),f(acq['relative_risk_mm2'][c]['upper95'])] for c in ('uniform','absolute','zero')])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.7),sharey=True)
    for ax,h,title in zip(axes,('mano','inspire'),('MANO','Inspire')):
        ax.bar(range(4),gv[h],color=['#426b9d','#73a2ca','#779b66','#b2c991']);ax.set_ylim(0,5);ax.set_xticks(range(4),['64\nmean','256\nmean','64\ndetail','256\ndetail']);ax.set_title(title)
    axes[0].set_ylabel('Parent-mean EPE (mm)');fig.tight_layout();fig.savefig(fd/'granularity.pdf');fig.savefig(fd/'granularity.png',dpi=160);plt.close(fig)
    names=('uniform','absolute','zero');points=[acq['relative_risk_mm2'][k]['point'] for k in names];ci=np.array([acq['relative_risk_mm2'][k]['central95'] for k in names])
    fig,ax=plt.subplots(figsize=(7.2,2.6));ax.errorbar(points,range(3),xerr=np.stack((np.array(points)-ci[:,0],ci[:,1]-points)),fmt='o',color='#426b9d',capsize=4);ax.axvline(0,color='black',ls='--',lw=1);ax.set_yticks(range(3),['vs uniform labels','vs absolute disagreement','vs zero effect']);ax.invert_yaxis();ax.set_xlabel('Contrast-acquisition relative risk (mm²); lower favors contrast');fig.tight_layout();fig.savefig(fd/'acquisition.pdf');fig.savefig(fd/'acquisition.png',dpi=160);plt.close(fig)
    (PAPER/'table_data-v12.json').write_text(json.dumps(provenance,indent=2)+'\n')
    source=(PAPER/'manuscript-v11.tex').read_text().replace('revision 11','revision 12').replace('2 October 2026','3 October 2026')
    abstract='''Revision 12 adds subsequent trained-policy comparisons and a native collision-
ownership correction: five shape filters were wrong under body-by-shape indexing.
Earlier physics results are retained as legacy observations, not corrected-system
Validation. Fixed data-scale, hand-transfer and 64/256-context studies fail their
gates. Corrected-data rigid transport has positive outcome-fitted capacity, but
causal learned heads miss matched gates. A full-mesh rotation screen and a
fixed-label-budget disagreement acquisition study also fail their decision gates.
Cm policy-training benefit remains unproved.
'''
    source=source.replace('All original failed gates, privileged-state limitations',abstract+'All original failed gates, privileged-state limitations',1)
    appendix=r'''
\clearpage
\section{Subsequent evidence and corrected physics}
This revision integrates later completed Probes while retaining every original
failed criterion. Unless explicitly marked corrected below, the historical
policy and randomized-effect results use legacy simulation filters. They cannot
be treated as corrected-physics Validation. The task is one airplane with three
synthetic references, not the user workspace's separate expert/motion results.
All listed policy experiments use self-trained actor parameters; pretrained
behavior in early data-source diagnostics is not evidence of this objective.

\subsection{Later policy-training comparisons}
Deterministic successor composition, an observed atomic successor distribution,
physical-feature-to-Q learning and equal-interaction-budget auxiliary learning
all fail their complete utility gates. Counts below are separate fixed recipes,
not a pooled effect estimate. Per-recipe environment seeds share one training
initialization and do not provide independent training-seed Validation. Cm's
occasional advantage over one control cannot replace the all-control gate.
\begin{table}[ht]\centering\small
\input{tables-v12/later-policy.tex}
\caption{Later trained-policy physical105 counts, each arm /384; legacy physics.
The task-Q column in the last two rows uses different declared budgets.}
\end{table}
\subsection{Collision ownership correction}
Native SDK inspection finds25rigid bodies but13collision shapes. Indexing body
names by shape number misassigns five shape filters. Actual SDK ownership fixes
non-thumb shapes2/4/6/8 from2to3 and thumb-base shape9 from3to2. The native
check has5mismatches before and0after under the same URDF. One frozen-policy
panel is independently rerun, with full geometry/command/label audits.
\begin{table}[ht]\centering
\input{tables-v12/collision.tex}
\caption{Fixed-policy sensitivity to corrected collision ownership. One seed,
no retraining, no exact paired solver states. PROMISING refers to sensitivity,
not Cm utility or performance improvement. All motion0counts remain zero.}
\end{table}
Subsequent representation studies reuse this corrected655 panel:384training and
384held episodes with a fixed whole-episode split. The holdout is repeatedly
viewed; it supplies route-selection evidence, not fresh Validation.

\clearpage
\subsection{Data scale, hand morphology and query granularity}
The source surface caches contain2048MANO and10135Inspire hand points.
The initial prior uses64object queries with four averaged hand neighbors per
query. Its22features include realized future hand motion; offline accuracy is
not a deployable command-conditioned forecast. Fifty training objects perhand
and nested512/2048/7168windows provide the scale comparison. Parent-equal EPE
is used, without selecting checkpoints on evaluation data. Pure hand morphology
is not identified because data distributions and simulation provenance differ.
\begin{table}[ht]\centering\small
\input{tables-v12/scale.tex}
\caption{Own-hand prediction EPE(mm), except the final row: MANO7168prior after
fixed256Inspire labels versus scratch on the same labels. That row evaluates
Inspire for both columns and is not an own-hand scale comparison.}
\end{table}
The initial-to-largest own-hand gains are8.015\%MANO and6.560\%Inspire,
below the10\%gate. Fixed256-label adaptation improves2.143\%against scratch,
below10\%, and2.330\%against shuffled pretraining, below5\%.
The granularity experiment fixes2048training windows perhand, common
initialization,23107parameters and1500updates perfit, changing64/256context
and mean/detail neighbor features. ALL supervised targets remain64queries;
this is not increased target resolution or a FLOP-matched comparison.
\begin{table}[ht]\centering\small
\input{tables-v12/granularity.tex}
\caption{Fixed-data own-hand granularity comparison; all six10\%gates fail.
These are eight fits, not independent training-seed replications.}
\end{table}
\begin{figure}[ht]\centering
\includegraphics[width=.95\linewidth]{figures-v12/granularity.pdf}
\caption{Recorded parent-equal errors with an untruncated zero-origin axis.
No uncertainty bars are invented from a single optimization seed.}
\end{figure}

\clearpage
\subsection{Corrected causal input and failed transfer calibration}
A54-coefficient train-only actuator fit reconstructs next hand geometry from
current state and commands, yielding0.900mm hand EPE against2.601mm velocity
persistence. This qualifies an execution bridge on the reused panel; it does
not establish hand-contact attribution or policy utility. Frozen priors fail
even with measured next hand motion. Calibration uses four matched1200-update
heads and fixed final checkpoints; only the hand-flow-removal contrast passes.
\begin{table}[ht]\centering
\input{tables-v12/transfer.tex}
\caption{Corrected-panel object-flow EPE(mm),6144windows/384heldepisodes.
Oracle rows use realized future hand motion and are nondeployable. Calibration
and frozen-prior variants are different models, not one ablation matrix.}
\end{table}
An independent calibration audit initially fails from NumPy indexing axis
order; the separate audit-only recovery retains all fitted weights and labels
without repeating any optimizer update. Protected raw packets and full held
forward/metric audits pass.

\clearpage
\subsection{Rigid transport capacity versus causal learnability}
The geometric capacity screen constructs13hand-link transport endpoints plus
zero/inertia. Outcome-fitted selection and scalar weights minimize actual
held errors on the union of15segments. It is not the full convex hull or a
physically feasible contact law. Its20.173\%advantage over outcome-fitted state
control is PROMISING capacity only: weights and endpoint choices use future
outcomes. Causal coefficient/score fitting then fails matched learning gates.
A separate state-anchored correction expands the mixture family and directly
optimizes flow EPE, preserving a frozen state baseline outside a hard current
2cm proximity gate. Its8.955\%gain misses the fixed10\%matched-state gate.
\begin{table}[ht]\centering\small
\input{tables-v12/transport.tex}
\caption{Different transport families, equal-episode EPE(mm) on corrected held
data. Each row has its own matched state control. The first row is an oracle;
it must not be read as a deployable winner over the learned rows.}
\end{table}
Far-field preservation is a hard rule, not learned generalization. Initial
state-anchor fitting fails before any head update because CPU normalization
changes the original GPU arithmetic; the repair restores the exact frozen
baseline and retains the original tolerance. The successful run completes
4500newheadupdates, separately from1500inherited state-model updates. All held
neural outputs/metrics pass;9early AdamWupdates are independently replayed,
4491remaining new updates are counted but not independently replayed.

\clearpage
\subsection{Rotational target adequacy}
An older6Dauxiliary target omits future rotation, whereas the64-point flow
already contains it. An observed geometry decomposition evaluates ALL155904
poses and155136transitions with the25002-vertex mesh and a fixed table upper
plane. Current-only eligibility requires30mm root rise and15--25mm clearance.
The true-future-translation diagnostic persists CURRENT orientation separately
at EACH step. It is not a simulated no-rotation trajectory or causal intervention.
\begin{table}[ht]\centering\small
\input{tables-v12/rotation.tex}
\caption{Equal-episode decision flips and weighted95th absolute rotational
clearance contribution. Held coverage and2mm magnitude gates pass;10\%flip
gate fails. Motion0has no eligible windows, so it was not assessed.}
\end{table}
All768full105 task labels agree between actual poses and this diagnostic,
with325successes (154/384held). CPU SciPy/NumPy independently rebuilds ALL
mesh poses and transitions; mesh differences are1.11e-16m versus GPU and
5.61e-8m versus retained native clearance. These observations close rotation
alone as the current remedy, not longer-horizon rotation or other geometries.

\clearpage
\subsection{Physical-response uncertainty for label acquisition}
World-model-guided exploration\cite{sekar2020}, structured curiosity for object
manipulation\cite{sancaktar2022} and tactile/contact-prioritized replay
\cite{vulin2021} are established. We test a specific offline mechanism rather
than claim generic exploration novelty: bootstrap disagreement of plus-minus
physical responses, which cancels common per-model state-dependent offsets.
This algebra does not prove calibrated epistemic uncertainty or useful labels.

Legacy randomized FIT492/493 contains3072windows. Three600-update bootstrap
models use512initiallabels, preserving shared actor/environment blocks. Each
acquisition method selects512additionallabels using only current contexts,
hypothetical actions and frozen models. Three final87--64--64--3MLPs share
initial weights, initial-only scales and1000-update schedules. Total4800new
updates; no native simulation or actor updates. The independently acquired
IID TEST494/495windows were previously viewed; they are not new Validation.

For physical response$Y$, randomized assignment$A$ with probability$p$, the
pseudo contrast for axis$j$is
\[
 Z_j=Y\left(\frac{\mathbf{1}(A=2j+1)}{p_{2j+1}}
           -\frac{\mathbf{1}(A=2j+2)}{p_{2j+2}}\right),\quad j=0,1,2.
\]
For fixed predictors$D_a,D_b$, relative squared causal-effect risk is identified
by the mean of$\|D_a\|^2-\|D_b\|^2-2\langle D_a-D_b,Z\rangle$.
Absolute effect RMSE is not identified. Two thousand descriptive bootstrap
draws preserve shared actor/environment blocks within test-seed/motion strata.
\begin{table}[ht]\centering\small
\input{tables-v12/acquisition.tex}
\caption{Action-contrast-disagreement acquisition versus uniform acquisition,
absolute-response-disagreement acquisition and zero-effect prediction.
Lower relative risk favors contrast acquisition. Only3/6gates pass: advantage
over absolute disagreement cannot rescue failure against uniform acquisition.}
\end{table}
\begin{figure}[ht]\centering
\includegraphics[width=.95\linewidth]{figures-v12/acquisition.pdf}
\caption{Recorded point estimates and central95\%descriptive intervals for fixed
fitted models. One-sided95\%upper bounds in the table use a different quantile;
the negative upper bound against absolute disagreement does not make its
central95\%interval exclude zero.}
\end{figure}
The uniform-comparison points are+0.069200mm$^2$on494 and+0.656493on495.
The complete gate is UNPROMISING. All6144rawresponses/assignments, every new
forward/rank/selection/metric/bootstrap and18early AdamWupdates independently
pass, with5.97e-9m forward discrepancy;4782updates are not independently
replayed. The actual run costs22.10s/9.92MB. All3072FITpool and3072TESTwindows
were already paid;1024selectedlabels must not be called1024online interactions.
The native acquisition route is stopped without further seed/budget rescans.
\clearpage
'''
    appendix=re.sub(r'(\d)([A-Za-z])',r'\1 \2',appendix)
    for a,b in [('All3072','All 3072'),('ALL155904','ALL 155904'),('ALL supervised','All supervised'),('All768','All 768'),('ALL held','All held'),('five shape','five shape'),('initiallabels','initial labels'),('additionallabels','additional labels'),('heldepisodes','held episodes'),('newheadupdates','new head updates'),('physicalresponses','physical responses'),('TESTwindows','TEST windows'),('FITpool','FIT pool'),('MLPs','MLPs')]:appendix=appendix.replace(a,b)
    protected={'v12','sekar2020','sancaktar2022','vulin2021','physical105','full105','corrected655'}
    for token in set(re.findall(r'[A-Za-z]+\d+',appendix))-protected:appendix=appendix.replace(token,re.sub(r'(?<=[A-Za-z])(?=\d)',' ',token))
    appendix=appendix.replace('6 Dauxiliary','six-dimensional auxiliary').replace('3of6','3 of 6')
    appendix=re.sub(r'(\d+\.\d+) e(-?\d+)',lambda m:'$'+m.group(1)+r'\times10^{'+m.group(2)+'}$',appendix)
    marker=r'\section{Limitations and research decision}';assert marker in source;source=source.replace(marker,appendix+'\n'+marker,1)
    source=source.replace('Revision 11 is compiled','Revision 12 is compiled')
    source=source.replace('All new work is isolated in branch \\texttt{agent/contact-response-cm}.','Current independent work is in \\texttt{agent/cm-active-acquisition}; earlier research branches and all historical revisions are retained.')
    source=source.replace(r'\end{thebibliography}',r'''\bibitem{sekar2020} R.Sekar et al. Planning to Explore via Self-Supervised World Models. ICML2020. \url{https://proceedings.mlr.press/v119/sekar20a.html}.
\bibitem{sancaktar2022} C.Sancaktar, S.Blaes, G.Martius. Curious Exploration via Structured World Models Yields Zero-Shot Object Manipulation. NeurIPS2022. \url{https://proceedings.neurips.cc/paper_files/paper/2022/hash/98ecdc722006c2959babbdbdeb22eb75-Abstract-Conference.html}.
\bibitem{vulin2021} N.Vulin et al. Improved Learning of Robot Manipulation Tasks via Tactile Intrinsic Motivation.2021. \url{https://arxiv.org/abs/2102.11051}.
\end{thebibliography}''')
    destination.write_text(source)
    # Snapshot actual inherited paper assets, not mutable historical code paths.
    for rel in re.findall(r'\\(?:input|includegraphics)(?:\[[^\]]*\])?\{([^}]+)\}',source):
        p=PAPER/rel;assert p.is_file(),p;inputs[str(p.resolve())]=sha(p)
    for p in (PAPER/'manuscript-v11.tex',PAPER/'native-v11/manuscript-v11.pdf',PAPER/'table_data-v12.json',Path(__file__),destination):inputs[str(p.resolve())]=sha(p)
    for p in BASE.glob('P-20261003-contrast-acquisition-r1/audit.json'):inputs[str(p.resolve())]=sha(p)
    result=dict(run_status='COMPLETED',source_sha256=inputs,new_tables=len(provenance),new_figures=2,prior_revision_preserved=True,journal_ready=False,
                scope='exploratory working revision; no matched Cm policy benefit established')
    (PAPER/'source_manifest-v12.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}))


if __name__=='__main__':main()
