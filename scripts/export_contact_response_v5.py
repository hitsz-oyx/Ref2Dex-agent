#!/usr/bin/env python3
"""Build portable evidence tables and a fifth, separately retained review PDF."""
from __future__ import annotations
from html import escape
import json
from pathlib import Path
import re
import sys

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph,SimpleDocTemplate,Spacer,Table,TableStyle,Image,KeepTogether
ROOT=Path(__file__).resolve().parents[1];PAPER=ROOT/'paper'
sys.path.insert(0,str(ROOT))
from scripts.export_contact_response_paper import plain,sha
BASE=ROOT/'src/task/CmResidual/research/contact_response/output'


def main():
    destination=PAPER/'manuscript-v5.pdf'
    if destination.exists():raise FileExistsError('retain previous render; choose a new explicit revision')
    paths=dict(physical=BASE/'P-20261001-contact-response-resolution-r1/analysis-v2/analysis.json',
               differential=BASE/'P-20261001-differential-response-learning-r1/results.json',
               factorization=BASE/'P-20261001-actuation-effect-factorization-r1/results.json',
               null=BASE/'P-20261001-null-action-recovery-r1/results.json',
               fresh=BASE/'P-20261001-fresh-causal-transfer-r1/failure_analysis.json',
               randomized=BASE/'P-20261001-randomized-effect-risk-r2/results.json',
               direct=BASE/'P-20261001-direct-randomized-response-test-r1/results.json',
               task=BASE/'P-20261001-randomized-task-selection-r1/results.json',
               protocol=BASE/'P-20261001-reference-hold-protocol-audit-r2/results.json',
               transfer=BASE/'P-20261001-hold-plateau-substrate-r1/analysis_correction_r1/results.json',
               curriculum=BASE/'P-20261001-hold-plateau-curriculum-r1/results.json',
               preload=BASE/'P-20261002-finger-preload-feasibility-r1/results.json')
    records={k:json.loads(p.read_text()) for k,p in paths.items()}
    for key in ('factorization','null','fresh','randomized','direct','task'):
        if records[key]['run_status']!='COMPLETED':raise ValueError('nonterminal evidence')
    headers={
        'physical':['Steps','Pulse RMS (mm)','Repeat RMS (mm)'],
        'differential':['Input/objective','Five steps (mm)','Ten steps (mm)'],
        'factorization':['Effect input','Near RMSE (mm)','All RMSE (mm)'],
        'null':['Variant','Near RMSE (mm)','Null response (mm)','Null inverse MSE'],
        'fresh':['Panel','Position RMS (mm)','Joint max','Quaternion max'],
        'randomized':['Comparison','Point (mm^2)','Central95%','Upper95%'],
        'direct':['Direct minus control','Point (mm^2)','Central95%','Upper95%'],
        'task':['Policy','Retained (%)','Stable (%)','Drop (%)','Max hold (s)'],
        'protocol':['Reference','Longest lift (steps)','Duration (s)','Final rise (mm)'],
        'holding':['Screen','Motion','Episodes','Retained75','Stable45','Rise (mm)'],
        'preload':['Dose (rad)','Pooled /48','Motion0 /16','Motion1 /16','Motion2 /16']}
    rows={key:[] for key in headers}
    for row in records['physical']['pooled']:rows['physical'].append([str(row['horizon']),f"{row['signal_rms_mm']:.3f}",f"{row['noise_rms_mm']:.3f}"])
    for label,method in [('Factual','factual'),('Differential','differential'),('No pulse','no_pulse')]:
        rows['differential'].append([label,*(f"{records['differential']['averaged'][method][str(h)]['contrast_rmse_mm']:.3f}" for h in (5,10))])
    for label,method in [('Global fit mean','global_fit_mean'),('Motion fit mean (privileged)','motion_fit_mean')]:
        rows['differential'].append([label,*(f"{records['differential']['baselines'][method][str(h)]['contrast_rmse_mm']:.3f}" for h in (5,10))])
    for label,method in [('State only','state_only'),('Native command','raw_command'),('Nominal PD motion','nominal_motion'),('Predicted hand motion','learned_motion'),('Measured future hand (privileged)','oracle_motion')]:
        values=records['factorization']['averaged'][method]
        rows['factorization'].append([label,f"{values['near']['rmse_mm']:.3f}",f"{values['all']['rmse_mm']:.3f}"])
    values=records['factorization']['passive'];rows['factorization'].append(['Passive',f"{values['near']['rmse_mm']:.3f}",f"{values['all']['rmse_mm']:.3f}"])
    for label,method in [('Factual only','factual'),('Full inverse','full_inverse'),('Effective inverse','effective_inverse'),('Hard quotient + inverse','quotient_inverse')]:
        values=records['null']['averaged'][method]['near']
        rows['null'].append([label,f"{values['factual_rmse_mm']:.3f}",f"{values['null_prediction_rms_mm']:.3f}",f"{values['inverse_null_mse']:.3f}" if 'inverse_null_mse' in values else 'not trained'])
    if records['fresh']['reason']!='INVALID_PRE_INTERVENTION_MATCH' or not records['fresh']['no_model_comparison']:
        raise ValueError('fresh acquisition audit semantics changed')
    for values in records['fresh']['panels']:
        rows['fresh'].append([values['panel'],f"{values['object_position_rms_mm']:.6f}",
                             f"{values['max_joint_abs_error']:.7f}",f"{values['max_object_orientation_abs_error']:.7f}"])
    for key,values in records['randomized']['risk_differences'].items():
        lo,hi=values['central95_mm2'];rows['randomized'].append([key.replace('_',' '),f"{values['point_mm2']:.3f}",f"[{lo:.3f},{hi:.3f}]",f"{values['upper95_mm2']:.3f}"])
    for key,values in records['direct']['risk_differences_direct_minus_control'].items():
        lo,hi=values['central95_mm2'];rows['direct'].append([key.replace('_',' '),f"{values['point_mm2']:.3f}",f"[{lo:.3f},{hi:.3f}]",f"{values['upper95_mm2']:.3f}"])
    for key,label in [('actor','Actor'),('random','Random'),('global','Global'),('conditional','Conditional factual')]:
        v=records['task']['policy_means'][key];rows['task'].append([label,*(f"{v[m]*100:.3f}" for m in ('retained_success','stable_success','drop_after_success')),f"{v['max_hold_seconds']:.3f}"])
    if records['protocol']['status']!='PASS':raise ValueError('protocol audit incomplete')
    for v in records['protocol']['reference_records']:
        rows['protocol'].append([Path(v['path']).parent.name,str(v['max_lift_run_frames']),f"{v['max_lift_run_frames']/30:.3f}",f"{v['end_lift_mm']:.3f}"])
    for screen,key in [('Actor transfer','transfer'),('PPO continuation','curriculum')]:
        record=records[key]
        if record['run_status']!='COMPLETED' or record['label']!='UNPROMISING':raise ValueError('holding screen semantics')
        for motion in ('0','1','2','Pooled'):
            v=record['pooled'] if motion=='Pooled' else record['motions'][motion]
            rows['holding'].append([screen,motion,str(v['episodes']),str(v['retained75_count']),str(v['stable45_count']),f"{v['mean_phase_height_mm']:.3f}"])
    if records['preload']['label']!='UNPROMISING':raise ValueError('preload primary boundary')
    for k,dose in enumerate((0.,.05,.15,.30)):
        v=records['preload']['arms'][str(k)]
        rows['preload'].append([f'{dose:.2f}',str(v['pooled']['retained75_count']),*(str(v['motions'][str(m)]['retained75_count']) for m in range(3))])
    tabledir=PAPER/'tables';tabledir.mkdir(exist_ok=True)
    tables={}
    for key in rows:
        filename=tabledir/(key+'-r1.tex');columns='l'+'r'*(len(headers[key])-1)
        value='\\begin{tabular}{'+columns+'}\n\\toprule\n'
        value+=' & '.join(headers[key])+r'\\'+'\n\\midrule\n'
        value+=''.join(' & '.join(row)+r'\\'+'\n' for row in rows[key])
        value+='\\bottomrule\n\\end{tabular}\n'
        if filename.exists() and filename.read_text()!=value:raise ValueError('different existing evidence table')
        if not filename.exists():filename.write_text(value)
        tables[key]=filename
    sourcepath=PAPER/'manuscript-v5.tex';source=sourcepath.read_text()
    expanded=source
    for key,path in tables.items():expanded=expanded.replace('\\input{tables/'+path.name+'}',path.read_text())
    if '\\input{' in expanded:raise ValueError('unresolved input')
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Body',fontName='Times-Roman',fontSize=10.2,leading=13.1,spaceAfter=7))
    styles.add(ParagraphStyle(name='TitlePaper',fontName='Times-Bold',fontSize=17,leading=20,spaceAfter=12,alignment=1))
    styles.add(ParagraphStyle(name='CaptionPaper',fontName='Times-Italic',fontSize=9,leading=11,spaceAfter=9))
    title=re.search(r'\\title\{(.*?)\}\s*\\author',source,re.S)[1].replace('\\\\',' ')
    story=[Paragraph(escape(plain(title)),styles['TitlePaper']),Paragraph('Exploratory working draft, revision 5; 2 October 2026',styles['CaptionPaper'])]
    abstract=re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',expanded,re.S)[1]
    story.append(Paragraph('Abstract',styles['Heading2']));story.append(Paragraph(escape(plain(abstract)),styles['Body']))
    body=expanded.split('\\end{abstract}',1)[1].split('\\begin{thebibliography}',1)[0]
    tokens=re.split(r'(\\section\*?\{[^}]+\}|\\begin\{table\}.*?\\end\{table\}|\\begin\{figure\}.*?\\end\{figure\})',body,flags=re.S)
    tableid=0;section=0;figureid=0
    for token in tokens:
        if token.startswith('\\section'):
            heading=re.search(r'\{([^}]+)\}',token)[1]
            if '\\section*' not in token:section+=1;heading=f'{section}. {heading}'
            story.append(Paragraph(escape(heading),styles['Heading2']))
        elif token.startswith('\\begin{table}'):
            tableid+=1;tabular=re.search(r'\\begin\{tabular\}\{[^}]+\}(.*?)\\end\{tabular\}',token,re.S)[1]
            data=[]
            for line in tabular.splitlines():
                if '&' in line and line.strip().endswith('\\\\'):
                    data.append([plain(x) for x in line.rstrip()[:-2].split('&')])
            table=Table(data,hAlign='CENTER',repeatRows=1)
            table.setStyle(TableStyle([('FONT',(0,0),(-1,-1),'Times-Roman',9),('FONT',(0,0),(-1,0),'Times-Bold',9),
                ('LINEABOVE',(0,0),(-1,0),.8,colors.black),('LINEBELOW',(0,0),(-1,0),.5,colors.black),
                ('LINEBELOW',(0,-1),(-1,-1),.8,colors.black),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('ALIGN',(1,0),(-1,-1),'RIGHT')]))
            caption=re.search(r'\\caption\{(.*?)\}',token,re.S)[1]
            story.append(KeepTogether([table,Spacer(1,6),Paragraph(f'Table {tableid}. '+escape(plain(caption)),styles['CaptionPaper'])]))
        elif token.startswith('\\begin{figure}'):
            figureid+=1
            relative=re.search(r'\\includegraphics(?:\[[^]]+\])?\{([^}]+)\}',token)[1]
            filename=(PAPER/relative).with_suffix('.png')
            from reportlab.lib.utils import ImageReader
            width,height=ImageReader(str(filename)).getSize()
            caption=re.search(r'\\caption\{(.*?)\}',token,re.S)[1]
            story.append(KeepTogether([Image(str(filename),width=6.5*inch,height=6.5*inch*height/width),Paragraph(f'Figure {figureid}. '+escape(plain(caption)),styles['CaptionPaper'])]))
        else:
            for paragraph in re.split(r'\n\s*\n',token):
                value=plain(paragraph)
                if value:story.append(Paragraph(escape(value),styles['Body']))
    story.append(Paragraph('References',styles['Heading2']))
    references=re.split(r'\\begin\{thebibliography\}\{[^}]+\}',expanded,maxsplit=1)[1].split('\\end{thebibliography}',1)[0]
    for key,value in re.findall(r'\\bibitem\{([^}]+)\}(.*?)(?=\\bibitem|$)',references,re.S):
        story.append(Paragraph(escape('['+key+'] '+plain(value)),styles['Body']))
    def footer(canvas,document):
        canvas.setFont('Times-Roman',8);canvas.drawString(.75*inch,.42*inch,'EXPLORATORY WORKING DRAFT | Frozen gates retained | No policy utility claim')
        canvas.drawRightString(7.75*inch,.42*inch,str(document.page))
    doc=SimpleDocTemplate(str(destination),pagesize=(8.5*inch,11*inch),leftMargin=.75*inch,rightMargin=.75*inch,topMargin=.7*inch,bottomMargin=.7*inch,title=plain(title))
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    secondary=BASE/'P-20261001-direct-randomized-response-test-r1/posthoc_controls.json'
    summary=json.loads(secondary.read_text())
    if summary['status']!='POSTHOC_EXPLORATORY' or summary['primary_label_unchanged']!='UNPROMISING':
        raise ValueError('secondary evidence boundary changed')
    for comparison in ('factual_minus_zero','factual_minus_global'):
        for metric in ('point_mm2','upper95_mm2'):
            if f"{summary['comparisons'][comparison][metric]:.3f}" not in source:
                raise ValueError('secondary prose numeric mismatch')
    inputs=[sourcepath,Path(__file__).resolve(),ROOT/'scripts/export_contact_response_paper.py',secondary,PAPER/'figures/reference_hold.pdf',PAPER/'figures/reference_hold.png',BASE/'P-20261001-reference-hold-figure-r1/figure_manifest.json',PAPER/'figures/finger_preload.pdf',PAPER/'figures/finger_preload.png',BASE/'P-20261002-finger-preload-figure-r1/figure_manifest.json',BASE/'P-20261001-static-hold-feasibility-r1/tabletop_axis_correction_r1/results.json',BASE/'P-20261001-static-hold-feasibility-r1/joint_tracking_audit_r1/results.json',*paths.values(),*tables.values()]
    manifest=dict(run_status='COMPLETED',rendering='ReportLab review copy; not native TeX compilation',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs},output_sha256=sha(destination),
        scientific_labels={key:value.get('label',value.get('classification')) for key,value in records.items()},tables_generated_from_recorded_json=True)
    (PAPER/'export_manifest-v5.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(pdf=str(destination),bytes=destination.stat().st_size,scientific_labels=manifest['scientific_labels'])))


if __name__=='__main__':main()
