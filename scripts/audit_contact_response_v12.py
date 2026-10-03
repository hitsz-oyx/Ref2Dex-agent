"""Independently check new paper cells, old assets, native text and page bounds."""
import hashlib,json,re,subprocess,xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'paper';B=ROOT/'src/task/CmResidual/research/contact_response/output'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(n):return json.loads((B/n/'results.json').read_text())
def cells(n):return [[v.strip() for v in line.rstrip()[:-2].split('&')] for line in (P/'tables-v12'/f'{n}.tex').read_text().splitlines() if '&' in line and line.rstrip().endswith('\\\\')][1:]
def numbers(n):return [row[1:] for row in cells(n)]


def main():
    out=B/'P-20261003-paper-v12-audit-r1';assert not out.exists();out.mkdir()
    m=json.loads((P/'native-v12/compile_manifest.json').read_text());assert m['run_status']=='COMPLETED' and sha(m['pdf'])==m['pdf_sha256']
    for p,h in m['source_sha256'].items():assert sha(p)==h,p
    old=json.loads((P/'native-v11/compile_manifest.json').read_text());assert sha(old['pdf'])==old['pdf_sha256']
    f=lambda x:format(x,'.6f');expect=[];raw_rows=0
    recipes=['P-20261002-option-model-policy-r2','P-20261002-empirical-successor-policy-r1','P-20261002-physical-encoder-critic-r1']
    for n in recipes:
        r=load(n);d=B/n
        if n==recipes[0]:d=Path(json.loads((d/'run_manifest.json').read_text())['source_run'])
        rows=[]
        for s in r['per_seed_counts_per192']:rows+=json.loads((d/f's{s}/rows.json').read_text())
        counts=[sum(q['physical105'] for q in rows if q['arm']==i) for i in (0,1,2,3)];assert counts==[r['physical105_counts_per384'][k] for k in ('p0','cm','dynamics_off','direct_q')]
        expect.append([str(v) for v in counts]);raw_rows+=len(rows)
    r=load('P-20261003-budgeted-physical-critic-r2')
    for key,q,seeds in [('blockA_counts_per384','cold_q',(655,656)),('blockB_counts_per384','equal_budget_q',(657,658))]:
        rows=[]
        for s in seeds:rows+=json.loads((B/'P-20261003-budgeted-physical-critic-r2'/f's{s}/rows.json').read_text())
        counts=[sum(v['physical105'] for v in rows if v['arm']==i) for i in (0,1,2,3)];assert counts==[r[key][k] for k in ('p0','cm','dynamics_off',q)];expect.append([str(v) for v in counts]);raw_rows+=len(rows)
    assert numbers('later-policy')==expect
    r=load('P-20261003-inspire-filter-impact-r1');assert numbers('collision')==[[str(v) for v in r[k]] for k in ('before_pooled_per192','after_pooled_per192')]
    r=load('P-20261003-cm-scale-cross-hand-r1')['reports'];assert numbers('scale')==[[f(r[h+'_'+str(n)][h+'_eval']['parent_epe_mm']) for h in ('mano','inspire')] for n in (512,2048,7168)]+[[f(r['adapt_mano_7168']['inspire_eval']['parent_epe_mm']),f(r['adapt_scratch']['inspire_eval']['parent_epe_mm'])]]
    r=load('P-20261003-cm-granularity-r1')['reports'];assert numbers('granularity')==[[f(r[h+'_'+c][h+'_eval']['parent_epe_mm']) for h in ('mano','inspire')] for c in ('64_mean','256_mean','64_detail','256_detail')]
    ex=load('P-20261003-surface-execution-input-r1');c=load('P-20261003-surface-calibration-r2');expect=[[f(ex['prior'][h][mode]['parent_epe_mm'])] for h in ('mano','inspire') for mode in ('oracle','action_velocity')]+[[f(c['reports'][mode]['parent_epe_mm'])] for mode in ('pretrained','scratch','shuffled')]+[[f(c['baselines']['persistence']['parent_epe_mm'])]];assert numbers('transfer')==expect
    expect=[]
    for topic,model in [('rigid-transport-capacity-r1','causal'),('rigid-coupling-learnability-r1','full'),('state-anchored-transport-r2','full')]:
        r=load('P-20261003-'+topic)['reports'];expect.append([f(r[model]['episode_epe_mm']),f(r['state_only']['episode_epe_mm'])])
    assert numbers('transport')==expect
    r=load('P-20261003-rotational-clearance-adequacy-r1');assert numbers('rotation')==[[str(r['summaries'][s]['windows'])+' / '+str(r['summaries'][s]['episodes']),f(100*r['summaries'][s]['episode_flip_rate']),f(1000*r['summaries'][s]['weighted_rotation_p95_m'])] for s in ('train','held','all')]
    r=load('P-20261003-contrast-acquisition-r1');assert numbers('acquisition')==[[f(r['relative_risk_mm2'][s]['point']),f(r['relative_risk_mm2'][s]['upper95'])] for s in ('uniform','absolute','zero')]
    tex=(P/'manuscript-v12.tex').read_text();oldtex=(P/'manuscript-v11.tex').read_text()
    oldinputs=re.findall(r'\\input\{([^}]+)\}',oldtex);assert all(r'\input{'+v+'}' in tex for v in oldinputs) and len(oldinputs)==21
    for rel in oldinputs:assert str((P/rel).resolve()) in m['source_sha256']
    first=json.loads((P/'prepare-v12-attempt1/export_failed.json').read_text());assert all(sha(P/'prepare-v12-attempt1'/rel)==h for rel,h in first['files_sha256'].items())
    subprocess.run(['pdftotext','-layout',m['pdf'],str(out/'paper.txt')],check=True);subprocess.run(['pdftotext','-bbox',m['pdf'],str(out/'bounds.html')],check=True)
    text=(out/'paper.txt').read_text();assert all(re.search(r'Table\s+'+str(i)+r':',text) for i in range(1,30));assert all(re.search(r'Figure\s+'+str(i)+r':',text) for i in range(1,7))
    normalized=re.sub(r'\s+',' ',text);assert all(s in normalized for s in ['revision 12','1.917404','7.039685','0.362847','-1.923996','4782','UNPROMISING','legacy','not corrected','not new Validation'])
    parsed=ET.parse(out/'bounds.html');bad=[]
    for page in parsed.findall('.//{http://www.w3.org/1999/xhtml}page'):
        width=float(page.attrib['width']);height=float(page.attrib['height'])
        for w in page.findall('.//{http://www.w3.org/1999/xhtml}word'):
            x0,x1,y0,y1=(float(w.attrib[k]) for k in ('xMin','xMax','yMin','yMax'))
            if not(42<=x0 and x1<=width-42 and 30<=y0 and y1<=height-24):bad.append(w.text)
    assert not bad,bad[:20]
    pages=text.rstrip('\f\n').split('\f');r=dict(run_status='COMPLETED',pages=len(pages),all8new_tables_verified=True,new_policy_raw_rows=raw_rows,
         all21inherited_tables_preserved=True,all29tables_and6figures_present=True,pdf_text_within_page_bounds=True,prior_v11_pdf_unchanged=True,failed_export_snapshot_retained=True,
         pdf_sha256=m['pdf_sha256'],journal_ready=False,source_inputs_verified=len(m['source_sha256']),new_section_pages=[i+1 for i,p in enumerate(pages) if any(t in p for t in ('collision ownership','Data scale, hand','Physical-response uncertainty'))])
    (out/'results.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))


if __name__=='__main__':main()
