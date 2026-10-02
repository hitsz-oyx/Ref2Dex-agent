"""Audit native paper against full raw evaluation counts and preserved revisions."""
import hashlib,json,re,subprocess,xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'src/task/CmResidual/research/contact_response/output'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    out=BASE/'P-20261002-paper-v11-audit-r1'
    if out.exists():raise FileExistsError(out)
    cm=json.loads((ROOT/'paper/native-v11/compile_manifest.json').read_text())
    assert cm['run_status']=='COMPLETED' and sha(cm['pdf'])==cm['pdf_sha256']
    assert all(sha(p)==h for p,h in cm['source_sha256'].items())
    for revision in (6,7,8,9,10):
        old=json.loads((ROOT/f'paper/export_manifest-v{revision}.json').read_text())
        assert sha(ROOT/f'paper/manuscript-v{revision}.pdf')==old['output_sha256']
        assert all(sha(ROOT/p)==h for p,h in old['source_sha256'].items())
    run=BASE/'P-20261002-continuous-critic-policy-resume-r3';rows=[]
    for seed in (568,569):rows+=json.loads((run/f's{seed}/rows.json').read_text())
    assert len(rows)==1536 and len({(q['seed'],q['environment']) for q in rows})==1536
    cohorts=[rows]+[[q for q in rows if q['seed']==s] for s in (568,569)]+[[q for q in rows if q['motion']==m] for m in range(3)]
    expected=[[str(sum(q['physical105'] for q in group if q['arm']==a)) for a in range(4)] for group in cohorts]
    def cells(path):
        return [[v.strip() for v in line.rstrip()[:-2].split('&')] for line in path.read_text().splitlines() if '&' in line and line.rstrip().endswith('\\\\')]
    actual=cells(ROOT/'paper/tables-v11/continuous_policy-r1.tex')
    assert [v[1:] for v in actual[1:]]==expected
    for new in (ROOT/'paper/tables-v11').iterdir():
        if new.name=='continuous_policy-r1.tex':continue
        original=ROOT/'paper/tables'/new.name
        decoded=new.read_text()
        for escaped,plain in [(r'\textasciicircum{}','^'),(r'\textasciitilde{}','~'),(r'\textbackslash{}','\\'),(r'\_','_'),(r'\%','%'),(r'\#','#'),(r'\$','$'),(r'\&','&'),(r'\{','{'),(r'\}','}')]:decoded=decoded.replace(escaped,plain)
        assert decoded==original.read_text(),new
    out.mkdir()
    textpath=out/'native-paper.txt';bboxpath=out/'native-paper-bbox.html'
    subprocess.run(['pdftotext','-layout',cm['pdf'],str(textpath)],check=True)
    subprocess.run(['pdftotext','-bbox',cm['pdf'],str(bboxpath)],check=True)
    text=textpath.read_text();normalized=re.sub(r'\s+',' ',text)
    assert all(re.search(r'Table\s+'+str(i)+r':',text) for i in range(1,22))
    assert all(re.search(r'Figure\s+'+str(i)+r':',text) for i in range(1,5))
    assert all(value in normalized for value in ['129/384','137/384','147/384','UNPROMISING','6 of seven','GPU4','float64','9060','15360','1536'])
    assert 'comparison has passed engineering and one' not in normalized
    parsed=ET.parse(bboxpath);bad=[]
    for page in parsed.findall('.//{http://www.w3.org/1999/xhtml}page'):
        width=float(page.attrib['width']);height=float(page.attrib['height'])
        for word in page.findall('.//{http://www.w3.org/1999/xhtml}word'):
            x0,x1,y0,y1=(float(word.attrib[k]) for k in ('xMin','xMax','yMin','yMax'))
            if not(42<=x0 and x1<=width-42 and 30<=y0 and y1<=height-24):bad.append(word.text)
    assert not bad,bad[:20]
    pages=text.split('\f')
    result=dict(run_status='COMPLETED',native_pdf_sha256=cm['pdf_sha256'],source_inputs_verified=len(cm['source_sha256']),
        previous_revisions6_to10_unchanged=True,all20_previous_table_values_preserved=True,
        continuous_table_rebuilt_from_all1536_rows=True,expected_continuous_table_counts=expected,
        all21_tables_and4_figures_present=True,pages=len(text.rstrip('\f\n').split('\f')),
        continuous_pages=[i+1 for i,p in enumerate(pages) if 'continuous policy-training' in p or 'Final-checkpoint-only physical105' in p],
        no_pdf_text_clipping=True,minor_tex_overfull_warnings_retained=True,one_optimization_seed_probe_only=True,journal_ready=False,
        audit_script_sha256=sha(Path(__file__)),pdf_text_sha256=sha(textpath))
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
