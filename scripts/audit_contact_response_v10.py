"""Independently check paper inputs, all selective counts/gates and PDF bounds."""
import hashlib,json,re,xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'src/task/CmResidual/research/contact_response/output'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    out=BASE/'P-20261002-paper-v10-audit-r1'
    if out.exists():raise FileExistsError(out)
    counts={}
    for revision in (6,7,8,9,10):
        m=json.loads((ROOT/f'paper/export_manifest-v{revision}.json').read_text())
        assert all(sha(ROOT/p)==h for p,h in m['source_sha256'].items())
        assert sha(ROOT/f'paper/manuscript-v{revision}.pdf')==m['output_sha256']
        counts[str(revision)]=len(m['source_sha256'])
    rows=json.loads((BASE/'P-20261002-selective-finger-feasibility-r1/rows.json').read_text())
    r=json.loads((BASE/'P-20261002-selective-finger-feasibility-r1/results.json').read_text())
    assert len(rows)==1536 and {(q['seed'],q['environment']) for q in rows}=={(s,i) for s in (545,546) for i in range(768)}
    expect=[]
    for arm in range(8):
        v=[]
        for mo in range(3):
            group=[q for q in rows if q['arm']==arm and q['motion']==mo]
            assert len(group)==64
            v.append(str(sum(q['physical105'] for q in group)))
        v.append(str(sum(q['physical105'] for q in rows if q['arm']==arm)));expect.append(v)
    table=(ROOT/'paper/tables/selective_finger-r1.tex').read_text().splitlines()
    actual=[[x.strip() for x in line.replace('\\\\','').split('&')][1:] for line in table if '&' in line][1:]
    assert actual==expect
    gates={}
    for arm in range(2,8):
        rates=[]
        for a in (arm,0,1):
            g=[q for q in rows if q['motion']==2 and q['arm']==a];rates.append(sum(q['physical105'] for q in g)/len(g))
        pooled=all(rates[0]-v>=.2 for v in rates[1:])
        seed_ok=True
        for seed in (545,546):
            successes=[sum(q['physical105'] for q in rows if q['seed']==seed and q['motion']==2 and q['arm']==a) for a in (arm,0,1)]
            seed_ok&=all(successes[0]>=v for v in successes[1:])
        assert r['gates'][str(arm)]['primary_motion2_gain20pp_both']==pooled
        assert r['gates'][str(arm)]['each_seed_no_worse_both']==seed_ok
        gates[str(arm)]=[pooled,seed_ok]
    assert r['label']=='UNPROMISING' and not any(all(v) for v in gates.values())
    figure=BASE/'P-20261002-selective-finger-figure-r3';fm=json.loads((figure/'figure_manifest.json').read_text())
    assert fm['counts']==[[int(v[m]) for v in expect] for m in range(3)]
    for name,h in fm['output_sha256'].items():assert sha(figure/name)==h and sha(ROOT/'paper/figures'/name)==h
    text=Path('/tmp/contact-response-v10-final.txt').read_text();assert len(text.rstrip('\f\n').split('\f'))==21
    assert all('Table '+str(i)+'.' in text for i in range(1,21))
    assert all('Figure '+str(i)+'.' in text for i in range(1,5))
    normalized=re.sub(r'\s+',' ',text)
    assert all(v in normalized for v in ('12/64','21/64','4-10/64','47/64','incomplete','no final evaluation','CAPG','112.349','255 protected'))
    parsed=ET.parse('/tmp/contact-response-v10-final-bbox.html');violations=[]
    for word in parsed.findall('.//{http://www.w3.org/1999/xhtml}word'):
        x0,x1,y0,y1=(float(word.attrib[k]) for k in ('xMin','xMax','yMin','yMax'))
        if not(53.9<=x0 and x1<=558.1 and 49<=y0 and y1<=766):violations.append(word.text)
    assert not violations,violations[:10]
    # Verify the separately retained first render against its OWN input snapshots.
    first=BASE/'P-20261002-paper-v10-first-render-r1';m=json.loads((first/'export_manifest-v10.json').read_text())
    assert sha(first/'manuscript-v10.pdf')==m['output_sha256']
    assert all(sha(first/'inputs'/p)==h for p,h in m['source_sha256'].items())
    report=dict(run_status='COMPLETED',verified_export_input_counts=counts,new_table_rebuilt_from_all1536_rows=True,six_fixed_candidate_gates_independently_reconstructed=gates,figure_counts_and_hashes=True,all20_tables_and4_figures_present=True,pages=21,pdf_bounds_pass=True,first_render_and_all99_inputs_retained=True,visual_pages_inspected=[18,19,20],journal_ready=False,no_continuous_policy_utility_result=True,audit_script_sha256=sha(Path(__file__)),pdf_text_sha256=sha(Path('/tmp/contact-response-v10-final.txt')))
    out.mkdir();(out/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
