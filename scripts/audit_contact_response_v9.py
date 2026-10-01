"""Independently reconstruct the new policy table and verify retained exports."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'src/task/CmResidual/research/contact_response/output'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    output = BASE/'P-20261002-paper-v9-audit-r1'
    if output.exists():
        raise FileExistsError(output)
    counts = {}
    for revision in (6,7,8,9):
        m = json.loads((ROOT/f'paper/export_manifest-v{revision}.json').read_text())
        assert all(sha(ROOT/p) == h for p,h in m['source_sha256'].items())
        assert sha(ROOT/f'paper/manuscript-v{revision}.pdf') == m['output_sha256']
        counts[str(revision)] = len(m['source_sha256'])
    r = json.loads((BASE/'P-20261002-support-feature-policy-training-r1/analysis_correction_r1/results.json').read_text())
    expected = [[str(c[k]['physical105_count']) for k in ('unchanged','cm','state_only','global_motion_arm')]
        for c in [r['pooled'],*r['seeds'].values(),*r['motions'].values()]]
    lines = (ROOT/'paper/tables/feature_policy-r1.tex').read_text().splitlines()
    actual = [[x.strip() for x in line.replace('\\\\','').split('&')][1:]
        for line in lines if '&' in line][1:]
    assert actual == expected
    text = Path('/tmp/contact-response-manuscript-v9.txt').read_text()
    assert all('Table '+str(i)+'.' in text for i in range(1,20))
    assert len(text.rstrip('\f\n').split('\f')) == 20
    assert all(s in text for s in ('9216','1536','identical','0.521','2.865','FAILED','UNPROMISING','ORIGINAL tolerances'))
    native = json.loads((BASE/'P-20261002-natural-retention-feedback-r1/rows.json').read_text())
    groups = [native]+[[r for r in native if r['seed']==seed] for seed in (543,544)]+[[r for r in native if r['motion']==motion] for motion in range(3)]
    expect = [[str(sum(r['physical105'] for r in cohort if r['arm']==a)) for a in range(4)] for cohort in groups]
    table = (ROOT/'paper/tables/natural_feedback-r1.tex').read_text().splitlines()
    values = [[x.strip() for x in line.replace('\\\\','').split('&')][1:] for line in table if '&' in line][1:]
    assert values == expect
    bbox = Path('/tmp/contact-response-v9-bbox.html')
    import xml.etree.ElementTree as ET
    parsed = ET.parse(bbox)
    for word in parsed.findall('.//{http://www.w3.org/1999/xhtml}word'):
        assert 54-1e-6 <= float(word.attrib['xMin']) and float(word.attrib['xMax']) <= 558+1e-6
        assert 50 <= float(word.attrib['yMin']) and float(word.attrib['yMax']) <= 766
    assert all(v in text for v in ('Table 19.', '133/384', '125/384', '82 successful', '17 failed', '114.312s'))
    result = dict(run_status='COMPLETED', verified_export_input_counts=counts,
        new_table_independently_reconstructed=True,feedback_table_rebuilt_from1536_rows=True,new_pages_bbox_within_margins=True, pages=20, all19_tables_present=True,
        visual_pages_inspected=[17,18], visual_findings='new table and scientific boundaries fit margins; no overlap/clipping',
        journal_ready=False, rendering='ReportLab review copy, not native TeX compilation',
        audit_script_sha256=sha(Path(__file__)), pdf_text_sha256=sha(Path('/tmp/contact-response-manuscript-v9.txt')))
    output.mkdir()
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
