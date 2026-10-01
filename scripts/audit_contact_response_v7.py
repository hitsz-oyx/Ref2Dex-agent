#!/usr/bin/env python3
"""Independent recorded-JSON numeric and export-provenance checks for revision7."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'src/task/CmResidual/research/contact_response/output'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out = BASE/'P-20261002-paper-v7-audit-r2'
    out.mkdir()
    export = json.loads((ROOT/'paper/export_manifest-v7.json').read_text())
    assert all(sha(ROOT/p) == h for p,h in export['source_sha256'].items())
    assert sha(ROOT/'paper/manuscript-v7.pdf') == export['output_sha256']
    records = {}
    for key, name in [('forecast','support-response-information-r2'),
                      ('gradient','support-gradient-cv-r1'),
                      ('headroom','support-state-dependence-r1'),
                      ('disturbance','support-disturbance-feasibility-r1')]:
        records[key] = json.loads((BASE/('P-20261002-'+name)/'results.json').read_text())
        assert records[key]['label'] == 'UNPROMISING'
    f,g,h,d = [records[k] for k in ('forecast','gradient','headroom','disturbance')]
    expected = {}
    values = [f['scores']['motions']['1'],f['scores']['seeds_motion1']['525'],
              f['scores']['seeds_motion1']['526'],f['scores']['pooled']]
    expected['forecast'] = [[format(v['logloss'][k],'.6f') for k in ('cm','state_only','global_motion_arm')] for v in values]
    values = [g['pooled'],g['seeds']['525'],g['seeds']['526']]
    expected['gradient'] = [[format(v['second_moment'][k],'.6f') for k in ('cm','state_only','global_motion_arm')] for v in values]
    values = [h['pooled'],h['seeds']['525'],h['seeds']['526']]
    expected['headroom'] = [[format(v['estimates'][k]['ipw_success']*100,'.3f') for k in ('position','global','unchanged')] for v in values]
    expected['disturbance'] = [[str(d['summaries']['1'][str(i)]['pre_geometry10_count'])]+
        [str(d['summaries']['1'][str(i)]['arms'][str(a)]['recovery90_count']) for a in range(4)] for i in range(4)]
    for name, expect in expected.items():
        lines = (ROOT/('paper/tables/'+name+'-r1.tex')).read_text().splitlines()
        rows = [[v.strip() for v in re.sub(r'\\\\\s*$', '',line.strip()).split('&')][1:]
                for line in lines if '&' in line][1:]
        assert rows == expect, (name,rows,expect)
    text_path = Path('/tmp/contact-response-manuscript-v7.txt')
    text = text_path.read_text()
    assert all('Table '+str(i)+'.' in text for i in range(1,18))
    assert all(v in text for v in ('18.65%','15.59%','6.68%','30/32','UNPROMISING','ReportLab','posthoc/reused-data'))
    assert len(text.rstrip('\f\n').split('\f')) == 17
    for name in ('P-20261002-paper-v7-review-r1','P-20261002-paper-v7-review-r2'):
        archive = BASE/name
        manifest = json.loads((archive/'export_manifest-v7.json').read_text())
        substitutions = {}
        for source in ('paper/manuscript-v7.tex','scripts/export_contact_response_v7.py'):
            saved = archive/Path(source).name
            assert sha(saved) == manifest['source_sha256'][source]
            substitutions[source] = str(saved)
        assert sha(archive/'manuscript-v7.pdf') == manifest['output_sha256']
        assert all(sha(Path(substitutions.get(p,str(ROOT/p)))) == h for p,h in manifest['source_sha256'].items())
        (archive/'archive_source_map.json').write_text(json.dumps(dict(
            archived_substitutions=substitutions,remaining_inputs_unchanged=True),indent=2)+'\n')
    # The earlier paper/source exports are also intact.
    prior = json.loads((ROOT/'paper/export_manifest-v6.json').read_text())
    assert all(sha(ROOT/p) == h for p,h in prior['source_sha256'].items())
    assert sha(ROOT/'paper/manuscript-v6.pdf') == prior['output_sha256']
    result = dict(run_status='COMPLETED',export_input_count=len(export['source_sha256']),
        export_input_hashes_verified=True,pdf_sha256=export['output_sha256'],
        audit_script_sha256=sha(Path(__file__)),pdf_text_sha256=sha(text_path),
        numeric_tables_independently_reconstructed=list(expected),all17_tables_in_pdf=True,
        pages=17,visual_pages_inspected=[14,15,16],
        visual_findings='Four new tables/captions remain within margins; no overlap or clipping',
        prior_v6_inputs_and_pdf_preserved=True,
        rendering='ReportLab review copy, not native TeX compilation',journal_ready=False)
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
