"""Verify retained terminal evidence; no rerun or scientific gate changes."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'src/task/CmResidual/research/contact_response/output'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def verify(mapping, relative=False):
    for p, digest in mapping.items():
        assert sha(ROOT/p if relative else p) == digest, p
    return len(mapping)


def main():
    parent = BASE/'P-20261002-support-feature-policy-training-r1'
    out = BASE/'P-20261002-support-feature-policy-closeout-r1'
    if out.exists():
        raise FileExistsError(out)
    m = json.loads((parent/'run_manifest.json').read_text())
    assert m['run_status'] == 'FAILED' and m['last_update'] == 12
    count = verify(m['input_sha256'])
    c = parent/'analysis_correction_r1'
    cm = json.loads((c/'run_manifest.json').read_text())
    assert cm['run_status'] == 'COMPLETED'
    corrected = verify(cm['input_sha256'])
    agreement = json.loads((c/'action_agreement_audit.json').read_text())
    verify(agreement['input_sha256'], relative=True)
    assert agreement['all_three_heads_agree_count'] == 1536
    maximum = {}
    for i in range(1, 13):
        a = json.loads((parent/f'u{i:02d}/gradient_audit.json').read_text())
        assert a['run_status'] == 'COMPLETED' and a['update'] == i
        verify(a['input_sha256'])
        for k, v in a['maximum_error'].items():
            maximum[k] = max(maximum.get(k, 0), v)
    for s in range(529, 543):
        a = json.loads((parent/f's{s}/panel_audit.json').read_text())
        assert a['run_status'] == 'COMPLETED'
    preserved = {}
    for v in (6, 7):
        export = json.loads((ROOT/f'paper/export_manifest-v{v}.json').read_text())
        preserved[str(v)] = verify(export['source_sha256'], relative=True)
        assert sha(ROOT/f'paper/manuscript-v{v}.pdf') == export['output_sha256']
    result = dict(run_status='COMPLETED', parent_status_preserved='FAILED',
        corrected_analysis_status='COMPLETED', scientific_label='UNPROMISING',
        protected_parent_inputs_verified=count, corrected_inputs_verified=corrected,
        panel_audits=14, gradient_audits=12, maximum_update_error=maximum,
        agreement_states=1536, own_run_bytes=sum(p.stat().st_size for p in parent.rglob('*') if p.is_file()),
        preserved_paper_input_counts=preserved, no_training_or_physics=True,
        input_sha256={str(p):sha(p) for p in (parent/'run_manifest.json', c/'run_manifest.json', c/'results.json', c/'action_agreement_audit.json', Path(__file__))})
    out.mkdir()
    (out/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
