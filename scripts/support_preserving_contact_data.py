"""Audited old replay plus actual HF21 generated-distribution observations."""
import json
from pathlib import Path
from structured_contact_data import load as old_load, BASE, ROOT
from run_paired_evaluator_resolution import sha


def load():
    import torch
    records, origins, hashes = old_load()
    source = BASE/'P-20261002-structured-contact-opportunity-source-r2'
    manifest_path = source/'run_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    results = source/'opportunity-results.json'
    stats_path = source/'opportunity-statistics-audit.json'
    stats = json.loads(stats_path.read_text())
    if (manifest['run_status'] != 'COMPLETED' or manifest['child_exit_code'] != 0
            or [p['seed'] for p in manifest['phases']] != list(range(611,619))
            or not stats['audit_passed'] or stats['source_sha256'] != sha(manifest_path)
            or stats['result_sha256'] != sha(results)):
        raise ValueError('accepted whole generated-distribution panel required')
    hashes.update(manifest['input_sha256'])
    for path in (manifest_path, results, stats_path, Path(__file__)):
        hashes[str(path.resolve())] = sha(path)
    for phase in manifest['phases']:
        directory = Path(phase['directory'])
        paths = (directory/'records.pt', directory/'planning.pt', Path(phase['audit']),
                 source/'phase-controls'/('seed%d' % phase['seed'])/'run_manifest.json')
        if (phase['run_status'] != 'COMPLETED' or phase['native_exit_code'] != 0 or phase['audit_exit_code'] != 0
                or sha(paths[0]) != phase['result']['record_sha256']
                or sha(paths[1]) != phase['result']['planning_sha256'] or sha(paths[2]) != phase['audit_sha256']):
            raise ValueError('complete native/planner audit hashes')
        audit = json.loads(paths[2].read_text())
        if audit['run_status'] != 'COMPLETED' or audit['run_manifest_sha256'] != sha(paths[3]):
            raise ValueError('full audited phase provenance')
        b = torch.load(paths[0], map_location='cpu', weights_only=False)
        if b['schema'] != 'ref2dex.structured_contact_opportunity_source.v1' or b['future_done'].any():
            raise ValueError('truthful actual source schema')
        records.append(b); origins.append(21)
        for path in paths: hashes[str(path.resolve())] = sha(path)
    seen = set()
    for b in records:
        for env,tick in zip(b['env_id'], b['trigger']):
            key = (b['seed'],int(env),int(tick))
            if key in seen: raise ValueError('reused actual window')
            seen.add(key)
    if len(seen) != 6263 or any(sha(Path(k)) != v for k,v in hashes.items()):
        raise ValueError('fixed actual union size/input drift')
    return records, origins, hashes
