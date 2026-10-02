"""Bounded, local native TeX compilation into a unique output directory."""
import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve(); output = args.output.resolve()
    if output.exists() or ROOT not in output.parents or source.parent != ROOT / 'paper':
        raise ValueError('unique own output and own paper source required')
    manifest = json.loads(args.source_manifest.read_text())
    for path, digest in manifest['source_sha256'].items():
        assert sha(path) == digest, path
    assert str(source) in manifest['source_sha256']
    local = ROOT / '.research-tools/tectonic-0.17.0'
    installation = json.loads((local / 'install_manifest.json').read_text())
    binary = local / 'tectonic'
    assert sha(binary) == installation['binary_sha256']
    command = [str(binary), '--outdir', str(output), '--keep-logs',
               '--keep-intermediates', str(source)]
    environment = dict(os.environ, XDG_CACHE_HOME=str(ROOT / '.research-tools/cache'),
                       RAYON_NUM_THREADS='2', OMP_NUM_THREADS='2')
    output.mkdir()
    path = output / 'compile_manifest.json'
    record = dict(run_status='STARTED', command=command, source_sha256=manifest['source_sha256'],
                  source_manifest_sha256=sha(args.source_manifest),
                  compiler_sha256=installation['binary_sha256'],
                  local_install_only=True, compilation_budget_seconds=300,
                  rendering='native TeX', journal_ready=False)
    path.write_text(json.dumps(record, indent=2) + '\n')
    begin = time.monotonic()
    try:
        with (output / 'compiler.log').open('w') as log:
            child = subprocess.run(command, cwd=str(source.parent), env=environment,
                                   stdout=log, stderr=subprocess.STDOUT, timeout=300)
        record['exit_code'] = child.returncode
        pdf = output / (source.stem + '.pdf')
        if child.returncode or not pdf.exists():
            raise RuntimeError('native compiler failed; retain source and full log')
        for filename, digest in manifest['source_sha256'].items():
            assert sha(filename) == digest, filename
        record.update(run_status='COMPLETED', pdf=str(pdf), pdf_sha256=sha(pdf),
                      all_source_inputs_unchanged=True)
    except BaseException as error:
        record.update(run_status='FAILED', error=repr(error))
        raise
    finally:
        record['wall_seconds'] = time.monotonic() - begin
        path.write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps({k: v for k, v in record.items() if k != 'source_sha256'}), flush=True)


if __name__ == '__main__':
    main()
