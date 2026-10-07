"""Verify an owned PPO checkpoint's ancestry back to random initialization."""
import hashlib
import json
from pathlib import Path

from .contracts import is_within


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def endpoint_epoch(manifest):
    command = manifest['command']
    epoch = int(command[command.index('--actual-epochs') + 1])
    if not 0 < epoch <= 500:
        raise ValueError('invalid trained endpoint epoch')
    if Path(manifest['checkpoint']).name != 'GRAB_%08d.pth' % epoch:
        raise ValueError('checkpoint filename does not match trained endpoint')
    return epoch


def self_trained_ancestry(run, owned_root):
    """Return frozen manifests/checkpoints; reject imported, changed or cyclic ancestry.

    This verifies recorded provenance, not whether PPO learned a useful policy.
    Each resumed source must be an endpoint of another completed owned Cm-off
    run, and its absolute epoch must agree with the recorded continuation.
    """
    root = Path(owned_root).resolve()
    run = Path(run).resolve()
    frozen, seen = {}, set()
    for _ in range(32):
        if not is_within(run, root) or run in seen:
            raise ValueError('checkpoint ancestry must be owned and acyclic')
        seen.add(run)
        path = run / 'run_manifest.json'
        manifest = json.loads(path.read_text())
        if manifest.get('run_status') != 'COMPLETED' or manifest.get('cm_enabled') is not False:
            raise ValueError('completed self-trained Cm-off ancestry required')
        checkpoint = Path(manifest['checkpoint']).resolve()
        if not is_within(checkpoint, run) or sha(checkpoint) != manifest['checkpoint_sha256']:
            raise ValueError('owned checkpoint identity changed')
        epoch = endpoint_epoch(manifest)
        frozen[str(path)] = sha(path)
        frozen[str(checkpoint)] = sha(checkpoint)
        if manifest.get('initialization') == 'random_scratch':
            if manifest.get('source_checkpoint') is not None or manifest.get('source_epoch') != 0:
                raise ValueError('scratch root must have no imported source')
            return frozen
        if manifest.get('initialization') != 'pinned_scratch_resume':
            raise ValueError('unrecognized checkpoint initialization')
        source = Path(manifest['source_checkpoint']).resolve()
        if (not is_within(source, root) or
                sha(source) != manifest['source_checkpoint_sha256']):
            raise ValueError('resumed checkpoint identity changed or is external')
        ancestor = next((p for p in source.parents
                         if is_within(p, root) and (p / 'run_manifest.json').is_file()), None)
        if ancestor is None:
            raise ValueError('resumed source has no owned training manifest')
        parent = json.loads((ancestor / 'run_manifest.json').read_text())
        if (Path(parent['checkpoint']).resolve() != source or
                parent['checkpoint_sha256'] != manifest['source_checkpoint_sha256'] or
                endpoint_epoch(parent) != manifest['source_epoch'] or
                not manifest['source_epoch'] < epoch):
            raise ValueError('resumed source does not match ancestor endpoint/epoch')
        run = ancestor
    raise ValueError('checkpoint ancestry exceeds bounded depth')
