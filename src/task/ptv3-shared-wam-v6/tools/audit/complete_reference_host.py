"""Run the production host with artifact paths confined to this Task.

Only output-directory/scratch validation changes. Physics/reset/PD code is
unaltered; save the exact host and diff before executing it.
"""
import difflib
from pathlib import Path
import sys

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
source = ROOT/'src/task/wam-rl-policy-v3/tools/run/inspire_physics_host.py'
out = Path(sys.argv[sys.argv.index('--output') + 1]).resolve()
if out.exists() or TASK/'outputs' not in out.parents:
    raise ValueError('fresh Task-local host output required')
raw = source.read_text()
changes = {
    "ROOT/'outputs/wam-rl-policy-v3' not in a.output.parents": "Path(" + repr(str(TASK/'outputs')) + ") not in a.output.parents",
    "scratch=ROOT/'tmp'/a.output.parent.name": "scratch=Path(" + repr(str(TASK/'tmp')) + ")/a.output.parent.name",
    # Native changes cwd into gym_runtime before connecting. Keep a
    # relative pathname short enough for AF_UNIX at that call site.
    'a.socket=a.socket.resolve()': "a.socket=Path(os.path.relpath(a.socket.resolve(), a.output/'gym_runtime'))",
}
modified = raw
for before, after in changes.items():
    if modified.count(before) != 1:
        raise ValueError('host artifact-path anchor changed: ' + before)
    modified = modified.replace(before, after)
evidence = out.parent/'host-code'
evidence.mkdir(exist_ok=False)
(evidence/'production.py').write_text(raw)
(evidence/'executed.py').write_text(modified)
(evidence/'paths.diff').write_text(''.join(difflib.unified_diff(
    raw.splitlines(True), modified.splitlines(True), fromfile='production.py', tofile='executed.py')))
exec(compile(modified, str(source), 'exec'), dict(__name__='__main__', __file__=str(source)))
