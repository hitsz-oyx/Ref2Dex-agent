"""Preserve completed trials and exact native steps after a lost policy process."""
import argparse,json,os,shutil
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run=a.worker.resolve();out=a.output.resolve()
    if out.exists():raise ValueError('fresh interruption receipt required')
    m=json.loads((run/'manifest.json').read_text());active=run/m['active_episode']['id'];native=json.loads((active/'physics/manifest.json').read_text());process=json.loads((active/'process.json').read_text())
    if any(Path('/proc',str(process[k])).exists() for k in ('client_pid','native_pid')):raise ValueError('recorded process still exists; do not double-run')
    if native['status']!='FAILED' or not ('IPC peer closed' in native.get('error','') or 'Connection reset by peer' in native.get('error','')):raise ValueError('unknown interruption semantics')
    if (active/'trajectory.npz').exists() or (active/'result.json').exists():raise ValueError('recover complete/partial trajectory first instead of discarding')
    completed=sum(r['controls'] for r in m['runs']);partial=int(native['steps'])
    if not 0<=partial<=600 or any(r['id']==active.name for r in m['runs']):raise ValueError('invalid active episode')
    # Previous carry may already include technical partial controls; do not lose them.
    prior_partial=int(m.get('partial_controls_retained',0));controls=completed+prior_partial+partial
    out.mkdir(parents=True);(out/'runtime-code').symlink_to(run/'runtime-code',target_is_directory=True);shutil.copyfile(run/'registration.json',out/'registration.json')
    for r in m['runs']:(out/r['id']).symlink_to(run/r['id'],target_is_directory=True)
    active_wall=max(0.,(active/'physics/manifest.json').stat().st_mtime-(active/'process.json').stat().st_mtime)
    result=dict(m,status='INTERRUPTED_POLICY_LOST',source_running_manifest=str(run/'manifest.json'),environment_controls=controls,elapsed_s=m['elapsed_s']+active_wall,
        interruption=dict(episode=active.name,native_steps=partial,native_error=native['error'],client_exit_cause='unknown; no worker exit receipt',completed_trials_kept=len(m['runs']),partial_terminal_outcome='unobserved; rerun only the same unfinished assigned initial/seed',technical_overhead_controls=partial))
    (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');print('PRESERVED',len(m['runs']),'completed;',partial,'interrupted controls;',controls,'all consumed controls')


if __name__=='__main__':main()
