# Local workflow runtime

Files in this directory are machine-local and are not research evidence.  The
autonomy lease and watchdog state live here so changing a conversation binding
does not change the Git history:

```text
SUPERVISOR_LEASE.json
root_watchdog/state.json
worker_event_poller/state.json
```

Use `python3 scripts/researchctl.py supervisor pause|resume|status` to change
the lease.  A missing lease is treated as disabled; the watchdog never resumes
a paused Goal without an enabled lease.
