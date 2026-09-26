# Codex research supervisor

`scripts/codex_research_supervisor.py` is a small external scheduler for the
global `/root` Codex conversation. It wakes an existing session only when the
session is idle and no message is already queued. It does not create a new
session, restart a failed turn, stop a process, allocate a GPU, or launch an
experiment.

## Safe dry run

Use the main conversation's thread ID explicitly. The current root thread is
`01a0d943-de74-7021-8d50-2a4e87fde613`; do not rely on automatic discovery for
this job.

```bash
cd /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent-main
python3 scripts/codex_research_supervisor.py \
  --thread 01a0d943-de74-7021-8d50-2a4e87fde613 \
  --codex-home /home2/wyy/oyx_ws/.codex_oyx_NewAPI \
  --node /home2/wyy/.nvm/versions/node/v24.19.0/bin/node \
  --codex-js /home2/wyy/.nvm/versions/node/v24.19.0/lib/node_modules/@openai/codex/bin/codex.js \
  --once --start-immediately --allow-blocked --dry-run --verbose
```

`--allow-blocked` is needed only because the current persisted root Goal is
blocked. It is an explicit opt-in; a blocked Goal is not silently resumed by
default.

## Long-running mode

After checking the dry-run output, run the real queue mode with a five-minute
wake interval and a finite 24-hour lifetime:

```bash
cd /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent-main
python3 scripts/codex_research_supervisor.py \
  --thread 01a0d943-de74-7021-8d50-2a4e87fde613 \
  --codex-home /home2/wyy/oyx_ws/.codex_oyx_NewAPI \
  --node /home2/wyy/.nvm/versions/node/v24.19.0/bin/node \
  --codex-js /home2/wyy/.nvm/versions/node/v24.19.0/lib/node_modules/@openai/codex/bin/codex.js \
  --interval 300 --poll-interval 30 --ack-timeout 180 \
  --max-runtime 86400 --start-immediately --allow-blocked \
  >> outputs/codex_research_supervisor/root.log 2>&1
```

The default prompt asks `/root` to inspect the three agent worktrees,
manifests, processes and merge readiness. Override it with `--message` when a
different bounded scheduling question is needed. The script waits while a
turn is active, while another message is queued, or while the previous wake
has not been acknowledged. A lock file prevents two copies from waking the
same thread.

Stop the process with `Ctrl-C` or `SIGTERM`. It performs no cleanup of
checkpoints, outputs, processes or Codex history. Runtime state is written to
`outputs/codex_research_supervisor/<thread>.json`, which is ignored by Git.

## Scope boundary

This scheduler is separate from `codex_overload_watchdog.py`. The overload
watchdog reacts only to structured `server_overloaded` events; this supervisor
performs periodic idle-turn scheduling. Neither tool is a substitute for the
research acceptance rules in `docs/AGENT_COORDINATION.md`. A wake-up is only a
request for `/root` to inspect and report; `/root` still decides whether a
child result can be merged or whether a Decision Checkpoint is required.
