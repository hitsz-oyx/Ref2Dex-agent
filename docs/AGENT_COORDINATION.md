# Ref2Dex Agent Coordination Contract

**Status:** ACTIVE · **schema:** `ref2dex.agent_coordination.v1` · **owner:** `/root`

This is the short operational contract for the four cooperating roles in the
Ref2Dex workspace. It is intentionally separate from research history. Agents
should link to this file instead of copying its rules into experiment cards or
activity logs.

## 1. Minimal context route

At the start of a task, read only:

1. `AGENTS.md`;
2. this file;
3. `docs/MISSION.md`, `docs/STATE.md`, and `docs/CAMPAIGN.md`;
4. the role-specific code and active experiment card.

Read `docs/RESEARCH_QUEUE.yaml` only when creating/approving a new Probe or
updating a hypothesis-family budget. Read `docs/SEED_LEDGER.yaml` only when
allocating or validating seeds. Do not bulk-read historical plans, activities,
old decision logs, or all experiment cards. Follow links only when current
evidence conflicts or a formal validation requires the historical baseline.

`STATE.md` is a compact routing index, not an experiment log. Detailed seeds,
metrics, tracebacks, and run paths belong in the card or run manifest.

## 2. Roles and hard boundaries

| Role / conversation | Owns | May do | Must not do | Required handoff |
| --- | --- | --- | --- | --- |
| `/root` (global supervisor) | Mission alignment, goals, parallel-resource allocation, acceptance | Assign bounded goals, inspect threads/branches/manifests, run read-only audits, issue Decision Memos | Send signals to Codex or jobs, blindly restart/kill work, duplicate a child’s collection, promote Probe evidence to Validation | Evidence-based status and next decision to the user |
| `agent/workflow` | Governance and context efficiency | Change workflow docs/templates/verification tests; run governance tests and `tools/verify.py` | GPU/Isaac Gym runs, research experiments, changing scientific claims without a checkpoint | Commit, test output, changed-file scope, known tradeoffs |
| `agent_baseline` | CPU-only canonical baseline/provenance audit | Inspect evaluator/config/checkpoint/motion hashes; build matched-off preflight manifests; run CPU smoke/tests | GPU, Isaac Gym, collectors, PPO, or consuming the Cm/HF02 slot | Branch/commit, manifest path, hashes, audit result, terminal status |
| `agent_Cm` (temporal/Cm) | One explicitly authorized Cm Probe | Run only the card’s frozen collection/fit, within its GPU/time/storage budget; record all provenance | Online/PPO follow-up, route/seed/metric drift, reuse of another run, or expanding scope after a failed gate | Card status, run manifests, resource evidence, Probe label, next decision |

The baseline and Cm tasks may run in parallel. Under the current supervisor
contract, baseline uses zero GPUs and `agent_Cm` owns at most one selected GPU;
`agent/workflow` uses zero GPUs. The campaign-wide four-GPU ceiling still
applies to any separately authorized work.

## 3. Goal and experiment contract

Every delegated goal must state, in one short message:

- the decision question and the cheapest discriminating test;
- the exact branch/commit and allowed paths;
- the experiment/card ID, seeds, controls, and resource budget;
- explicit stop conditions and deliverables;
- what the agent is not authorized to start.

“Continue” by itself is not a goal. A new route, claim, resource class, or
online follow-up requires a new goal (and a Decision Memo when `AGENTS.md`
requires one).

## 4. Preflight gate (before any non-smoke run)

The owner records a machine-readable `preflight.json` before starting the
run. It must verify:

1. branch, `HEAD`, worktree scope, and card schema/experiment ID;
2. route, config, evaluator, checkpoint, motion, and code hashes;
3. seed ownership and matched-control definitions;
4. output directory is unique and will not overwrite evidence;
5. GPU/process ownership, selected device, memory snapshot, and campaign
   limits;
6. command-line dry-run/tests and expected output schema;
7. wall-time, storage, nonfinite-data, drift, and row-count stop conditions.

If a check fails, status is `PREFLIGHT_FAILED` and the owner stops. An
implementation-only repair may be attempted only after recording the failed
attempt as `INVALID_IMPLEMENTATION`, preserving the same card and inputs;
changing the scientific design requires a new goal.

## 5. Supervision cadence and state machine

The supervisor uses read-only, event-aware polling:

| Phase | Default cadence | Inspect |
| --- | --- | --- |
| queued / preflight / blocked | every 45–60 s | thread turn, latest message, branch/HEAD, resource ownership |
| active GPU collection | every 2 min (never slower than 5 min) | process owner, GPU memory/utilization, manifest/log heartbeat, elapsed budget |
| active CPU analysis | every 5 min | process, output growth, elapsed budget, terminal errors |
| terminal or exception event | immediately, then one completion audit | manifest, card, hashes, commit, scoped processes, next decision |

The normal states are `PLANNED → PREFLIGHT → RUNNING → COMPLETED` or
`FAILED/STOPPED/UNKNOWN`; an import/wiring defect is
`INVALID_IMPLEMENTATION`, not a scientific result. No automatic restart is
allowed. The existing overload watchdog is limited to overload detection and
queueing; it is not a substitute for this supervisor and must not send process
signals.

At each poll, record only a compact status line. Do not append raw logs to
`STATE.md` or the conversation.

## 6. Stop, escalation, and evidence labels

Stop and report on hash/route drift, ambiguous process ownership, resource or
time-budget breach, missing manifest, nonfinite output, schema/row-count
failure, or a Decision Checkpoint. Three valid Probes without North-star
progress trigger route review rather than another local tweak.

Exploration may end only as `PROMISING`, `UNPROMISING`, or `UNCLEAR`.
`SUPPORTED`, `REFUTED`, and other formal claims require the Validation path.
Engineering smoke and `INVALID_IMPLEMENTATION` runs never count as scientific
Probe evidence.

## 7. Handoff format

Each owner ends with one compact record:

~~~text
STATUS=<terminal state>  LABEL=<if applicable>
BRANCH=<name>  HEAD=<sha>
CARD=<experiment/card id>  RUN=<run id or manifest path>
INPUTS=<key hashes>  RESOURCES=<GPU/process/time/storage>
EVIDENCE=<tests/metrics/known limits>
NEXT=<the single decision or blocker>
~~~

The supervisor accepts a task only when the record is reproducible from the
commit, card, manifest, and process audit. Otherwise it remains active or is
escalated to the user; it is not silently marked complete.

## 8. Integration ownership

The canonical integration worktree is:

~~~text
/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent-main
branch: main (tracking origin/main)
~~~

Only /root may merge an accepted child commit into main. Before merging,
/root verifies the child terminal record, diff scope, tests/verification,
worktree cleanliness, absence of owned processes, and whether the branch
contains unrelated research history. If a route branch has mixed ancestry,
merge only the explicitly accepted commits (or a clean integration branch);
never merge the entire branch by name merely because its tip passed a test.
After merge, run the mainline verification again and report the resulting
main commit to the user.
