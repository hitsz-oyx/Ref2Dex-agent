# HF08 main evidence integration preparation — 2026-10-01

`TASK_ID=T-20261001-hf08-main-evidence-integration-prep`  `STATUS=COMPLETED`

This is a read-only integration preparation. Main was not edited, and the candidate was
built from the complete main `docs/STATE.md` bytes. The base guard is main commit
`3b775ebc84820e07a749fcb22e61f546c7df2725` with main STATE SHA256
`5f4fc88dfe1090f7fc1f21c67d39548bce876e2e31d8b46a464de2fa84408cce`, mode `0664`, size
`4179` bytes.

The candidate [HF08_MAIN_STATE_CANDIDATE_20261001.md](HF08_MAIN_STATE_CANDIDATE_20261001.md)
keeps main's North-star table, research history, links, layout, and workflow text. Its
only research-state correction is the HF08 block: it records the accepted R2 facts
`15/1920` source primary successes, `1/384` holdout success, and `7680` optimizer updates
in each e420 V checkpoint; keeps HF08 `UNPROMISING`/`PAUSED`/`1-of-1` and Cm policy utility
`OPEN`; states V sufficiency/current-policy calibration as `UNCLEAR` without a retraining or
convergence claim; records that the initial complete-state restoration audit is no longer
the current blocker; and describes CPU-only native Gaussian frame-0 frozen-policy complete
episode diagnostic preparation with no new GPU/training. A single realized MC error is not
called bias. Candidate diff from main is 8 additions and 3 removals.

The evidence manifest in the JSON has 11 exact entries: six R1/R2 audit archive files
(R1 script+MD+JSON and R2 script+MD+JSON), two root decision memos, the
`HF08_STATE_SYNC` MD, and the `HF08_AUDIT_STATE_LEDGER_SYNC` MD+JSON. The existing main
`D-20260930-hf08-evaluation-throughput.md` is an `EXISTS_MATCH` and is not overwritten;
all other listed evidence targets are absent and marked `ABSENT_READY_FOR_ROOT_IMPORT`.
There are zero `NEEDS_HELP` conflicts. The manifest is descriptive only; it does not copy
files into main.

The root workflow-version STATE/QUEUE, skills, root-owned commits, workflow/identity/runtime/
config/AGENTS files, and all other worktree content are explicitly excluded. Do not merge
this branch. Root may mechanically map the candidate to main STATE and import only the
listed absent evidence files after review.

Output hashes:

- Candidate: SHA256 `25b686b7cd12862c1f4d4ca38c922ca7bdbbd98cc17f3fa822a50cea56cc7cb6`, mode `0664`, size `4534` bytes.
- JSON: SHA256 `23eb308737f933b39de4154fae45160d13dd0df84c30acc507da882be5c067a1`, mode `0664`, size `6699` bytes.
- This handoff: its final SHA is returned in the canonical Broker handoff.

Verification and resources: main hash/commit guard passed; candidate minimal diff and
manifest conflict scan passed; candidate/JSON/MD formatting and trailing-whitespace checks
are required before terminal handoff. GPU `0`, two CPU threads maximum, no experiment,
training, scientific script, daemon, or new process; under 10 minutes and 0.1 GiB.
Commit is `none`; use `BYTE_EXACT_ROOT_IMPORT`.
