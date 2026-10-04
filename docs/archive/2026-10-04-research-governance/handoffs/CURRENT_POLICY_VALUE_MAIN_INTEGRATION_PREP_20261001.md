# Current-policy value main integration preparation — 2026-10-01

`TASK_ID=T-20261001-current-policy-diagnostic-main-integration-prep`  `STATUS=COMPLETED`

This is a read-only integration preparation. The actual main base observed at delivery was
commit `96c55c2d03b6d52a460ae1b3180f9cfb8b207f78` with
`main/docs/STATE.md` SHA256 `25b686b7cd12862c1f4d4ca38c922ca7bdbbd98cc17f3fa822a50cea56cc7cb6`,
mode `0664`, size `4865` bytes. This actual base superseded the older dispatch context hash;
no main bytes were written.

The candidate [CURRENT_POLICY_VALUE_MAIN_STATE_CANDIDATE_20261001.md](CURRENT_POLICY_VALUE_MAIN_STATE_CANDIDATE_20261001.md)
was copied from the actual main STATE and changes only the current engineering-status block.
Its minimal diff is 6 additions and 2 removals. It records accepted collector and V-diagnostic
engineering, R1 native-cwd asset failure, R2 wrapper failure before GPU ownership, no new data,
and no V-sufficiency conclusion. It records RL r3 recovery within the remaining original
1200-second/2-GiB budget and CM real-input-guard work as active but unaccepted. HF08 remains
`UNPROMISING`/`PAUSED`/slot `1/1`, Cm policy utility remains `OPEN`, and a single realized MC
error is not called bias. Main North-star, history, and workflow text are preserved.

The JSON manifest lists 14 accepted root files with source SHA256/mode/size and main target
status: the two collector scripts, collection contract test, collector R2 runtime MD/JSON,
value audit script, diagnostic test and contract MD/JSON, latest diagnostic card, and
collection R1/R2 MD/JSON. All 14 target paths are currently absent in main and marked
`ABSENT_READY_FOR_ROOT_IMPORT`; `NEEDS_HELP` conflicts are zero. The manifest records only
future integration targets; it does not copy files into main. R3 and the CM guard are
explicitly excluded until separately accepted.

Excluded from this preparation are old long STATE/QUEUE replacement, workflow/identity/runtime/
config/AGENTS changes, shared Git writes, branch merge/cherry-pick, and scientific script or
experiment execution. Root should mechanically review the candidate against the actual main
base and import only the listed absent files after audit.

Output files:

- Candidate: SHA256 `ba5ad0eae1ca6b0c0ca6be329aae543ffa912bc53c89ba6b47d9ce74b93310eb`, mode `0664`, size `4865` bytes.
- JSON: SHA256 `93bd50db1175cc4f1234ef4d6baf526b8b812f9543fd4515f5ce8fa90dc2bd61`, mode `0664`, size `8580` bytes.
- This handoff: its final SHA is returned in the canonical Broker handoff.

Verification/resource boundary: actual main commit and STATE hash guard, manifest target scan,
candidate minimal diff and format checks must pass before terminal handoff. GPU `0`, at most
2 CPU threads, no experiment/training/scientific analysis/daemon/new process, under 600 seconds
and 0.1 GiB. Commit is `none`; use `BYTE_EXACT_ROOT_IMPORT`.
