# Ref2Dex 工作流执行层迁移

Status: ready-for-agent

## Problem Statement

当前工作流规范与监督脚本冗余，任务排队未构成实际运行时投递，研究监督依赖 Codex 内部存储。用户希望保留自主科研判断，采用已有执行工具，并让多个独立账号在前台关闭后仍可持续推进。

## Solution

保留 Codex 研究主管和固定逻辑角色；采用按角色与账号隔离的外部执行后端。提供统一命令入口与可独立运行的后台监督循环。研究暂停停止新派发，现有实验继续收尾；意外退出有限恢复。旧任务由旧系统收尾，新系统接新任务。文档收敛为工作流规范、操作指南及角色定义。

## User Stories

1. As a researcher, I want fixed logical roles with replaceable sessions, so that conversation changes do not change research ownership.
2. As a researcher, I want independent account/provider bindings, so that workers cannot silently borrow another account.
3. As a researcher, I want root to remain Codex, so that my research supervisor retains its existing tool environment.
4. As a researcher, I want background supervision after closing the frontend, so that research can progress without an open conversation.
5. As a researcher, I want pause to stop new dispatches, so that I can control further spending.
6. As a researcher, I want already started experiments to finish while paused, so that useful evidence is preserved.
7. As a researcher, I want completed deliveries recorded while paused, so that resuming does not lose results.
8. As a researcher, I want runtime completion separated from evidence acceptance, so that process success does not become a scientific claim.
9. As a researcher, I want root to accept or reject evidence with a reason, so that next tasks follow research judgment.
10. As a researcher, I want bounded automatic recovery, so that transient supervisor failures do not end research or loop forever.
11. As a researcher, I want recovery to reconcile existing executions, so that a crash cannot launch a duplicate experiment.
12. As a researcher, I want uncertain launch outcomes to stop automatic retries, so that unknown work is not duplicated.
13. As a researcher, I want a single dispatch owner, so that frontend reconnects and supervisor restarts cannot race.
14. As a researcher, I want each task to state a decision test, budget, stop conditions and deliverables, so that work is bounded and reviewable.
15. As a researcher, I want resource permissions enforced before dispatch, so that a worker cannot exceed its role or campaign limits.
16. As a researcher, I want to query factual backend status, so that stale local mirrors do not mislead me.
17. As a researcher, I want a finite campaign dispatch/time budget, so that background operation cannot spend indefinitely.
18. As a researcher, I want old experiments to finish under their old owner, so that migration does not interrupt research.
19. As a researcher, I want an explicit cutover guard, so that two workflow generations cannot dispatch concurrently.
20. As a maintainer, I want one workflow entry and an operator guide, so that new conversations need not reconstruct scattered rules.
21. As a maintainer, I want required artifact and process observers retained separately, so that retiring session control does not erase evidence tools.
22. As a maintainer, I want execution state and account credentials kept local, so that research history stays portable.
23. As a maintainer, I want real shell smoke validation without model spending, so that the adapter is verified before live use.
24. As a researcher, I want unavailable or malformed provider results reported clearly, so that missing evidence never looks like success.

## Implementation Decisions

- A local Markdown issue is the current spec publication surface; no remote publish or push is included.
- One public workflow command boundary provides status, pause, resume, dispatch, acceptance, reconcile, and supervision. Legacy control remains explicit for draining tasks.
- Stable role permissions remain one tracked definition; machine-local JSON binds roles to workspace, runtime, CODEX_HOME, provider identity and isolated backend stores. Root runtime must be Codex.
- External execution state belongs to Orchestrator; local durable state contains only supervision intent, research delivery acceptance, launch intent and execution association. It does not copy a complete execution state machine.
- Backend identity is sealed against provider/account/config changes; rebinding requires a new store. Secrets are never printed.
- Launch intent is committed before calling the backend. Unique names allow reconciliation after interrupted launch. An uncertain outcome is never automatically relaunched.
- A process lock owns supervision; transactional control serializes pause and dispatch. Background root emits one validated action at a time. Prompt instructions do not substitute for command validation.
- Root decisions are accept, reject, dispatch or idle. Completed delivery must be handled before assigning the same role another task. Idle decisions wait for a real state change or user resume; no repeated model heartbeat.
- The user explicitly starts background operation with finite task/time/GPU bounds and confirms that old dispatch owners are disabled. An active experiment remains allowed to drain. This branch does not modify live runtime bindings or stop running processes.
- Recovery preserves existing execution association, with finite consecutive retries and backoff. Budget or identity failures cannot silently fallback to another account.
- Default implementation uses bounded process tasks rather than perpetual native Goals. Fresh Codex root turns receive repository context and durable accepted deliveries; native thread continuity is not required for research identity.
- Deprecated contracts move out of the default reading path. Compatibility code needed by old jobs is marked drain-only until old jobs finish; it cannot be deleted while dependencies remain.

## Testing Decisions

- Test externally observable behavior through the workflow CLI and its public JSON output; these are the high-level seams discussed during design. Do not assert private database tables or function call order.
- Use executable fake backend processes at the external runtime boundary, with independent account identity and known task results, to test deterministic crash, pause and recovery cases without invoking models.
- Cover actual dispatch, role/store isolation, sealed identity, pause with completion, acceptance versus execution, supervisor continuation, duplicate-owner exclusion, crash reconciliation, uncertain outcomes, invalid decisions and budget enforcement.
- Existing workflow command tests and temporary-workspace fixtures provide prior art; preserve legacy runtime behavior required for old task drain.
- Run a real Orchestrator shell smoke, static type checking, focused tests, and the repository suite once. Report unavailable research dependencies explicitly. Real-account smoke is a separate deployment gate, never inferred from fake account tests.

## Out of Scope

Changing research objectives or scientific claims; migrating running model sessions or experiments; changing external read-only projects; launching GPU training or real model campaigns during implementation; adding dynamic long-term roles; rewriting provider protocols; remote push; production cutover without live-account isolation validation.

## Further Notes

All nine conversation decisions are resolved. The user's instruction to synthesize and directly implement confirms the overall direction and supersedes another design interview. Technical validation may expose a material backend gap; in that case preserve the old system and report concrete evidence.
