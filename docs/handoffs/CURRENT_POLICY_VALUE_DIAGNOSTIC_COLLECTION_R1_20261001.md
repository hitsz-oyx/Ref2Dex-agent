# Current-policy value diagnostic collection R1 — FAILED

## Terminal status

`T-20261001-current-policy-value-diagnostic-collection-r1` is **FAILED** due to a bounded runtime engineering exception. This is not a scientific negative and no value, utility, convergence, or bias claim is made.

The required diagnostic card was written first at `docs/experiments/probes/P-20261001-current-policy-value-diagnostic.md`. It states the three decision questions: distinguish frozen-critic lambda fit insufficiency from complete Monte Carlo distribution difference; use that distinction to choose the next adaptation, target, or representation check; and use the cheapest two-e420 complete-first-episode diagnostic. The HF08 utility slot remains closed.

## CPU and input checks

- Broker lease: `C1s4FDrUQ9PZgqVMLE_WS5Ig`, 3900 seconds requested.
- CPU preflight used `/home2/wyy/miniconda3/envs/graspenv/bin/python`, `PYTHONDONTWRITEBYTECODE=1`, and support-local `TORCH_EXTENSIONS_DIR`/`XDG_CACHE_HOME`.
- Preflight returned `READY_FOR_ROOT_RUNTIME_TASK`, `isaac_imported=false`, `gpu_started=false`, `collection_started=false`, with the 4-row/2-episode reset-boundary fixture passing.
- Accepted implementation hashes remained exact: `collect_current_policy_value.py` `4c5b4687491d9e784ca2ea3f271162114834965990f8c58c4a691aebbc0216de`; `run_current_policy_value_environment.py` `ffe1c061dbd9accb947e248a82cf7c01c2e6afa9229e2949080b9cfdeee474a7`.
- Source runner hash stayed `1fc914f75b5579078eea533660a57f2c820414995da18cd7304858054ab26a57`; environment and training hashes stayed `f66b3d16d685678ec24e757972b1521de96df6d51970a7abca2185fb376a37b9` and `a9b46c7bad03df3518cca07734fcb1a9d78f448ccf951ed119c999ff97eebf38`.
- Worker HEAD was `39d0615db896ee66befa41504f605e87a6f9d136`; root accepted implementation commit was `ff48f6d61fb4e63d2c3525896237971512bfb576`; source HEAD observed during preflight was `f3d65e153783be16de10c3068a5acb0435754428`.

## GPU and runtime evidence

GPU 4 matched the requested UUID `GPU-0606f00a-d9d0-3a00-5b49-c9e747b77307`, reported 2 MiB, and had an empty compute-PID list immediately before launch. The same target remained at 2 MiB with no compute owner after termination. Other GPU owners were not touched. Evidence is in `gpu_preflight_s286.json` and `gpu_post_s286.json`.

The single owned process group was launched for s286 only. Isaac Gym imported and the environment began construction; no first episode, transition row, results file, or shard was emitted. At `2026-10-01T05:48:48.942461+00:00`, the frozen source path failed with:

```text
ValueError: string is not a file: `dexplore/data/assets/mjcf/objects/airplane/airplane.obj`
```

The failure occurred because the frozen source's relative `assetRoot` was resolved from the worker cwd. The partial `runtime_manifest.json` is `run_status=FAILED` and preserves the traceback provenance. Per the task stop condition, s287 was not started and no cwd or tool change/retry was attempted.

The exact attempted cwd was `/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent-rl`. The source-root asset exists at `/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/third_party/DExplore/dexplore/data/assets/mjcf/objects/airplane/airplane.obj`; the worker-cwd relative target does not exist. Native r7 manifests do not record `cwd` (the field is absent), while their successful logs resolve the same relative `dexplore/data/assets/...` path. This records the observed cwd difference without treating it as a scientific result.

## Artifacts and limits

- Output root contains only `runtime_root.json` and the failed s286 `runtime_manifest.json`; rows: `0`; complete episodes: `0`; `results.json`: none; transition shards: none.
- Support root retains preflight, GPU ownership records, process-start record, log, and the support-local `gymtorch` build. Combined output/support usage was about 3.56 MB, under the 2 GiB ceiling; runtime elapsed was about 28.14 s of the shared 1200 s budget.
- Exact artifact hashes, modes, sizes, input/model hashes, and failure manifest are in the companion JSON handoff. Existing tools, source code, old reports, main, and shared Git metadata were not modified.

CONTROL 249 was applied before terminal handoff. This handoff records an engineering failure only. A separate cwd-only engineering repair and a new governed collection task are required before any scientific analysis or `agent_cm` value audit.
