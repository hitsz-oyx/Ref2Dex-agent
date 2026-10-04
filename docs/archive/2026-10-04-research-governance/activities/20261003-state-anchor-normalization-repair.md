# Prefit frozen-base arithmetic repair

State-anchored transport r1 atf6ec05f FAILED before any newoptimizerupdate,
21.673869s. CPU smoke passed and two base prediction files were written;
strict frozenheld prediction equality stopped the fit before constructing
newhead checkpoints. All owned parent/children absent; failedlog/manifest/
basefiles retained, no model fits to replay. This is INVALID engineering
execution, not a scientific negative result and not an extra valid Probe.

Reproduction: identicalwinner/all6144rows, maxcoeff3.57628e-7,
score8.34465e-7, worldprediction5.79451e-9m. Newbaseline inputs normalized
onCPU beforeGPU whereas originalcoupling normalized onGPU; float32 division
arithmetic differs. Repair freezes the original GPUmean/std subtraction/
division and state-token selection path. No changednormalizer statistics,
modelweights, architecture/loss/schedule/labels/gates or relaxed1e-12m equality.
Newrun r2 from repaired commit, uniqueoutput/fullprotection; r1 retained.
Goal ACTIVE, paper NOTREADY.
