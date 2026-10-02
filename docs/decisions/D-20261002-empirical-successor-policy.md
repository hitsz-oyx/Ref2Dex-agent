# Preserve observed successor tuples for actual physical-model actor learning

Question: is there enough task-value headroom to justify a different physical
representation after deterministic actor and corrected derivative failure?

Evidence: fixed actual8-step successor V Brier.08441 versus directQ.12246
and learned Cm composition.12047 on all reused618episodes; all prospectively
frozen headroom/gap gates pass. This does not identify Jensen bias or show that
any feasible current-state model can access the oracle's information.

Action: learn probabilities over complete observed131physical successor
tuples from FIT603/604 only, using physical observations and a fixed kernel
observation score. Integrate the existing held-option continuation V over
support instead of feeding it a coordinatewise average. Train actual Cm/on
and physical-action-removed actors, then execute fresh623/624native evaluation
against those, P0 and the already trained strong direct-Q actor. Do not rerun
the control's1000valid optimizer steps. Keep its same data/init/update budget.
No explicit motion index/support mask or extra actor observation is supplied.

Support's observed tuples avoid synthesizing independently averaged physical
features; they do not guarantee feasible successors conditional on a new
query, correct physical uncertainty or distribution calibration. Known future
plan/clock remains analytic. Leave own episode out of support in both model
and actor fitting; no TEST/618donor or return-filtered support selection.

Cost<=1200s/1GiB, one idle GPU,3000new physical-model/2000new actor steps,
1536reusedFITtrajectories and1536newEVALtrajectories. Sourcepretraining costs
are reported separately. Actual unchanged5ppALLcontrols/eachseed/motion1 gates
choose the outcome; no intervening forecast/likelihood gate. Positive triggers
formal matched Validation, expected-value/mean mechanism control and novelty
assessment; negative closes this atomic/kernel representation without width,
kernel, temperature, support-size, epoch, penalty, seed or checkpoint scans.
Mission and external/resource permissions unchanged. Generic kernels/world
models are established; no distinct-method or publication claim now.
