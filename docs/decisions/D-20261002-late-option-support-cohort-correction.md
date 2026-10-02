# Correct the source cohort identity; preserve original support audit

Support-r1 labelled arms1/2/3 as random. Original collector and matched_inputs
show arms0/1are zero options, arm2Gaussian andarm3its antithetic opposite.
Recorded raw12 confirms arm1exactzero. Counts/labels for all individual arms are
valid, but the pooled random label is incorrect. Original r1output and863e792
code remain preserved; no native or optimizer work is repeated.

Correct summaries only: separate arm0P0, arm1duplicateP0, pooled zero0/1 and
random2/3. Verify ALLraw options against independentseed+18000draw and antithetic
assignment. Original >=8motion0success/transient gate and source episodes,
labels/window and decision logic unchanged. No statistical independence of
antithetic actions or exact state pairing is assumed.

Separate r2purelabel run<=60s/16MiB CPU with both original source and r1artifacts
protected bySHA. Current route until correction is provisional; no policy benefit
or task infeasibility claim from r1. This is an engineering identity correction,
not a new threshold/action/horizon/seed experiment.
