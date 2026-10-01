# P-20261002-observation-hold-aggregation

Class Decision; label pending. Run_id r1, isolated worktree contact-response.
Design frozen in D-20261002-observation-hold-aggregation.md. Distinguishes
teacher-state-only regression from one on-policy corrective-label aggregation.
Old baseline failure remains UNPROMISING. New collection513/514; new test515/516.
All58176rows retained, fixed2000updates, final only, no Cm or PPO.
Physical105 gate: motion1>=50%pooled,>=25%eachseed. Source actor bootstrap
never acts; all control comes from our trained policy. Collection expert labels
queried offline from current state and available reference; queries not executed.
Model training/inference and native simulation GPU; pure label auditing CPU.
Stop on provenance drift, query saturation, nonfinite data,1800s or1GiB.
