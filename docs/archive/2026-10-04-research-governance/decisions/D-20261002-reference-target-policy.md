# Decision: reference-relative absolute-target actor

Scratch BC and one aggregation both fail0/192. Preserve both frozen outcomes.
Do not run another aggregation or tune their update budgets. Teacher and actor
already share the reference plan. Native wrist commands are incremental, so
regression error can accumulate; learned absolute-target output with explicit PD
inverse is a distinct structural hypothesis. This is not a claim that the
observed failures have a uniquely identified cause.

Train a scratch512/256/128ReLU/Tanh model on all original510teacher19392rows,
same causal70context, FIT-only mean/std, no511/512/515/516test reuse. Outputs
are reference-relative target residuals scaled [.02m]*3+[.10rad]*3+[.40rad]*12.
Labels are saved teacher PD target minus current planned reference, divided by
these scales. Hardware-null dependent coordinates are canonical0. Desired wrist
target is planned q plus learned residual, independent of the previous command;
finger parent bounds/coupling and thumb yaw are enforced natively. The exact
PD inverse uses current q to produce normalized commands. No teacher fallback,
timed curl schedule or outcome test is used during learned-policy evaluation.
The reference plan remains available to both sides as in the prior designs.

One random seed736,2000Adamupdates,.001lr,batch512,unit residual MSE,gradclip10,
finalonly. New tests517/518,96each,202ticks, physical105sameprospective gate:
motion1>=50%pooled and>=25%eachseed, allmotions reported; strictforce75secondary.
Changing native output coordinates is established control practice, not a novel
Cm method. Only initializer feasibility can follow. If promising, prioritize
matched Cm-on/off policy training. If it fails, review the full baseline family
rather than expanding it with local tuning. One admittedGPU<=1800s/1GiB.
