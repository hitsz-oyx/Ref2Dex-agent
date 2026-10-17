# WAM-v4 architecture contract

The model keeps the existing goal-conditioned shared encoder and separates four
semantic levels:

- `G_task`: the final task goal supplied to WAM;
- local 24-step goal: a short-horizon target that a future scheduler will produce;
- `tau`: hand trajectory forecast over 24 WAM frames;
- `C`: object-centric interaction forecast used by a downstream executor.

The current v4 model still consumes the existing `goal_object_delta` as the
short-horizon goal input. A scheduler can replace that field with an analytic
or learned local goal without changing the model head contract. The optional
reference path should later map `Ref -> C_ref`; it should not be concatenated to
raw WAM tokens in this task.

The first matched decision experiment after an engineering smoke is:

```text
same executor: state + tau       versus       state + tau + C
```

with the same data, seed and WAM checkpoint. This is a Decision probe for
whether interaction prediction carries control-relevant information.
