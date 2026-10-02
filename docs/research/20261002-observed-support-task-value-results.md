# Current observed support task value: paired gate fails

Frozen3635750, executed3a945db. Fresh597/598,1536 trajectories,63682FIT and
63048TEST rows/576TEST episodes. Two matched1500-update networks;3000GPU
optimizer steps. COMPLETED / UNPROMISING,319.342s/536427817bytes within600s/
768MiB. All protected inputs unchanged; all owned phases terminal.

| Value input | Held-out Brier |
| --- | ---: |
| Compact current72 | 0.0540529346 |
| Current72 + measured SDK support80 | 0.0531833419 |
| FIT-only motion/time control | 0.0806563586 |

Support's descriptive1.6088% compact gain passes the1% magnitude gate but
its paired95% interval[-0.0020704497,+0.0003227968] crosses zero. Motion/time
passes both gates; retain overall UNPROMISING without extra seeds, updates,
subgroups or threshold changes. No basis to promote this exact support-value
interface as the main critic fix, nor a universal claim that support is useless.

Both native audits pass. Separate independent raw context/history and Torch64
SDK pose/force/flow reconstruction is exact on both cohorts (maxerror0).
All TEST NumPy network forwards pass6.474e-7/1.688e-6; FIT statistics explicitly
reuse the float32 decoder after independent reconstruction. Controls,192-group
bootstrap and gates rebuilt; no optimizer replay. ModelSHA
`1d94da3d93fd0a0e2f88bec4d4f2a4fcc4980c7bfe5365dce4195171f90fc534`;
predictionSHA`4763d6096af63c3132df30f6c490a2df3eea26a13ab283719bbf10766952bcab`.

Current state information is distinct from useful candidate-action differences.
Stop this exact value predictor and return to directly executed task opportunity
under matched initial conditions and a common pre-intervention policy. This
does not adopt any failed physical predictor, establish Cm benefit or novelty,
or change the physical105 endpoint.
