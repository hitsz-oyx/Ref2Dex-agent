# Contact-response manuscript

- `manuscript.tex`: complete English working draft with the actual two Probe results.
- `figures/contact_response.pdf`: standalone vector figure, copied from corrected analysis-v2.
- `manuscript.pdf`: locally rendered review copy; not a journal submission.

The source is a normal LaTeX article. No TeX engine is installed on this host.
`scripts/export_contact_response_paper.py` renders a PDF review copy using the system
Python's existing ReportLab package, verifies the manuscript's numeric tables against
recorded experiment JSON, and records source/input hashes in an export manifest.

Physical run: `src/task/CmResidual/research/contact_response/output/P-20261001-contact-response-resolution-r1`.
Learning run: `src/task/CmResidual/research/contact_response/output/P-20261001-differential-response-learning-r1`.
These local ignored outputs contain all original evidence; the manuscript is an interpretation.

Both fixed Probe labels are UNPROMISING. There is no policy utility, hardware,
object generalization, or top-journal readiness claim. Source and rendering are kept
so later evidence can be incorporated without inventing results.
