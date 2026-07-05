# Why this folder is excluded from the reported results

These scripts were an early attempt, in a prior version of this project, to extend the
spectral analysis to a natural-language corpus (TinyStories, Eldan & Li, 2023).

They are included here **only for transparency**, not as supporting material for any
claim in the paper. During revision, we determined that `tinystories_hvp_train.py`
(the intended "flexible baseline" condition) trained on **randomly generated token
sequences** (`dummy_data_generator()`), not on real TinyStories text, while
`tinystories_hvp_ablated.py` (the "semantic collapse" ablation) did correctly load
real, pre-processed TinyStories text via `tinystories_data_ablated.py`. As a result,
the comparison between the two conditions, as originally reported, compared real text
against random noise, not a valid manipulation of semantic redundancy.

This issue was caught during the current revision, before submission, and the
corresponding analysis was removed from the paper entirely rather than corrected
within the scope of this revision (see the "Task domain" paragraph in the paper's
Limitations section). We are keeping the original, flawed scripts here, unmodified,
for anyone who wants to verify this finding directly or build a corrected version of
this experiment.

**None of the files in this folder were used to produce any result, table, or figure
in the paper.**
