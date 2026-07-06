# Experimental / retired code

The scripts in this folder implement a natural-language validation experiment
(TinyStories corpus) that was part of an earlier draft of this project. They
are kept here for transparency but are **not used** to produce any result
reported in the associated paper.

## Why this was retired

During revision, we discovered that `tinystories_hvp_train.py` (the intended
"flexible baseline" run) trains on synthetic random token sequences generated
by `dummy_data_generator()`, not on the actual TinyStories text. Only the
semantic-collapse ablation (`tinystories_hvp_ablated.py`, via
`tinystories_data_ablated.py`) correctly loads real (though lexically
collapsed) TinyStories text. This means the comparison between the two was
not a valid like-for-like ablation, and any spectral trajectory previously
attributed to "natural language" in earlier drafts should not be trusted.

Additionally, both scripts were configured for only 200 optimization steps,
which cannot support the extended, multi-phase spectral trajectory (including
a curvature peak and a transient negative-eigenvalue region) described in an
earlier draft's natural-language section.

## Status

This code is left as-is, unfixed. Extending the spectral analysis to a real
natural-language corpus, with a corrected, matched-duration experimental
design across both the baseline and the ablated condition, is noted as an
open direction for future work (see the Limitations section of the paper).

If you plan to pick this up: the immediate fix is to replace the
`dummy_data_generator()` in `tinystories_hvp_train.py` with the real
DataLoader already implemented correctly in `tinystories_data.py`, and to
extend `max_steps` in both training loops to be long enough for the
grokking-like transition to fully unfold and stabilize, matching the
duration used for the ablated condition.
