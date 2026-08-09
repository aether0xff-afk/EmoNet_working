# v5.10 batched execution note

Status: committed **before any successful v5.10 scientific benchmark result**.

The v5.10 scientific protocol is unchanged:

- same frozen v5.7/v5.8 equations and seeded matrices;
- same vector worlds `101, 211, 307`;
- same recurrent seeds `7, 13, 21, 42, 100`;
- same two hidden-prior tasks;
- same delays `0, 1, 3`;
- same 80 pairs per task/world with 50 train / 30 test;
- same visible `A B C D` window and common `Z`;
- same intact / fast-reset / slow-reset / both-reset / opposite-history controls;
- same leave-one-vector-world-out ridge protocol;
- same preregistered gates.

Only execution is changed: independent samples are stacked into a matrix batch and the same equations are evaluated with matrix multiplications instead of one Python sample at a time.

The scalar reference helpers remain in the benchmark module for tests/debugging.

Mandatory equivalence tests compare the batched implementation with the frozen scalar v5.7/v5.8 implementation event by event, including:

- residual vectors;
- every tick of the fast neural trace;
- final fast state;
- slow EMA state;
- v5.8 adaptation state;
- intervention reset semantics.

The accepted numerical tolerance is at most a few `1e-6`, reflecting only floating-point matrix evaluation order. No result-based parameter or sample change is allowed.

Previous scalar/debug workflow attempts that did not produce a valid scientific result are excluded from v5.10 interpretation. The first successful batched run that passes all equivalence and benchmark-construction tests is the formal v5.10 result.
