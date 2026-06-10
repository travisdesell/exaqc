# Cross-stage analysis (Stages A → B → C, plus D and E)

Companion to the auto-generated `all_stages_compare.md`. Same data, prose interpretation.

## TL;DR

1. **Each level of staged evolution adds accuracy, with diminishing returns.**
   Mean test_acc: A 0.707 → B 0.794 → C 0.862. Roughly +0.087 from evolving the ansatz (A→B) and another +0.068 from evolving N (B→C).
2. **The two Stage-D encoders (`reupload_euler`, `linear_proj`) dominate everything else.**
   Their Stage-B mean test_acc (0.983, 0.972) is ~0.08 above the next-best encoder (`learned` at 0.903) and ~0.4 above `fixed_amplitude`. Stage D's contribution was the single biggest accuracy improvement we shipped.
3. **Full co-evolution (Stage E) underperforms staged (Stage C).**
   Stage E mean 0.790 vs Stage C mean 0.862, and Stage E loses on every dataset by 0.12 to 0.30. Evolving everything at once is harder than evolving things in layers.
4. **Champions are all Stage B with universal encoders, all at N≤8.** No champion needed a learned encoder structure or more than 8 qubits.

## Stage-level summary

| stage | n cells | mean acc | best acc | worst acc |
|---|---|---|---|---|
| A | 64 | 0.707 | 0.967 | 0.306 |
| B | 96 | 0.794 | 1.000 | 0.333 |
| C | 24 | 0.862 | 1.000 | 0.528 |
| E | 4  | 0.790 | 0.851 | 0.700 |

Stage C has the highest *mean*; Stage B has more cells at the ceiling (1.000) but also more low-tier cells dragging its mean. Stage E sits between A and B despite being the most ambitious stage.

## Per-encoder accuracy across stages

Mean test_acc by encoder, by stage:

| encoder | A | B | C |
|---|---|---|---|
| `fixed_basis`     | 0.678 | 0.618 | 0.654 |
| `fixed_angle`     | 0.785 | 0.753 | 0.844 |
| `fixed_amplitude` | 0.493 | 0.534 | 0.832 |
| `learned`         | 0.871 | 0.903 | 0.882 |
| `linear_proj`     |   —   | 0.972 | 0.968 |
| `reupload_euler`  |   —   | 0.983 | 0.993 |

Reading row-by-row:
- **Fixed encoders barely benefit from ansatz evolution** (A→B mostly flat or negative for `fixed_basis`/`fixed_angle`). The encoder is the bottleneck; a better ansatz on top of a weak encoder doesn't move much.
- **`fixed_amplitude` exploded from C** (0.493 → 0.534 → 0.832). The reason is visible in the evolved-N column: when the encoder is amplitude embedding, the natural register size is `ceil(log2(D))` and evolved-N let Stage C find it (e.g. iris shrunk to N=2 = log2(4)). Stages A and B were stuck sweeping N ∈ {4,6,8,10} and never tried small N.
- **`learned` saturated by Stage B.** Adding evolved N (Stage C) didn't help (0.903 → 0.882). The encoder picks up enough capacity from its 4N trainable params that further structural search doesn't add much.
- **The two universal encoders are the new ceiling**: `reupload_euler` is 0.983–0.993 across both stages it appears in.

## Champion per dataset

| dataset | stage | encoder | N | test_acc | test_loss | gates |
|---|---|---|---|---|---|---|
| iris          | B | `learned`        | 4 | 1.000 | 0.185 | 3 |
| wine          | B | `linear_proj`    | 4 | 1.000 | 0.145 | 5 |
| seeds         | B | `reupload_euler` | 6 | 1.000 | 0.188 | 2 |
| breast_cancer | B | `reupload_euler` | 4 | 0.974 | 0.133 | 2 |

Every champion is in **Stage B**. Three datasets hit 1.000 there. Stage C ties 1.000 on iris/wine/seeds but loses on breast_cancer (0.974 vs 0.974 — a true tie there). Stage E never wins. **Lesson: when the encoder is strong, evolving the ansatz suffices; you don't need to also evolve N.**

## A → B → C progression on `learned`

Most-comparable apples-to-apples slice (same encoder, increasing evolution scope):

| dataset | A | B | C | A→B Δ | B→C Δ |
|---|---|---|---|---|---|
| iris          | 0.967 | 1.000 | 0.967 | +0.033 | -0.033 |
| wine          | 0.833 | 0.861 | 0.833 | +0.028 | -0.028 |
| seeds         | 0.881 | 0.976 | 0.833 | +0.095 | -0.143 |
| breast_cancer | 0.895 | 0.939 | 0.895 | +0.044 | -0.044 |

A→B is always positive (evolving the ansatz helps). **B→C is always negative** — Stage B's manual N sweep beat Stage C's evolved N on every dataset with `learned`. Two readings:
- Pessimistic: register evolution isn't worth it for already-strong encoders.
- Optimistic: 80 genomes wasn't enough budget for evolution to find the best N when the manual sweep got to try 4 separate values directly. Multi-seed reruns with larger budgets would distinguish.

## Evolved-N histogram (Stage C)

| N | count of best cells |
|---|---|
| 2 | 2 |
| 4 | 14 |
| 5 | 5 |
| 6 | 3 |

The mode is N=4 (no change from initial). Evolution preferred staying small. Only 2 cells shrank (both `iris/fixed_amplitude → N=2` and `wine/fixed_angle → N=2`), and none grew past 6. **The narrative "more qubits = better" doesn't hold** for these classification tasks at this evolution budget.

## Stage E underperformance

| dataset | best of A/B/C | Stage E | Δ |
|---|---|---|---|
| iris          | 1.000 | 0.700 | -0.300 |
| wine          | 1.000 | 0.778 | -0.222 |
| seeds         | 1.000 | 0.833 | -0.167 |
| breast_cancer | 0.974 | 0.851 | -0.123 |

Stage E (`enc_ry/rx/rz` feature-dependent gates evolved jointly with the ansatz and register) loses on every dataset, by 0.12 to 0.30. Hypothesis:
- The structural search space is much larger (encoder + ansatz + N all simultaneously), and 120 genomes isn't enough for evolution to find good co-adapted (encoder, ansatz) pairs.
- The Stage E encoder gates each use a single `a·x[q%D] + b` per gate — strictly less expressive than the universal encoders (full linear projection or Euler-decomposed reupload).

Worth testing: rerun Stage E with `n_genomes=300+` and the same gate pool extended to include richer encoder gates, to see if the gap closes.

## Key takeaways

- **Picking the right encoder beats evolving the ansatz.** Going from `learned` to `reupload_euler` gave bigger gains than every level of evolution added to `learned`. Future work should prioritize the encoder space.
- **Evolved N has narrow value.** It rescues amplitude embedding (which has a "right" N = log2(D)) and otherwise produces small changes. Manual sweep over a few values is competitive.
- **Stage E's full co-evolution doesn't pay off at this budget.** Staged decomposition (find encoder → evolve ansatz → evolve N) is more sample-efficient than co-evolving everything.
- **`fixed_basis` is the only encoder that ansatz evolution actively hurts** — its A→B delta is negative across most cells. Probably because the basis encoder produces one of a small number of computational-basis states and the ansatz mostly destroys that structure.
- **All four datasets are solvable at 1.000 (or 0.974 for BC) by Stage B + universal encoder.** The harder open questions are about *parameter efficiency* and *robustness* (multi-seed error bars), not raw accuracy.

## Open questions / next steps

1. **Multi-seed reruns** to put error bars on these numbers. Stage A's original report flagged this as ~17 hr of compute; with the universal encoders being clearly best, we could prune to just `learned`, `linear_proj`, `reupload_euler` and run 3-5 seeds.
2. **Parameter efficiency frontier.** Champions all use small gate counts (1–5 enabled gates) but the encoders themselves have very different parameter counts (`learned`: 4N, `linear_proj`: 2·N·(D+1), `reupload_euler`: 6·N·(D+1)). Worth plotting acc vs total trainable params to see if `learned` is Pareto-dominated.
3. **Stage E budget bump.** Re-run with 300+ genomes to see if co-evolution catches up.
4. **Fairness control for `learned`.** Stage A note flagged that `learned` has +4N extra params vs fixed encoders. With universal encoders even further ahead in param count, the same control question applies: would `fixed_*` plus extra ansatz layers close the gap?
5. **Stage C low-N exploration.** Only 2/24 cells shrunk; both to N=2 and both improved. Worth a targeted experiment starting at N=8 or N=10 to see if evolution shrinks more aggressively.
