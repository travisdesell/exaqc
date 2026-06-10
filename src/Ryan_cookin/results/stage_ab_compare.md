# Stage A vs Stage B comparison

Cells show `A_acc / B_acc (Δ)` where Δ = B − A. Bold means Stage B wins; italic means Stage A wins; plain means tie (|Δ| < 1e-6). `gates` is the enabled gate count in Stage B's best evolved genome — for reference, Stage A's hand-built ansatz always has `2N` rotation gates + `2(N-1)` CNOTs.


## iris

| encoder | N=4 | N=6 | N=8 | N=10 |
|---|---|---|---|---|
| `fixed_angle` | _0.733 / 0.600 (-0.133) g=3_ | 0.733 / 0.733 ( 0.000) g=3 | _0.733 / 0.633 (-0.100) g=7_ | _0.733 / 0.667 (-0.067) g=4_ |
| `fixed_amplitude` | 0.667 / 0.667 ( 0.000) g=3 | **0.333 / 0.400 (+0.067) g=4** | 0.333 / 0.333 ( 0.000) g=3 | 0.333 / 0.333 ( 0.000) g=6 |
| `fixed_basis` | 0.767 / 0.767 ( 0.000) g=5 | _0.700 / 0.333 (-0.367) g=7_ | _0.767 / 0.667 (-0.100) g=4_ | 0.767 / 0.767 ( 0.000) g=5 |
| `learned` | **0.933 / 1.000 (+0.067) g=3** | **0.933 / 0.967 (+0.033) g=5** | 0.967 / 0.967 ( 0.000) g=12 | _0.933 / 0.833 (-0.100) g=3_ |


## wine

| encoder | N=4 | N=6 | N=8 | N=10 |
|---|---|---|---|---|
| `fixed_angle` | 0.806 / 0.806 ( 0.000) g=9 | _0.806 / 0.667 (-0.139) g=2_ | _0.806 / 0.472 (-0.333) g=3_ | _0.806 / 0.500 (-0.306) g=4_ |
| `fixed_amplitude` | **0.806 / 0.833 (+0.028) g=7** | **0.306 / 0.389 (+0.083) g=3** | **0.389 / 0.611 (+0.222) g=5** | 0.389 / 0.389 ( 0.000) g=9 |
| `fixed_basis` | _0.611 / 0.472 (-0.139) g=4_ | 0.472 / 0.472 ( 0.000) g=3 | **0.472 / 0.583 (+0.111) g=6** | _0.611 / 0.472 (-0.139) g=3_ |
| `learned` | **0.806 / 0.833 (+0.028) g=0** | _0.833 / 0.806 (-0.028) g=1_ | **0.806 / 0.861 (+0.056) g=3** | **0.778 / 0.833 (+0.056) g=0** |


## seeds

| encoder | N=4 | N=6 | N=8 | N=10 |
|---|---|---|---|---|
| `fixed_angle` | **0.738 / 0.857 (+0.119) g=3** | **0.738 / 0.905 (+0.167) g=1** | **0.738 / 0.833 (+0.095) g=2** | **0.738 / 0.905 (+0.167) g=3** |
| `fixed_amplitude` | _0.667 / 0.619 (-0.048) g=3_ | **0.333 / 0.595 (+0.262) g=1** | 0.333 / 0.333 ( 0.000) g=8 | 0.333 / 0.333 ( 0.000) g=3 |
| `fixed_basis` | 0.667 / 0.667 ( 0.000) g=3 | _0.667 / 0.595 (-0.071) g=5_ | _0.667 / 0.429 (-0.238) g=7_ | _0.667 / 0.595 (-0.071) g=8_ |
| `learned` | 0.833 / 0.833 ( 0.000) g=3 | **0.833 / 0.976 (+0.143) g=1** | **0.881 / 0.976 (+0.095) g=2** | **0.857 / 0.952 (+0.095) g=1** |


## breast_cancer

| encoder | N=4 | N=6 | N=8 | N=10 |
|---|---|---|---|---|
| `fixed_angle` | 0.860 / 0.860 ( 0.000) g=6 | **0.868 / 0.877 (+0.009) g=7** | **0.860 / 0.877 (+0.018) g=3** | _0.860 / 0.851 (-0.009) g=3_ |
| `fixed_amplitude` | _0.728 / 0.719 (-0.009) g=4_ | _0.675 / 0.658 (-0.018) g=2_ | **0.632 / 0.658 (+0.026) g=4** | **0.632 / 0.675 (+0.044) g=5** |
| `fixed_basis` | 0.754 / 0.754 ( 0.000) g=3 | **0.754 / 0.763 (+0.009) g=4** | 0.754 / 0.754 ( 0.000) g=4 | **0.754 / 0.789 (+0.035) g=5** |
| `learned` | **0.877 / 0.886 (+0.009) g=4** | **0.895 / 0.904 (+0.009) g=6** | **0.877 / 0.939 (+0.061) g=4** | _0.895 / 0.886 (-0.009) g=6_ |


## Summary

- Win counts (of 64 cells): **B wins 27**, A wins 20, ties 17

### Mean Δacc by dataset (B − A, positive = B better)

| dataset | n | mean Δacc |
|---|---|---|
| iris | 16 | -0.044 |
| wine | 16 | -0.031 |
| seeds | 16 | +0.045 |
| breast_cancer | 16 | +0.011 |

### Mean Δacc by encoder

| encoder | n | mean Δacc |
|---|---|---|
| fixed_angle | 16 | -0.032 |
| fixed_amplitude | 16 | +0.041 |
| fixed_basis | 16 | -0.061 |
| learned | 16 | +0.032 |

### Win counts by encoder (B vs A)

How often Stage B's evolved ansatz beats Stage A's fixed RY-CNOT chain, holding the encoder fixed. The `learned` row answers "does learn + evolution beat learn + standard ansatz?".

| encoder | B wins | A wins | ties | n |
|---|---|---|---|---|
| fixed_angle | **6** | 7 | 3 | 16 |
| fixed_amplitude | **7** | 3 | 6 | 16 |
| fixed_basis | **3** | 7 | 6 | 16 |
| learned | **11** | 3 | 2 | 16 |

### Best cell per dataset

| dataset | Stage A best (cell, acc) | Stage B best (cell, acc, gates) |
|---|---|---|
| iris | `learned` N=8, acc=0.967 | `learned` N=4, acc=1.000, gates=3 |
| wine | `learned` N=6, acc=0.833 | `learned` N=8, acc=0.861, gates=3 |
| seeds | `learned` N=8, acc=0.881 | `learned` N=6, acc=0.976, gates=1 |
| breast_cancer | `learned` N=6, acc=0.895 | `learned` N=8, acc=0.939, gates=4 |
