# All-stages comparison (A vs B vs C vs E)

Progression of what gets *evolved* (instead of fixed/swept) across stages:

| stage | encoder | ansatz structure | N (qubits) |
|---|---|---|---|
| A | trained params | fixed (RY-CNOT chain) | swept |
| B | trained params | evolved | swept |
| C | trained params | evolved | **evolved** |
| E | **evolved** (feature-dep gates) | evolved | evolved |

## Best test_acc per dataset, across all stages

| dataset | Stage A | Stage B | Stage C | Stage E |
|---|---|---|---|---|
| iris | **0.967** (learned, N=8) | **1.000** (learned, N=4, g=3) | **1.000** (reupload_euler, N=4, g=5) | **0.700** (N=6, enc=4, ans=4) |
| wine | **0.833** (learned, N=6) | **1.000** (linear_proj, N=4, g=5) | **1.000** (reupload_euler, N=4, g=2) | **0.778** (N=6, enc=4, ans=2) |
| seeds | **0.881** (learned, N=8) | **1.000** (reupload_euler, N=6, g=2) | **1.000** (linear_proj, N=4, g=1) | **0.833** (N=5, enc=5, ans=3) |
| breast_cancer | **0.895** (learned, N=6) | **0.974** (reupload_euler, N=4, g=2) | **0.974** (reupload_euler, N=6, g=1) | **0.851** (N=6, enc=5, ans=2) |

## Mean test_acc by encoder, by stage (averaged over datasets and N)

| encoder | Stage A (mean acc) | Stage B (mean acc) | Stage C (mean acc) |
|---|---|---|---|
| fixed_basis | 0.678 | 0.618 | 0.654 |
| fixed_angle | 0.785 | 0.753 | 0.844 |
| fixed_amplitude | 0.493 | 0.534 | 0.832 |
| learned | 0.871 | 0.903 | 0.882 |
| linear_proj | — | 0.972 | 0.968 |
| reupload_euler | — | 0.983 | 0.993 |

## Stage E summary (encoder is part of the genome)

| dataset | test_acc | test_loss | best N | enc gates | ansatz gates |
|---|---|---|---|---|---|
| iris | 0.700 | 0.6437 | 6 | 4 | 4 |
| wine | 0.778 | 0.5417 | 6 | 4 | 2 |
| seeds | 0.833 | 0.3308 | 5 | 5 | 3 |
| breast_cancer | 0.851 | 0.3256 | 6 | 5 | 2 |

## Stage C: how N evolved (best genomes)

Reveals whether evolution preferred bigger or smaller registers. Stage C starts at N=4; values below 4 mean evolution shrunk the register, above 4 mean it grew.

| dataset/encoder | initial N | best N | delta |
|---|---|---|---|
| iris/fixed_angle | 4 | 5 | +1 |
| iris/fixed_amplitude | 4 | 2 | -2 |
| iris/fixed_basis | 4 | 6 | +2 |
| iris/learned | 4 | 4 | +0 |
| iris/linear_proj | 4 | 4 | +0 |
| iris/reupload_euler | 4 | 4 | +0 |
| wine/fixed_angle | 4 | 2 | -2 |
| wine/fixed_amplitude | 4 | 4 | +0 |
| wine/fixed_basis | 4 | 5 | +1 |
| wine/learned | 4 | 4 | +0 |
| wine/linear_proj | 4 | 5 | +1 |
| wine/reupload_euler | 4 | 4 | +0 |
| seeds/fixed_angle | 4 | 4 | +0 |
| seeds/fixed_amplitude | 4 | 4 | +0 |
| seeds/fixed_basis | 4 | 4 | +0 |
| seeds/learned | 4 | 4 | +0 |
| seeds/linear_proj | 4 | 4 | +0 |
| seeds/reupload_euler | 4 | 4 | +0 |
| breast_cancer/fixed_angle | 4 | 4 | +0 |
| breast_cancer/fixed_amplitude | 4 | 5 | +1 |
| breast_cancer/fixed_basis | 4 | 6 | +2 |
| breast_cancer/learned | 4 | 5 | +1 |
| breast_cancer/linear_proj | 4 | 4 | +0 |
| breast_cancer/reupload_euler | 4 | 6 | +2 |

### Stage C evolved-N histogram (across all cells)

| N | count |
|---|---|
| 2 | 2 |
| 4 | 14 |
| 5 | 5 |
| 6 | 3 |

## Stage E: evolved register sizes

| dataset | best N | enc gates | ansatz gates |
|---|---|---|---|
| iris | 6 | 4 | 4 |
| wine | 6 | 4 | 2 |
| seeds | 5 | 5 | 3 |
| breast_cancer | 6 | 5 | 2 |

## Champion per dataset (best test_acc across A, B, C, E)

| dataset | stage | encoder | N | test_acc | test_loss | gates |
|---|---|---|---|---|---|---|
| iris | B | learned | 4 | 1.000 | 0.1849 | 3 |
| wine | B | linear_proj | 4 | 1.000 | 0.1447 | 5 |
| seeds | B | reupload_euler | 6 | 1.000 | 0.1881 | 2 |
| breast_cancer | B | reupload_euler | 4 | 0.974 | 0.1328 | 2 |

## Stage-level summary (mean / best test_acc across all cells)

| stage | n cells | mean acc | best acc | worst acc |
|---|---|---|---|---|
| A | 64 | 0.707 | 0.967 | 0.306 |
| B | 96 | 0.794 | 1.000 | 0.333 |
| C | 24 | 0.862 | 1.000 | 0.528 |
| E | 4 | 0.790 | 0.851 | 0.700 |

## A → B → C progression on `learned` encoder (best cell per dataset)

Tracks the same encoder family across increasing evolution scope: Stage A (fixed ansatz, N swept) → Stage B (evolved ansatz, N swept) → Stage C (evolved ansatz AND N).

| dataset | A best (N) | B best (N) | C best (evolved N) | A→B Δ | B→C Δ |
|---|---|---|---|---|---|
| iris | 0.967 (N=8) | 1.000 (N=4) | 0.967 (N=4) | +0.033 | -0.033 |
| wine | 0.833 (N=6) | 0.861 (N=8) | 0.833 (N=4) | +0.028 | -0.028 |
| seeds | 0.881 (N=8) | 0.976 (N=6) | 0.833 (N=4) | +0.095 | -0.143 |
| breast_cancer | 0.895 (N=6) | 0.939 (N=8) | 0.895 (N=5) | +0.044 | -0.044 |

## Stage E vs best of staged (A/B/C) per dataset

Does fully co-evolving the encoder structure (Stage E) beat the best staged formulation? Δ > 0 = E wins.

| dataset | best of A/B/C | Stage E | Δ (E − best) |
|---|---|---|---|
| iris | 1.000 | 0.700 | -0.300 |
| wine | 1.000 | 0.778 | -0.222 |
| seeds | 1.000 | 0.833 | -0.167 |
| breast_cancer | 0.974 | 0.851 | -0.123 |
