# Architecture-aware circuit evolution (optimal control + EXAQC)

This directory and the files listed below add a hardware-aware path to EXAQC.
Circuits are no longer scored against one generic depolarising strength per gate
class. Each gate in the universal set {H, X, Z, S, T, CNOT} (EXAQC keys
`h x z s t cx`, plus `sdg tdg`) carries its own error channel, and that channel
comes from optimising the gate's control pulse on a modelled transmon device
under an open-system model. The per-gate results are collected in a JSON
*device card*. EXAQC then evolves circuits (Bell, GHZ_n, general state
preparation) against the card. Swapping the card changes the architecture or
the noise model without touching the genome.

The pulse optimisation lives outside this repository (different Python and
dependencies, no shared code):

| Stage | Where | What |
|---|---|---|
| 1. Gate-level optimal control | `RIT_research/Non-Markov/code/python` (pseudomode GRAPE, JAX) and `RIT_research/Spectral_Master_Equations/code/python` (spectral-in-time solver) | optimised pulses + error channels per gate and model |
| Card assembly | `make_device_card.py` (pseudomode), `spectral_device_card.py` (spectral) | device card JSON |
| 2. Circuit search | this repo | NSGA-II over circuits, scored with the card |

Project notes, plan and status: `RIT_research/Arch_Aware_Circuit_Evolution/`
(`PLAN.md`, `WHERE_WE_ARE.md`, `RESUME_SPECTRAL.md`, and an overview PDF in
`overview/`).

## Open-system models

Every card is built under one of four models of the same device:

- `schrod`: closed system (Schrödinger), the control-only floor.
- `lindblad_markov`: Markovian Lindblad dissipation (T1, T2).
- `lindblad_nm`: non-Markovian bath with a memory kernel (pseudomode embedding,
  or the memory integral directly in the spectral solver).
- `mixed`: both at once.

Two reference cards come from a practitioner who has no device simulation
(`scripts/make_naive_card.py`): `naive_closed` (perfect gates) and
`naive_depol` (uniform depolarising). `scripts/naive_vs_simulated.py` asks
whether the simulated cards change which circuits win.

## What a device card holds

For each gate: `T_ns` (pulse duration), `infidelity` (1 − average gate fidelity
on the computational subspace), `process_infidelity`, `leakage` (time-averaged
population outside the computational subspace), `trace_loss`, the
Pauli-twirled error probabilities `pauli_error` of E = V ∘ U†, and
`ptm_error_diag`. `idle` holds the zero-control channel at each gate duration,
which is applied to spectator wires. `conventions` documents each field, and
`sources` records the run id, seed and every solver parameter behind every
gate.

The twirl probabilities sum to 1 − `trace_loss`, not to 1, because whatever
leaks out of the computational subspace is missing from them. Small negative
entries can appear when the channel is not Pauli-dominated;
`DeviceCardNoiseModel` clips them and renormalises.

## Cards in `device_cards/`

Copied from `RIT_research/Arch_Aware_Circuit_Evolution/device_cards/` on
2026-10-20. That folder stays the canonical copy, and the scripts default to it
through the relative path `../../Arch_Aware_Circuit_Evolution/device_cards`.

| Card | Built from | Status |
|---|---|---|
| `transmon_{schrod,lindblad_markov,lindblad_nm,mixed}` | pseudomode GRAPE, fixed resonant coupling (runs `sweep-N100-cluster-2026-07-07`, `gateset-1q-2026-09-14`, `gateset-cnot-2026-09-14`) | The CNOT on this device is not a valid gate for the coupling used. Kept as the counterexample. Two-qubit numbers from these cards must be regenerated. |
| `transmon_detuned_*` | pseudomode GRAPE, 4 levels, detuned second transmon (`arch-detuned-2026-09-26`) | Current cards for Stage 2. Note `delta_2q = 2π·0.3 rad/ns` (300 MHz), which was found on 2026-10-20 to be a frequency collision. CNOT cards are being rebuilt at 150 MHz. |
| `naive_closed`, `naive_depol` | no simulation | Baselines. |

Spectral-solver cards (5 levels for the 1q gates; CNOT pending, see below) will
be added as `transmon_spectral_*` once the CNOT is resolved.

## Code added to EXAQC

Noise and objectives:

- `src/noise/DeviceCardNoiseModel.py`: per-gate Pauli channels from a card,
  plus idle noise on waiting wires from an ASAP schedule.
- `src/objectives/hardware_objectives.py`: `DeviceCard` loader and
  simulation-free objectives `p_fail` (independent-error estimate),
  `n_entanglers`, and `total_time_ns` (ASAP makespan, i.e. depth in
  nanoseconds).
- `src/objectives/{bell,ghz,state}_prep_objective.py`: task losses for state
  preparation.
- `src/evolution/pareto_population.py`: NSGA-II multi-objective population,
  with task loss as a feasibility constraint.

Changes to existing EXAQC files:

- `src/circuits/circuit.py`
  - `CircuitGenome.copy()` copied gates by reference, so in-place mutations
    reached back into parents that were still in the population. Archived
    Pareto genomes ended up with every gate disabled. It now copies each gate.
  - New noise hooks: `begin_circuit(genome)` before the first gate,
    `before_gate(wires, gate=)` for idle noise, and
    `after_gate(wires, gate=)` with the gate passed through.
  - `before_measurement` used to run only when `return_probs` was set. It now
    runs for every measurement type, so state and expval evaluations are no
    longer silently noiseless.
- `src/noise/BaseNoiseModel.py`: adds `device_card` to the noise types, and
  imports qiskit-aer lazily. Loading it at module scope breaks the pure
  PennyLane path on aarch64 nodes.
- `src/noise/PennyLaneNoiseModel.py`: no-op `begin_circuit` and `before_gate`,
  and an optional `gate=` argument on `after_gate`.

Scripts (`scripts/`):

| Script | Purpose |
|---|---|
| `device_card_smoke.py` | hand-built Bell genome through the card noise model |
| `evolve_bell_pareto.py`, `evolve_ghz_pareto.py`, `evolve_state_pareto.py` | NSGA-II searches under a card |
| `merge_pareto.py` | union of the fronts from several restarts |
| `reference_circuits.py`, `standard_vs_learned.py` | textbook chain/tree circuits vs learned ones under every card |
| `cross_evaluate.py` | learn under model k, run under model k′ |
| `naive_vs_simulated.py`, `make_naive_card.py` | does the device simulation change the answer? |
| `ablate_drives.py`, `prune_trace.py` | feasibility-preserving pruning of learned circuits |
| `ghzn_inflection.py` | at what n hardware-aware refinement starts to matter |
| `compare_distributions.py` | measurement distributions, on-paper vs learned |
| `trace_to_qasm.py`, `make_circuit_movie.py` | QASM and animation of a search trace |
| `make_search_cells.py`, `run_campaign.py` | SLURM-array cells and a bounded-worker driver for tigris |

Example:

    python scripts/evolve_ghz_pareto.py \
        --card arch_aware/device_cards/device_card_transmon_detuned_mixed.json \
        --n 3 --genomes 60000 --pop 80 --seed 1 --out results/ghz3/mixed_s1

Run several `--seed` values and merge them with `scripts/merge_pareto.py`.
The landscape has feasible local optima that no single gate deletion
escapes.

## Status (2026-10-20)

- The 1q spectral cards at 5 levels are complete: all 80 seeds converged, with
  infidelities from 4.6e-6 (closed system) to about 4e-4 (`mixed`).
- The spectral CNOT at 4 levels reaches 1 − F ≈ 2e-3 (`schrod`,
  `lindblad_nm`) to 6.5e-3 (`mixed`). However, mid-pulse population in the top
  level reaches 24–31%, so the result is truncation-limited and not yet used
  as a card. Next is a re-run at 150 MHz detuning with 4 and 5 levels.
- Until then, Stage-2 results use the `transmon_detuned` pseudomode cards. They
  carry the 300 MHz collision caveat above.
