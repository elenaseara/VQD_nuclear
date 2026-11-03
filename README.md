# Nuclear ADAPT-VQD with Qibo and PyNucShell

This repository computes low-lying shell-model nuclear states using an ADAPT-VQD workflow implemented with PyNucShell Hamiltonians and Qibo circuit simulation.

The code supports two objective-evaluation routes:

1. `statevector_exact`: the ansatz circuit is simulated as a dense statevector, and energies and overlaps are evaluated exactly from the simulator statevector.
2. `full_quantum_circuit`: energies are estimated from shot-based shell-model measurement circuits and VQD deflation overlaps are estimated with destructive-SWAP circuits. 

The second route is a sampled, ideal circuit-simulation workflow. It is not a hardware execution workflow: no device noise model, transpilation to hardware-native gates, readout mitigation, or backend calibration data are included.

## Repository structure

```text
vqd_nuclear/
  src/vqd_nuclear/
    config.py          # dataclass-based configuration
    hamiltonian.py     # qUBshell Hamiltonian loading and exact diagonalization
    ansatz.py          # reference determinant selection and ADAPT ansatz construction
    measurements.py    # statevector, paper-style shot measurements, QWC fallback, and destructive-SWAP estimators
    parallel.py        # safe worker helpers
    validation.py      # beta rule and lightweight validation helpers
    vqd.py             # ADAPT-VQD driver
    io.py              # TXT, CSV, and JSON outputs
  scripts/
    run_vqd.py         # command-line entry point
  tests/
    test_measurements.py
    test_beta.py
    test_smoke.py
```


### Install PyNucShell

PyNucShell is not installed automatically from the main PyPI index. Install it explicitly from TestPyPI before installing this repository:

```bash
python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple pynucshell
```

Equivalent command using the provided requirements file:

```bash
python -m pip install -r requirements-pynucshell.txt
```

Check that qUBshell is available in the active environment:

```bash
python -c "import pynucshell; print(pynucshell.__file__)"
```

If this command fails, the nuclear Hamiltonian examples cannot run.

## Example run

A small smoke-style run can be launched with the short aliases:

```bash
python scripts/run_vqd.py \
  --shell p \
  --protons 2 \
  --neutrons 0 \
  --J 0 \
  --n-states 2 \
  --mode statevector_exact \
  --max-adapt-iter 2 \
  --max-candidates 8 \
  --optimizer-maxiter 50
```

The equivalent long-form command is:

```bash
python scripts/run_vqd.py \
  --shell p \
  --n-valence-protons 2 \
  --n-valence-neutrons 0 \
  --total-j 0 \
  --n-states 2 \
  --mode statevector_exact \
  --max-adapt-iter 2 \
  --max-candidates 8 \
  --optimizer-maxiter 50
```

The `--mode` option accepts:

- `statevector_exact`: exact simulator energies and exact statevector overlaps;
- `full_quantum_circuit`: shot-based shell-model energy estimates and destructive-SWAP overlap estimates;
- `both`: run both routes.

The sampled energy route uses the measurement strategy by default:

```bash
python scripts/run_vqd.py --mode full_quantum_circuit --measurement-strategy paper ...
```

This route measures diagonal terms in the computational basis, groups single-hopping terms by their hopping indices and applies the M_jk basis change, and groups double-hopping terms by their four active indices and applies the M_ijkl basis change. If a transformed term is not diagonal under these basis changes, it is handled by a residual qubit-wise-commuting fallback to preserve numerical correctness. For comparison with the previous generic implementation, use:

```bash
python scripts/run_vqd.py --mode full_quantum_circuit --measurement-strategy qwc ...
```

The older flags `--statevector true/false` and `--full-quantum-circuit true/false` are still supported for backward compatibility, but `--mode` is recommended.

### Convergence and gradient selection

The default ADAPT operator selection now uses the analytic gradient of the full VQD objective instead of shot-based finite differences. For the ground state this reduces to the usual ADAPT-VQE commutator gradient:

```text
dE/dtheta_mu = -i <psi|[H, G_candidate]|psi>
```

For excited states, the same analytic selector also includes the derivative of the overlap-penalty terms in the deflated VQD objective:

```text
dF/dtheta_mu = dE/dtheta_mu + sum_i beta_i d|<psi_i|psi>|^2/dtheta_mu
```

This is the recommended setting because finite-difference gradients are very sensitive to sampling noise in the full-circuit route. The legacy finite-difference selector is still available for comparisons:

```bash
python scripts/run_vqd.py --gradient-method finite_difference ...
```

Gradient convergence is evaluated before extending the ansatz, over all eligible operators. Small objective improvements (`--energy-improvement-tol`) and tiny new parameters (`--new-theta-tol`) are diagnostic flags; neither ends a run with significant eligible gradients. A failed provisional optimization leaves the accepted ansatz unchanged and triggers the next ranked candidate.

## Revised specification: safe ADAPT iterations

The existing `full_vqd`/`energy_only` selection modes, gradient formulas and COBYLA radii are retained. For excited states, both selectors always optimize the full VQD objective; `energy_only` changes only the candidate ranking. No new penalty-gradient measurement protocol is introduced.

The reference is the minimum diagonal-energy determinant in the physically allowed many-body basis. Ties use the canonical tuple of occupied orbital indices. Reference selection never probes the pool or substitutes another determinant because of inactivity. Determinants retain nuclear-matrix basis order for validation, while a shared physical orbital mapping keeps preparation and Jordan-Wigner operators consistent. All modes of each populated species are represented, including modes absent from sparse operators.

In the installed PyNucShell API, `total_J` fixes the projection `M = J_z`; the basis is an M-scheme determinant basis, not a projection onto exact total angular momentum J. The pool consists of the library's two-body transitions connecting determinants of the whole sector, restricted by default to transitions present in H. We validate conservation of proton number, neutron number and M term by term. No claim of exact J conservation by individual generators is made. Pool metadata records this construction and any structural exclusions; invalid mappings or symmetry-changing operators produce explicit errors.

The base pool is independent of the chosen reference and stays fixed. A zero initial action or gradient does not remove an operator. Only the last accepted operator is excluded: `[7, 0, 2, 7]` is allowed and `[7, 0, 0, 2]` is forbidden. There is no layer limit tied to pool size. With one configured candidate, the next step stops with `no_eligible_operators`. `--max-candidates` remains an explicit diagnostic cap, reported separately from the full base-pool size; its default is no cap.

Each provisional extension starts at `(old_theta, 0)`. Exact mode checks equality of the state and full VQD cost with the old ansatz using `--identity-tol` (default `1e-10`). Before COBYLA, the new parameter is tested at `-sign(g_total) * eta` for `eta = 0.1, 0.01, 0.001, 0.0001`, with all old parameters fixed. The best of these points and zero seeds the optimizer, which then reoptimizes every parameter. The seed gradient is always the derivative of the full optimized objective, including when the ranking uses `energy_only`. The existing `--cobyla-rhobeg`, `--cobyla-rhoend` and `--optimizer-maxiter` remain configurable.

The lowest-cost evaluated point is retained even if COBYLA ends at a worse point. A new layer is committed only if its independently reevaluated exact VQD objective improves the last accepted objective by **more than** `--acceptance-tol` (default `1e-12`). Rejection restores all previous operators, parameters and cost; `layer_added` and `accepted_candidate` always agree. Candidate diagnostics are computed before committing. Excited-state physical energy may increase while the penalized objective decreases.

One gradient ranking is computed per accepted ansatz, with canonical IDs breaking relative `1e-9` ties. Rejected extensions trigger the next candidate from that ranking without another gradient scan or a permanent blacklist. `--max-candidate-attempts` defaults to 10. Accepted operators remain reusable in nonconsecutive layers. The attempt limit is separate from the diagnostic `--max-candidates` pool cap.

In `full_quantum_circuit`, acceptance uses fresh independent estimates of the old and candidate full costs (`--sampled-acceptance-repeats`, default 3, minimum 2). The mean improvement must exceed `acceptance_tol + z * SE`, where `SE = sqrt((var(old) + var(candidate)) / repeats)` and `--sampled-acceptance-z` defaults to 2. This empirical uncertainty check is not an exact monotonicity guarantee or a calibrated confidence interval for a small number of repeats. Previously stored measured costs can increase when independently resampled estimates justify acceptance.

Every result records `stop_reason`: `gradient_converged`, `max_iterations`, `no_eligible_operators`, `candidate_attempt_limit`, `no_improving_candidate`, `optimizer_failure`, or `numerical_error`. `candidate_attempt_limit` means only part of the eligible ranking was tried; `no_improving_candidate` means all eligible candidates were tried without sufficient improvement. Neither means gradient convergence. `optimizer_failure` reports an optimizer exception; a budget-limited result can still be accepted if it improves the objective, or rejected in favor of another candidate. `energy_stagnation` is no longer a termination reason. The final `stop_details` remains available with history disabled. `n_iterations` counts provisional optimizer calls, including rejections, while `--max-adapt-iter` limits outer ADAPT rounds. Each round accepts at most one layer.

Each history record contains `candidate_attempts`, including candidate gradients, provisional parameters and costs, descent trials, optimizer status, timings and acceptance. Machine outputs additionally flatten these into `candidate_attempts_<run_label>.csv`; rejected proposals never appear in final parameters.

Validation of the transactional update on 2026-11-03: all 370 F18 candidate
gradients were compared with central differences at `h = 1e-3, 1e-4, 1e-5`,
for ground/excited states and both selection criteria, including the final
states of a 20-round, 300-evaluation-per-optimizer run. The largest final
gradient discrepancy at `h = 1e-5` was below `4e-10`. Full-state callbacks were
forbidden throughout the run and export; the backend stayed in dimension 28.
The workflow, including both beta families, took about 10 s locally.

With the locally installed interaction, the exact ground energy is
`-11.6014757411 MeV`, different from `-13.4131718520 MeV` in the supplied report.
Under identical local settings, the previous implementation's ground error was
`1.2249e-3 MeV`; the corrected implementation reached `1.9546e-4 MeV`, with
every committed layer improving the objective. These runs reached
`max_iterations` with significant gradients and do not establish convergence.
The calculated excited state matched exact eigenvector index 2, not index 1;
its energy label alone must not be interpreted as finding the first excited
state. The tests also exercise rejected candidates, retries and rollback using
controlled objectives, since this particular corrected F18 run accepted its
first ranked candidate in every round.

CSV/TXT/JSON include reference identity/energy, initial and post-reference pool sizes, eligible IDs/count, selected gradient, maximum eligible gradient and norm, operator history, energies and VQD costs before/after, optimizer status/message, raw versus accepted points, new theta, and the termination condition. Exact eigenvectors are used only for final validation through `exact_state_fidelities` and sector probability; fidelities within degenerate eigenspaces depend on the chosen eigenbasis. Sampled paper/QWC and destructive-SWAP interfaces are preserved, with measurement plans built lazily.

The 6He regression from the revised PDF can be reproduced with:

```bash
python scripts/run_vqd.py --shell p --protons 0 --neutrons 2 --J 0 --n-states 3 --mode statevector_exact --max-adapt-iter 20 --optimizer-maxiter 1500 --grad-tol 1e-5 --energy-improvement-tol 1e-6 --new-theta-tol 1e-5 --n-jobs 1 --n-workers 1 --seed 47433 --results-dir validation_results --run-label he6_pdf_revised
```

`tests/test_pdf_compliance.py` verifies two-run reproducibility, pool invariance under changing the reference, operators that become active later, nonconsecutive reuse, zero-layer identity, VQE/VQD acceptance, numerical/optimizer failures, termination, matrix/basis consistency, sampled interfaces, and the existing gradients against actual-circuit finite differences.

Historical validation on 2026-10-26, before transactional acceptance: **56 tests passed**, using PyNucShell 0.1.14, Qibo 0.3.3 and SciPy 1.15.3. The command above selects occupied qubits `[0, 3]` (nuclear-basis index 0, diagonal energy `1.56793`) and preserves all 8 pool IDs; later selections have 7 eligible candidates. With the original COBYLA radii (`rhobeg=0.03`, `rhoend=0.005`):

| State | Physical energy | Absolute energy error | Target fidelity | Layers | Stop reason |
|---|---:|---:|---:|---:|---|
| Ground | -3.048764531010 | 3.05144e-5 | 0.99999568 | 3 | energy_stagnation |
| First excited | 0.478382718950 | 7.49740e-6 | 0.99999123 | 7 | energy_stagnation |
| Second excited | 4.763392481748 | 4.42482e-4 | 0.99646164 | 20 | max_iterations |

The largest final squared overlap between distinct states is `8.38e-7`. All accepted histories satisfy the numerical monotonicity checks, and a repeated run reproduces operator sequences and energies. The remaining second-excited-state error is not reported as convergence: the run reaches its configured iteration limit. The two lower states terminate with the documented stagnation criterion while eligible gradients remain above `grad_tol`; this is distinguishable from gradient convergence. No optimizer tolerances were silently tightened to obtain these results. Local CSV/TXT/JSON evidence is in `validation_results/he6_pdf_revised/` (runtime outputs are ignored by Git).

## Direct model-space projection

Model-space initialization projects Pauli strings directly onto the ordered
physical determinants, without allocating statevectors of length `2**n_qubits`.
Contributions to each external destination are summed before measuring leakage;
the existing Hermiticity checks and full-state fallback remain in place. The
same projection is used for the Hamiltonian and lazy generator construction.
`tests/test_model_space_projection.py` compares it against the original dense
implementation on small systems and checks ADAPT-VQD trajectories with both beta
methods. A separate profiling command measures the F18 context and all generators:

```bash
python scripts/profile_model_space.py --shell sd --protons 1 --neutrons 1 --J 0 --all-generators --output validation_local/projection_f18.json
```

The JSON records context/projection timings, peak process memory (including
imports), traced allocations, generator preservation flags and environment
versions. Timings include memory tracing overhead. Full-state operator calls
are forbidden during this diagnostic.

Analytic ADAPT scans also use the reduced basis when the Hamiltonian, current
ansatz, previous ansatze and **all candidates** preserve it. A failing check sends
the entire scan through the full-state route; no candidate is removed. Individual
exact gradients, zero-parameter extension checks and final fidelities use the
same reduced representation where applicable. Fidelity diagnostics explicitly
reorder determinants to the exact eigenvectors' basis. The derivative convention
is `-1j * A_S @ psi_S`, with the phase applied once. The shared constant
`ANSATZ_DERIVATIVE_PHASE` lives in `conventions.py`; `GRADIENT_PHASE` remains a
compatibility alias.

Reduced propagation also requires pairwise commutation of each generator's
full Pauli strings. A generator can preserve the physical sector while its sum
exponential differs from the product of Pauli exponentials in the circuit.
Such a generator uses the full-state route and remains in the candidate pool.
The norm is checked on newly propagated and cached reduced states. A nonfinite
norm or an error above `1e-10` disables the reduced engine, clears its state
cache and makes callers reevaluate the entire operation in the full basis.
This norm check detects numerical failure; operator leakage is checked separately
during projection. Reduced states are never renormalized to hide a failure.

These checks implement the equivalence conditions in the projection report of
2026-10-28. Its statement that gradients remain unprojected describes the earlier
version; this branch extends the same exact representation to analytic gradients.
The canonical pool and deterministic tie rule are already in place, and gradient
convergence is checked before adding a layer or running the optimizer. Sampled
`full_quantum_circuit` selection, propagation and measurement use the full route.

Each analytic scan records `gradient_profile` in the ADAPT history and logs it:
backend, state dimension, candidate count, state preparation time, Hamiltonian
action time, penalty time and total scan time (including lazy projections).
History also records extension checks, optimization and acceptance times;
results record context construction and final diagnostics times, in seconds.
These fields are retained in raw JSON outputs.

`tests/test_model_space_gradients.py` checks energy, penalty and total gradients
against dense evolution and finite differences, significant penalties with zero
energy gradient, nonconsecutive reuse, fallback, basis permutation and complete
small-system ADAPT trajectories for both selection criteria. Full-state calls
are forbidden during reduced trajectory tests, including optimizer evaluations,
identity checks and diagnostics.
`tests/test_model_space_projection.py` additionally compares actual circuit
states (after global phase alignment), energies, spectra and overlaps for Be6,
He6 and Li6, and checks noncommuting-generator and invalid-norm fallback.

## VQD beta formulas

Each selected execution route runs two independent VQD families, with a penalty calculated separately for each target state `i` (ground state is `i=0`):

```text
beta_exact_i = 2 * abs(E_i - E_0)
beta_approximate_i = abs(e_i - e_0 + 2 * B)
```
`E_i` are exact eigenenergies. `e_i` are the diagonal entries of the many-body Hamiltonian sorted in ascending order, retaining repeated values. `B = max_r sum_(c != r) abs(H_rc)` is the largest absolute off-diagonal row sum. The first excited state uses `e_1`; the second uses `e_2`. Sparse matrices are processed without densifying them.

Only these two automatic rules are supported. Ground-state runs use no penalty. Each family starts from its own ground-state calculation, deflates against its own previous states, and starts with the same seed and numerical settings.

For `N` states, each route returns `N` exact-beta results followed by `N` approximate-beta results. TXT, CSV and JSON outputs record `beta_method` and the beta actually used for each state. Energy matching, sorting and gradient diagnostics distinguish both variants. The local PowerShell launcher compares model-space and full-circuit results separately for each beta variant and execution mode.

## Outputs

By default, each execution creates one self-contained run folder under `runs/`:

```text
runs/
  <run_label>/
    vqd_results_<run_label>.txt
    summary_<run_label>.csv
    history_<run_label>.csv
    candidate_attempts_<run_label>.csv
    parameters_<run_label>.csv
    overlap_matrix_<run_label>.csv
    raw_results_<run_label>.json
    energy_sorted_summary_<run_label>.csv
    matched_validation_summary_<run_label>.csv
```


## Reproducibility

The command-line interface exposes `--seed`. The seed is applied to NumPy and to Qibo when the installed Qibo version exposes a seeding function. Shot-based runs are therefore reproducible to the extent supported by the active Qibo backend.

## Tests

```bash
python -m pytest -v
```

The smoke test is intentionally small. If qUBshell is not installed, integration tests are skipped. The repository also includes a GitHub Actions workflow under `.github/workflows/tests.yml` for automatic test execution on push and pull request events.

## Validation summaries

Sequential VQD stores states in the order in which they are obtained. For strongly correlated systems, the first ADAPT run may not converge to the lowest exact eigenstate, and truncated runs can return a correct set of energies in a permuted order. A raw comparison by state index can therefore produce misleadingly large validation errors. Each run writes two additional diagnostics:

```text
energy_sorted_summary_<run-label>.csv
matched_validation_summary_<run-label>.csv
```

`energy_sorted_summary_<run-label>.csv` sorts computed energies and exact energies by energy rank. `matched_validation_summary_<run-label>.csv` is the stricter validation table: it solves the one-to-one assignment between computed states and exact states that minimizes the total absolute energy error. Inspect `matched_abs_error`, `raw_index_abs_error`, and `state_label_changed` to distinguish a real non-converged state from a simple state-label permutation. The same information is included in the text report under `MATCHED VALIDATION SUMMARY` and `ENERGY-SORTED SUMMARY`.
