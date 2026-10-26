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

Gradient convergence is evaluated before extending the ansatz, over all eligible operators. A tiny new parameter alone is an advisory flag. After the first extension, the combination of objective stagnation (`--energy-improvement-tol`) and a tiny new parameter (`--new-theta-tol`) stops with `energy_stagnation`; it does not imply global convergence of the pool. Setting either threshold to zero disables this combined criterion.

## Revised specification: safe ADAPT iterations

The revised PDF is implemented excluding section 6: the existing `full_vqd`/`energy_only` selection modes, gradient formulas and defaults are retained. No new penalty-gradient measurement protocol is introduced.

The reference is the minimum diagonal-energy determinant in the physically allowed many-body basis. Ties use the canonical tuple of occupied orbital indices. Reference selection never probes the pool or substitutes another determinant because of inactivity. Determinants retain nuclear-matrix basis order for validation, while a shared physical orbital mapping keeps preparation and Jordan-Wigner operators consistent. All modes of each populated species are represented, including modes absent from sparse operators.

In the installed PyNucShell API, `total_J` fixes the projection `M = J_z`; the basis is an M-scheme determinant basis, not a projection onto exact total angular momentum J. The pool consists of the library's two-body transitions connecting determinants of the whole sector, restricted by default to transitions present in H. We validate conservation of proton number, neutron number and M term by term. No claim of exact J conservation by individual generators is made. Pool metadata records this construction and any structural exclusions; invalid mappings or symmetry-changing operators produce explicit errors.

The base pool is independent of the chosen reference and stays fixed. A zero initial action or gradient does not remove an operator. Only the operator selected in the immediately preceding iteration is excluded: `[7, 0, 2, 7]` is allowed and `[7, 0, 0, 2]` is forbidden. There is no layer limit tied to pool size. With one configured candidate, the next step stops with `no_eligible_operators`. `--max-candidates` remains an explicit diagnostic cap, reported separately from the full base-pool size; its default is no cap.

Each extension starts at `(old_theta, 0)` and optimizes every parameter. Exact mode checks equality of the state and full VQD cost with the old ansatz using `--identity-tol` (default `1e-10`). The optimizer's endpoint and the initial point are reevaluated through the same fixed objective. The lowest-cost evaluated point is retained, even if COBYLA reports success at a worse point. `--acceptance-tol` (default `1e-12`) controls numerical cost-consistency/progress checks. These tolerances are not optimizer convergence tolerances. Excited-state physical energy may increase while the penalized objective decreases. With shots, comparisons concern measured estimates and do not establish statistical monotonicity.

Every result records `stop_reason`: `gradient_converged`, `max_iterations`, `no_eligible_operators`, `energy_stagnation`, `optimizer_failure`, or `numerical_error`. A small new theta is recorded as `small_new_theta`; it is not a standalone stop condition. An unsuccessful optimizer stops only if it also fails to make numerical progress; useful budget-limited steps can continue. The final `stop_details` is available even with history disabled. `n_iterations` counts attempted optimizations; history also includes terminal gradient scans that add no layer.

CSV/TXT/JSON include reference identity/energy, initial and post-reference pool sizes, eligible IDs/count, selected gradient, maximum eligible gradient and norm, operator history, energies and VQD costs before/after, optimizer status/message, raw versus accepted points, new theta, and the termination condition. Exact eigenvectors are used only for final validation through `exact_state_fidelities` and sector probability; fidelities within degenerate eigenspaces depend on the chosen eigenbasis. Sampled paper/QWC and destructive-SWAP interfaces are preserved, with measurement plans built lazily.

The 6He regression from the revised PDF can be reproduced with:

```bash
python scripts/run_vqd.py --shell p --protons 0 --neutrons 2 --J 0 --n-states 3 --mode statevector_exact --max-adapt-iter 20 --optimizer-maxiter 1500 --grad-tol 1e-5 --energy-improvement-tol 1e-6 --new-theta-tol 1e-5 --n-jobs 1 --n-workers 1 --seed 47433 --results-dir validation_results --run-label he6_pdf_revised
```

`tests/test_pdf_compliance.py` verifies two-run reproducibility, pool invariance under changing the reference, operators that become active later, nonconsecutive reuse, zero-layer identity, VQE/VQD acceptance, numerical/optimizer failures, termination, matrix/basis consistency, sampled interfaces, and the existing gradients against actual-circuit finite differences.

Validation on 2026-10-26: **56 tests passed**, using PyNucShell 0.1.14, Qibo 0.3.3 and SciPy 1.15.3. The command above selects occupied qubits `[0, 3]` (nuclear-basis index 0, diagonal energy `1.56793`) and preserves all 8 pool IDs; later selections have 7 eligible candidates. With the original COBYLA radii (`rhobeg=0.03`, `rhoend=0.005`):

| State | Physical energy | Absolute energy error | Target fidelity | Layers | Stop reason |
|---|---:|---:|---:|---:|---|
| Ground | -3.048764531010 | 3.05144e-5 | 0.99999568 | 3 | energy_stagnation |
| First excited | 0.478382718950 | 7.49740e-6 | 0.99999123 | 7 | energy_stagnation |
| Second excited | 4.763392481748 | 4.42482e-4 | 0.99646164 | 20 | max_iterations |

The largest final squared overlap between distinct states is `8.38e-7`. All accepted histories satisfy the numerical monotonicity checks, and a repeated run reproduces operator sequences and energies. The remaining second-excited-state error is not reported as convergence: the run reaches its configured iteration limit. The two lower states terminate with the documented stagnation criterion while eligible gradients remain above `grad_tol`; this is distinguishable from gradient convergence. No optimizer tolerances were silently tightened to obtain these results. Local CSV/TXT/JSON evidence is in `validation_results/he6_pdf_revised/` (runtime outputs are ignored by Git).

## VQD beta rule

For excited-state calculations with `--n-states >= 2` and `--beta auto`, the code sets the VQD deflation penalty automatically as

```text
beta = 2 * max_i abs(E_i - E_0)
```
For a single-state run, beta is not used.
Explicit numeric beta values retain the existing behavior: they are respected and receive a warning if below the largest requested gap.

## Outputs

By default, each execution creates one self-contained run folder under `runs/`:

```text
runs/
  <run_label>/
    vqd_results_<run_label>.txt
    summary_<run_label>.csv
    history_<run_label>.csv
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
