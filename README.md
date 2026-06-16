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

Convergence no longer stops merely because the most recently added theta is below `--new-theta-tol`. A tiny new parameter is treated as convergence only when the VQD objective has also stagnated and the selected gradient is small. This avoids premature stopping when the optimizer temporarily returns a near-zero parameter.

## VQD beta rule

For excited-state calculations with `--n-states >= 2`, the code sets the VQD deflation penalty automatically as

```text
beta = 2 * max_i abs(E_i - E_0)
```
For a single-state run, beta is not used.

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
