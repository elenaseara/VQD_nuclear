# Nuclear ADAPT-VQD with Qibo and PyNucShell

This repository computes low-lying shell-model nuclear states using an ADAPT-VQD workflow implemented with PyNucShell Hamiltonians and Qibo circuit simulation.

The many-body determinants, Hamiltonian and excitation pool share an explicit orbital mapping derived from the PyNucShell single-particle space. All orbitals of each populated species are retained. Neutron-only calculations use a compact register (6 qubits in the p shell, 12 in sd); mixed systems retain both species. Fermionic operators are relabeled before Jordan-Wigner encoding so parity strings follow the same convention as reference circuits. Reference selection uses the existing shell-model matrix diagonal. The mapping tests compare determinant energies with that diagonal. TXT and JSON reports record the orbital mapping.

The reference is the determinant with the smallest Hamiltonian diagonal entry within the supplied many-body basis and symmetry sector. Equal minima select the first determinant in basis order. Selection does not probe the operator pool or discard disconnected determinants, and it never switches to another reference automatically. Reports record the selected basis index and diagonal energy. ADAPT can reuse operators across iterations, excluding only the immediately preceding operator. If no eligible candidates remain or their gradients are below tolerance, ADAPT reports its usual stopping reason.

PyNucShell 0.1.14 uses global fermionic labels, while 0.1.16 uses local labels for neutron-only Hamiltonians and excitation pools. Determinant labels remain global. The Hamiltonian and the complete pool are checked independently against the full global and local mode spaces before encoding. Ambiguous mappings are rejected; the convention is never inferred solely from the maximum index. Reports record separate maps for determinants, Hamiltonian terms and pool terms.

The code supports two objective-evaluation routes:

1. `statevector_exact`: the ansatz circuit is simulated as a dense statevector, and energies and overlaps are evaluated exactly from the simulator statevector.
2. `full_quantum_circuit`: energies are estimated from shot-based shell-model measurement circuits and VQD deflation overlaps are estimated with destructive-SWAP circuits. 

The second route is a sampled, ideal circuit-simulation workflow. It is not a hardware execution workflow: no device noise model, transpilation to hardware-native gates, readout mitigation, or backend calibration data are included.

## Repository structure

```text
vqd_nuclear/
  src/vqd_nuclear/
    config.py          # dataclass-based configuration
    hamiltonian.py     # PyNucShell Hamiltonian loading and exact diagonalization
    orbitals.py        # shared physical-orbital to JW qubit mapping
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

The gradient is checked before adding a layer. A tiny new parameter stops ADAPT only when the VQD objective also stagnates (`energy_stagnation`). Set either `--new-theta-tol 0` or `--energy-improvement-tol 0` to disable this combined criterion. An unsuccessful optimizer stops ADAPT only if it also fails to improve the objective (`optimization_failure`); useful steps remain accepted even if the optimization budget is exhausted.

Each extension starts with the previous optimized parameters followed by zero. In `statevector_exact`, ADAPT checks that this preserves both the state and objective to an absolute tolerance of `1e-10`. COBYLA optimizes all parameters, and acceptance retains the best evaluated full VQD objective, including the initial point. Physical excited-state energy need not decrease. In sampled mode, this comparison concerns measured values; fresh measurements can fluctuate.

Results record `stop_reason` (`max_iterations`, `gradient_converged`, `pool_exhausted`, `energy_stagnation`, or `optimization_failure`) and `stop_details`, including pool counts and the final selection gradient. The history records initial/accepted parameters, energies and objectives, the new theta, optimizer status/message, raw optimizer objective, and tolerances. `n_iterations` counts optimization steps even with history disabled. Exact simulation also records eigenvector fidelities, sector probability and energy variance; fidelities within degenerate eigenspaces depend on the eigenbasis.

An operator may be selected more than once, but never in consecutive iterations: `[0, 2, 0]` is allowed and `[0, 0]` is forbidden. At each step, only the immediately preceding operator is excluded from the candidate list. The default pool policy admits only operators active on the reference. Four candidates can therefore generate more than four layers, up to `--max-adapt-iter`, subject to the stopping criteria. A single candidate becomes exhausted after its first selection. Operators inactive on the reference can become relevant later; compare `--pool-policy full` to evaluate the complete pool with the same restriction on consecutive selections. `--max-candidates` truncates the configured candidate list before excluding the last operator. `--optimizer-tol` exposes COBYLA's existing default `5e-3` so optimizer accuracy can be studied independently of ADAPT's `--grad-tol`. The PowerShell launcher exposes the same controls.

See the [6He diagnosis and reproduction commands](docs/adapt_vqd_diagnostics.md) for the measured effects of optimizer precision and pool policy.

## VQD beta rule

Each selected execution route runs two independent VQD families, with a penalty calculated separately for each target state `i` (ground state is `i = 0`):

```text
beta_exact_i = 2 * abs(E_i - E_0)
beta_approximate_i = abs(e_i - e_0 + 2 * B)
```
`E_i` are exact eigenenergies. `e_i` are the diagonal entries of the many-body Hamiltonian sorted in ascending order, retaining repeated values. `B = max_r sum_(c != r) abs(H_rc)` is the largest absolute off-diagonal row sum. Thus the first excited state uses `e_1`, and the second uses `e_2`. The approximate penalty is computed directly from the matrix without using eigenenergies.

Beta is calculated automatically using these two methods. Ground-state calculations use no penalty. Each family starts with its own ground state and deflates only against its own previous states. Both families use the same seed and numerical settings.

TXT, CSV, and JSON results identify the variant as `beta_method` (`exact` or `approximate`) and record the beta used for each state. Energy sorting and validation are performed separately for each variant. With both execution routes enabled, four families are computed. For `n_states = N`, each route's result list contains `N` exact-beta results followed by `N` approximate-beta results.

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
