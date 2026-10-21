"""Regression coverage for acceptance, stopping and the actual He6 circuits."""
import numpy as np
import pytest
from scipy.optimize import OptimizeResult

pytest.importorskip("qibo")
pytest.importorskip("pynucshell")

from vqd_nuclear import vqd
from vqd_nuclear.config import NumericalConfig, ProblemConfig, RunConfig


@pytest.mark.parametrize("best_trial,expected", [(None, [0.2, 0.0]), ([0.1, 0.0], [0.1, 0.0])])
def test_optimizer_keeps_initial_or_best_evaluation(monkeypatch, best_trial, expected):
    calls = []

    def cost(context, thetas, ops, previous, betas, **kwargs):
        # The physical energy alone prefers the *worse* endpoint. The full
        # deflated functional prefers the initial point or the best trial.
        calls.append(list(thetas))
        physical = -float(thetas[0])
        return physical + betas[0] * float(thetas[0]) ** 2

    def minimize(objective, x0, **kwargs):
        np.testing.assert_array_equal(x0, [0.2, 0.0])
        if best_trial is not None:
            objective(np.array(best_trial))
        endpoint = np.array([0.8, 0.5])
        return OptimizeResult(x=endpoint, fun=objective(endpoint), success=True, message="done", nfev=2)

    monkeypatch.setattr(vqd, "vqd_cost", cost)
    monkeypatch.setattr(vqd, "minimize", minimize)
    thetas, value, result = vqd.optimize_ansatz(
        None, [0.2, 0.0], [7, 2], [{"ops": [], "thetas": []}], [10.0],
        overlap_shots=1, energy_shots=1, optimizer_maxiter=10, n_jobs=1, cost_mode="statevector_exact",
    )
    assert thetas == expected
    assert value == pytest.approx(-expected[0] + 10 * expected[0] ** 2)
    assert result.used_best_evaluated
    assert result.raw_fun > result.fun
    assert calls[0] == [0.2, 0.0]


@pytest.fixture(scope="module")
def he6_context():
    return vqd.build_context(
        ProblemConfig(shell="p", n_valence_protons=0, n_valence_neutrons=2, total_j=0),
        RunConfig(n_states=3, run_full_quantum_circuit=False, n_workers=1),
        NumericalConfig(n_jobs=1),
    )


def test_he6_embedding_and_operator_parameter_association(he6_context):
    builder = he6_context.ansatz_builder
    thetas, ops = [0.31, -0.47], [7, 0]
    state = builder.state(thetas, ops)
    for candidate in range(len(he6_context.hamiltonian.qubit_generator_pool)):
        np.testing.assert_allclose(builder.state(thetas + [0.0], ops + [candidate]), state, atol=1e-12)
    assert not np.allclose(builder.state(thetas, ops[::-1]), state)
    with pytest.raises(ValueError, match="one parameter"):
        builder.build([0.1], [7, 0])
    with pytest.raises(ValueError, match="outside"):
        builder.build([0.1], [-1])


def test_he6_full_vqd_gradients_match_circuit_finite_differences(he6_context):
    context = he6_context
    thetas, ops = [0.31, -0.47], [7, 0]
    previous = [{"thetas": [0.23], "ops": [2]}, {"thetas": [-0.36, 0.18], "ops": [4, 7]}]
    betas = [9.0, 13.0]
    for candidate in range(len(context.hamiltonian.qubit_generator_pool)):
        _, analytic = vqd.analytic_gradient((context, candidate, thetas, ops, previous, betas))
        _, numeric = vqd.finite_difference_gradient(
            (context, candidate, thetas, ops, previous, betas, 1e-6, 1, 1, "statevector_exact"))
        assert analytic == pytest.approx(numeric, abs=2e-7, rel=2e-6)


def test_fidelity_uses_shell_basis_order_and_qibo_bit_order(he6_context, monkeypatch):
    from vqd_nuclear.ansatz import get_hamiltonian_basis_references

    data = he6_context.hamiltonian
    references = get_hamiltonian_basis_references(
        data.nucleus, n_qubits=data.n_qubits, n_valence_particles=2, orbital_mapping=data.orbital_mapping)
    state = np.zeros(2 ** data.n_qubits, dtype=complex)
    for occupied, amplitude in zip(references, data.exact_vectors[:, 2]):
        state[sum(1 << (data.n_qubits - 1 - q) for q in occupied)] = amplitude
    monkeypatch.setattr(he6_context.ansatz_builder, "state", lambda *args: state)
    result = vqd.exact_state_diagnostics(he6_context, [], [])
    assert result["exact_state_fidelities"][2] == pytest.approx(1, abs=1e-12)
    assert result["best_matching_exact_state"] == 2
    assert result["energy_variance"] < 1e-12


@pytest.mark.slow
def test_he6_three_states_monotone_without_consecutive_repetition(he6_context):
    context = he6_context
    betas = vqd.resolve_beta_comparison(context.hamiltonian.exact_energies, context.hamiltonian.matrix, 3)
    for method in ("exact", "approximate"):
        results = vqd.run_vqd_family(context, cost_mode="statevector_exact", beta_method=method,
                                    state_betas=betas[method]["betas"], options=dict(
                                        max_adapt_iter=6, optimizer_maxiter=200, n_jobs=1,
                                        grad_tol=1e-5, new_theta_tol=0, energy_improvement_tol=0))
        for result in results:
            assert all(a != b for a, b in zip(result["ops"], result["ops"][1:]))
            assert result["stop_reason"] in {"max_iterations", "gradient_converged", "optimization_failure"}
            assert result["n_iterations"] <= 6
            for i, record in enumerate(result["history"]):
                assert record["vqd_objective"] <= record["vqd_objective_before"] + result["monotonicity_tol"]
                assert abs(record["zero_parameter_objective_error"]) < 1e-10
                assert record["initial_thetas"][-1] == 0
                if i:
                    assert record["initial_thetas"][:-1] == result["history"][i - 1]["thetas"]
            assert sum(result["exact_state_fidelities"]) == pytest.approx(1, abs=1e-9)
            assert result["shell_sector_probability"] == pytest.approx(1, abs=1e-9)


@pytest.mark.parametrize("reason,options", [
    ("gradient_converged", {"grad_tol": 1e10}),
    ("max_iterations", {"max_adapt_iter": 1}),
    ("pool_exhausted", {"max_candidates": 0}),
])
def test_stop_reason_and_iteration_count_without_history(he6_context, reason, options):
    result = vqd.run_adapt_vqd(he6_context, store_history=False, n_jobs=1, **options)
    assert result["stop_reason"] == reason
    assert result["history"] == []
    assert result["n_iterations"] == (1 if reason == "max_iterations" else 0)


@pytest.mark.parametrize("pool_policy", ["reference_active", "full"])
@pytest.mark.parametrize("gradient_method", ["analytic", "finite_difference"])
def test_only_last_operator_is_excluded(he6_context, monkeypatch, pool_policy, gradient_method):
    context = he6_context
    options = dict(eps=1e-6, max_candidates=None, n_jobs=1, overlap_shots=1, energy_shots=1,
                   cost_mode="statevector_exact")
    candidates = vqd.get_candidate_operators(context, pool_policy=pool_policy)
    evaluated = []

    def gradient(task):
        candidate = task[1]
        evaluated.append(candidate)
        return candidate, float(100 - candidate)

    monkeypatch.setattr(vqd, "analytic_gradient", gradient)
    monkeypatch.setattr(vqd, "finite_difference_gradient", gradient)
    used = list(candidates)
    assert vqd.select_next_operator(context, [0.0] * len(used), used, [], [],
                                    pool_policy=pool_policy, gradient_method=gradient_method, **options)[0] == candidates[0]
    assert set(evaluated) == set(candidates[:-1])
    evaluated.clear()
    assert vqd.select_next_operator(context, [0.0] * len(used), used, [], [],
                                    pool_policy=pool_policy, gradient_method=gradient_method,
                                    **(options | {"max_candidates": 1}))[0] == candidates[0]
    assert evaluated == [candidates[0]]
    assert vqd.select_next_operator(context, [0.0], [candidates[0]], [], [],
                                    pool_policy=pool_policy, gradient_method=gradient_method,
                                    **(options | {"max_candidates": 1}))[0] is None


def test_reuse_can_extend_beyond_pool_size_and_counts_remain_correct(he6_context, monkeypatch):
    def stuck_optimizer(objective, x0, **kwargs):
        return OptimizeResult(x=x0.copy(), fun=objective(x0), success=True, message="test", nfev=1)

    monkeypatch.setattr(vqd, "minimize", stuck_optimizer)
    monkeypatch.setattr(vqd, "analytic_gradient", lambda task: (task[1], float(100 - task[1])))
    result = vqd.run_adapt_vqd(he6_context, max_adapt_iter=6, n_jobs=1,
                               energy_improvement_tol=0, new_theta_tol=0)
    candidates = sorted(he6_context.active_operator_indices)
    assert result["ops"] == candidates[:2] * 3
    assert result["n_iterations"] == result["n_layers"] == 6
    assert result["stop_reason"] == "max_iterations"
    assert [r["eligible_count"] for r in result["history"]] == [len(candidates)] + [len(candidates) - 1] * 5


@pytest.mark.parametrize("success,reason,count", [(False, "optimization_failure", 1), (True, "energy_stagnation", 2)])
def test_unsuccessful_optimization_is_distinct_from_stagnation(he6_context, monkeypatch, success, reason, count):
    def stuck_optimizer(objective, x0, **kwargs):
        return OptimizeResult(x=x0.copy(), fun=objective(x0), success=success, message="stuck", nfev=1)

    monkeypatch.setattr(vqd, "minimize", stuck_optimizer)
    result = vqd.run_adapt_vqd(he6_context, n_jobs=1)
    assert result["stop_reason"] == reason
    assert result["n_iterations"] == count
    assert result["history"][-1]["optimizer_message"] == "stuck"
