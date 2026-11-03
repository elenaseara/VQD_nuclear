"""Transactional ADAPT acceptance, descent initialization and candidate retries."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from vqd_nuclear import NumericalConfig, ProblemConfig, RunConfig, vqd


@pytest.fixture(scope="module")
def f18_context():
    return vqd.build_context(ProblemConfig(shell="sd", n_valence_protons=1, n_valence_neutrons=1),
                             RunConfig(n_states=2, n_workers=1), NumericalConfig(n_jobs=1))


@pytest.mark.parametrize("excited", [False, True])
@pytest.mark.parametrize("criterion", ["energy_only", "full_vqd"])
def test_f18_every_candidate_gradient_against_three_fd_steps(f18_context, excited, criterion):
    context = f18_context
    theta, ops = [0.17, -0.3, 0.2], [140, 96, 98]
    previous = [{"thetas": [0.23], "ops": [140]}] if excited else []
    betas = [2 * abs(np.diff(context.hamiltonian.exact_energies[:2])[0])] if excited else []
    candidates = list(context.base_operator_indices)
    scan = context.measurements.scan_candidate_gradients(theta, ops, candidates, previous, betas)
    scores = vqd.selection_scores(scan, criterion)
    ms = context.measurements.model_space
    assert len(ms.indices) == 28 and len(candidates) == 370
    assert context.measurements.last_gradient_profile["gradient_backend"] == "model_space"
    observable_norm = np.linalg.norm(ms.hamiltonian_s, 2) + (sum(betas) if criterion == "full_vqd" else 0)

    def objective(t, mu):
        if criterion == "energy_only":
            return context.measurements.statevector_energy(theta + [t], ops + [mu])
        return vqd.vqd_cost(context, theta + [t], ops + [mu], previous, betas,
                            overlap_shots=1, energy_shots=1, n_jobs=1, cost_mode="statevector_exact")

    for mu in candidates:
        generator_norm = max(abs(ms.generator(mu).eigenvalues))
        for step in (1e-3, 1e-4, 1e-5):
            numeric = (objective(step, mu) - objective(-step, mu)) / (2 * step)
            # Central-difference third-derivative bound for unitary evolution.
            tolerance = step ** 2 * (2 * generator_norm) ** 3 * observable_norm / 6 + 1e-8
            assert abs(scores[mu] - numeric) <= tolerance, (criterion, excited, mu, step, scores[mu], numeric)


@pytest.fixture
def toy(monkeypatch):
    calls = []
    def cost(context, theta, ops, previous, betas, **kwargs):
        return -float(sum(theta))
    def scan(theta, ops, candidates, *args, **kwargs):
        calls.append(list(ops))
        scores = {mu: -float(3 - mu) for mu in candidates}
        return SimpleNamespace(candidates=candidates, g_energy=scores, g_total=scores)
    context = SimpleNamespace(
        available_workers=1, base_operator_indices=(0, 1, 2), build_context_seconds=0.,
        reference=SimpleNamespace(occupied_orbitals=[], diagonal_energy=0.0),
        hamiltonian=SimpleNamespace(exact_energies=[-3., -2.], exact_vectors=None),
        measurements=SimpleNamespace(scan_candidate_gradients=scan, last_gradient_profile={},
                                     exact_overlap=lambda *args: 1.,
                                     model_space_states=lambda *args: [np.ones(1) for _ in args]))
    monkeypatch.setattr(vqd, "vqd_cost", cost)
    monkeypatch.setattr(vqd, "evaluate_energy", lambda context, theta, ops, **kw: -float(sum(theta)))
    return context, calls, cost


def provisional_optimizer(cost, change):
    def optimize(context, theta, ops, previous, betas, **kwargs):
        optimized = list(theta)
        optimized[-1] = change(list(ops), kwargs["new_parameter_gradient"])
        value = cost(context, optimized, ops, previous, betas)
        start = cost(context, theta, ops, previous, betas)
        result = OptimizeResult(success=True, message="provisional", nfev=1,
            raw_x=np.array(optimized), raw_fun=value, end_cost=value, start_cost=start,
            optimizer_initial_x=np.array(theta), descent_trials=[], accepted_source="best_evaluated")
        return optimized, value, result
    return optimize


@pytest.mark.parametrize("limit,reason,count", [(1, "candidate_attempt_limit", 1), (10, "no_improving_candidate", 3)])
def test_rejected_layers_never_change_ansatz(toy, monkeypatch, limit, reason, count):
    context, scans, cost = toy
    monkeypatch.setattr(vqd, "optimize_ansatz", provisional_optimizer(cost, lambda *args: 0.0))
    result = vqd.run_adapt_vqd(context, max_adapt_iter=20, max_candidate_attempts=limit, n_jobs=1)
    assert result["ops"] == result["thetas"] == []
    assert result["n_layers"] == result["n_parameters"] == 0
    assert result["vqd_objective"] == 0
    assert result["stop_reason"] == reason
    assert result["n_iterations"] == count
    assert scans == [[]]
    assert len(result["history"]) == 1
    record = result["history"][0]
    assert not record["accepted_candidate"] and not record["layer_added"]
    assert record["new_theta"] is None
    assert len(record["candidate_attempts"]) == count
    assert all(not a["accepted_candidate"] for a in record["candidate_attempts"])


def test_next_ranked_candidate_is_accepted_without_rescanning(toy, monkeypatch):
    context, scans, cost = toy
    monkeypatch.setattr(vqd, "optimize_ansatz", provisional_optimizer(cost, lambda ops, grad: 0 if ops[-1] == 0 else 0.2))
    result = vqd.run_adapt_vqd(context, max_adapt_iter=1, n_jobs=1)
    assert result["ops"] == [1] and result["thetas"] == [0.2]
    assert result["vqd_objective"] == -0.2
    record = result["history"][0]
    assert record["gradient_winner_operator_id"] == 0 and record["selected_operator_id"] == 1
    assert record["accepted_candidate"] and record["layer_added"]
    assert [a["operator_id"] for a in record["candidate_attempts"]] == [0, 1]
    assert [a["accepted_candidate"] for a in record["candidate_attempts"]] == [False, True]
    assert scans == [[]]


def test_acceptance_tolerance_is_strict_and_preserves_old_parameters(toy, monkeypatch):
    context, scans, cost = toy
    monkeypatch.setattr(vqd, "optimize_ansatz", provisional_optimizer(cost, lambda ops, grad: 0.2 if len(ops) == 1 else 0.01))
    result = vqd.run_adapt_vqd(context, max_adapt_iter=4, acceptance_tol=0.02, max_candidate_attempts=1, n_jobs=1)
    assert result["ops"] == [0] and result["thetas"] == [0.2]
    assert result["vqd_objective"] == -0.2
    assert result["stop_reason"] == "candidate_attempt_limit"
    assert result["history"][-1]["candidate_improvement"] < 0.02
    assert not result["history"][-1]["layer_added"]


def test_nonconsecutive_reuse_and_monotonicity_with_real_acceptance(toy, monkeypatch):
    context, scans, cost = toy
    monkeypatch.setattr(vqd, "optimize_ansatz", provisional_optimizer(cost, lambda ops, grad: 0.2))
    result = vqd.run_adapt_vqd(context, max_adapt_iter=8, n_jobs=1)
    assert result["ops"] == [0, 1] * 4
    assert len(scans) == 8
    assert result["n_layers"] > len(context.base_operator_indices)
    for record in result["history"]:
        assert record["layer_added"] == record["accepted_candidate"]
        assert record["vqd_cost_after"] < record["vqd_cost_before"] - result["acceptance_tol"]
        assert record["zero_parameter_state_error"] == 0


def test_rejected_reoptimization_restores_every_old_parameter(toy, monkeypatch):
    context, _, cost = toy
    initial_optimizer = provisional_optimizer(cost, lambda *args: 0.2)
    def optimize(context, theta, ops, previous, betas, **kwargs):
        optimized, value, result = initial_optimizer(context, theta, ops, previous, betas, **kwargs)
        if len(ops) > 1:
            optimized = [0.1, 0.0]  # Changes an old parameter and worsens the objective.
            value = cost(context, optimized, ops, previous, betas)
            result.raw_x = np.array(optimized)
            result.raw_fun = result.end_cost = value
        return optimized, value, result
    monkeypatch.setattr(vqd, "optimize_ansatz", optimize)
    result = vqd.run_adapt_vqd(context, max_adapt_iter=3, max_candidate_attempts=1, n_jobs=1)
    assert result["thetas"] == [0.2] and result["ops"] == [0]
    assert result["vqd_objective"] == -0.2


def test_failed_candidate_diagnostics_do_not_partially_commit(toy, monkeypatch):
    context, _, cost = toy
    monkeypatch.setattr(vqd, "optimize_ansatz", provisional_optimizer(cost, lambda *args: 0.2))
    monkeypatch.setattr(vqd, "evaluate_energy", lambda context, theta, ops, **kwargs: np.nan if ops else 0.)
    result = vqd.run_adapt_vqd(context, max_adapt_iter=1, n_jobs=1)
    assert result["stop_reason"] == "numerical_error"
    assert result["thetas"] == result["ops"] == []
    assert not result["history"][0]["layer_added"]


def test_warm_start_tests_descent_with_old_parameters_fixed(monkeypatch):
    evaluated = []
    def cost(context, theta, *args, **kwargs):
        evaluated.append(list(theta))
        return float(theta[-1])
    def minimize(objective, x0, **kwargs):
        np.testing.assert_array_equal(x0, [0.7, -0.1])
        return OptimizeResult(x=x0, fun=objective(x0), success=True, message="seed", nfev=1)
    monkeypatch.setattr(vqd, "vqd_cost", cost)
    monkeypatch.setattr(vqd, "minimize", minimize)
    theta, value, result = vqd.optimize_ansatz(None, [0.7, 0], [0, 1], [], [],
        new_parameter_gradient=1.0, overlap_shots=1, energy_shots=1,
        optimizer_maxiter=10, n_jobs=1, cost_mode="statevector_exact")
    assert theta == [0.7, -0.1] and value == -0.1
    assert [trial["theta"] for trial in result.descent_trials] == [-0.1, -0.01, -0.001, -0.0001]
    assert all(t[0] == 0.7 for t in evaluated)
    assert result.start_cost == result.initial_cost == 0


def test_ranked_candidates_use_canonical_ties_in_diagnostics():
    from vqd_nuclear.gradient_diagnostics import build_diagnostics
    scores = {9: 2.0, 2: 2.0 * (1 - 1e-10), 1: 0.5}
    assert vqd.ranked_candidates([9, 1, 2], scores) == [2, 9, 1]
    scan = SimpleNamespace(candidates=[9, 1, 2], g_energy=scores, g_total=scores,
                           g_penalty=dict.fromkeys(scores, 0.), max_overlap_amplitude=0.)
    assert build_diagnostics(scan, state_index=0, iteration=0).winner_full_vqd == 2
    with pytest.raises(FloatingPointError, match="Non-finite"):
        vqd.ranked_candidates([1], {1: np.nan})


def test_energy_only_ranking_uses_full_vqd_gradient_for_warm_start(toy, monkeypatch):
    context, _, _ = toy
    def scan(theta, ops, candidates, *args):
        return SimpleNamespace(candidates=candidates, g_energy={mu: -3. for mu in candidates},
                               g_total={mu: 2. for mu in candidates})
    context.measurements.scan_candidate_gradients = scan
    def cost(context, theta, ops, previous, betas, **kwargs):
        assert previous and betas == [5.]
        return sum(theta)  # VQD derivative opposes the energy derivative.
    monkeypatch.setattr(vqd, "vqd_cost", cost)
    def change(ops, gradient):
        assert gradient == 2.
        return -0.1
    monkeypatch.setattr(vqd, "optimize_ansatz", provisional_optimizer(cost, change))
    result = vqd.run_adapt_vqd(context, previous_ansatze=[{"thetas": [], "ops": []}], betas=[5.],
                               max_adapt_iter=1, n_jobs=1, selection_criterion="energy_only")
    assert result["vqd_objective"] == -0.1
    assert result["history"][0]["energy_after"] > result["history"][0]["energy_before"]


@pytest.mark.parametrize("noisy", [False, True])
def test_sampled_acceptance_validates_uncertainty_without_exact_monotonicity(toy, monkeypatch, noisy):
    context, _, _ = toy
    # These estimates are independent from the optimizer's trial measurements.
    samples = iter([0., 0., -1.] + ([0., 10., -10., -1., 9., -11.] if noisy else [10.] * 3 + [9.] * 3))
    monkeypatch.setattr(vqd, "vqd_cost", lambda *args, **kwargs: next(samples))
    monkeypatch.setattr(vqd, "optimize_ansatz", provisional_optimizer(lambda *a: -1., lambda *a: 0.1))
    result = vqd.run_adapt_vqd(context, max_adapt_iter=1, max_candidate_attempts=1, n_jobs=1,
                               cost_mode="full_quantum_circuit")
    record = result["history"][0]
    if noisy:
        assert not record["accepted_candidate"] and result["ops"] == []
        assert record["acceptance_margin"] > record["candidate_improvement"]
    else:
        assert record["accepted_candidate"] and result["ops"] == [0]
        assert record["vqd_cost_after"] > record["vqd_cost_before"]


def test_noisy_cost_improvement_does_not_accept_identical_parameters(toy, monkeypatch):
    context, _, _ = toy
    samples = iter([0., 0., -1.] + [10.] * 3 + [9.] * 3)
    monkeypatch.setattr(vqd, "vqd_cost", lambda *args, **kwargs: next(samples))
    monkeypatch.setattr(vqd, "optimize_ansatz", provisional_optimizer(lambda *a: -1., lambda *a: 0.))
    result = vqd.run_adapt_vqd(context, max_adapt_iter=1, max_candidate_attempts=1,
                               n_jobs=1, cost_mode="full_quantum_circuit")
    assert result["ops"] == result["thetas"] == []
    assert result["history"][0]["unchanged_parameters"]
    assert not result["history"][0]["accepted_candidate"]
