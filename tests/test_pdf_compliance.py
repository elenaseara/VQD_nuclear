"""Revised PDF sections 3-5 and 7-9; selection-gradient modes stay unchanged."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

pytest.importorskip("pynucshell")
pytest.importorskip("qibo")

from pynucshell.operators import QubitOperator
from vqd_nuclear import FullQuantumLimits, NumericalConfig, ProblemConfig, RunConfig
from vqd_nuclear import vqd
from vqd_nuclear.ansatz import (AnsatzBuilder, build_reference_circuit, diagonal_energy,
                               find_reference_from_hamiltonian_basis, get_hamiltonian_basis_references)
from vqd_nuclear.measurements import MeasurementEngine, apply_qubit_operator_to_state
from vqd_nuclear.orbitals import OrbitalMapping


@pytest.fixture(scope="module")
def context():
    return vqd.build_context(
        ProblemConfig(shell="p", n_valence_protons=0, n_valence_neutrons=2, total_j=0),
        RunConfig(n_states=3, n_workers=1, seed=47433), NumericalConfig(n_jobs=1),
    )


def stationary_optimizer(objective, x0, **kwargs):
    return OptimizeResult(x=x0.copy(), fun=objective(x0), success=True, message="stationary", nfev=1)


def fixed_scan(thetas, ops, candidates, *args):
    scores = {mu: float(100 - mu) for mu in candidates}
    return SimpleNamespace(candidates=candidates, g_energy=scores, g_total=scores)


def test_base_pool_does_not_depend_on_reference_or_probe(monkeypatch):
    original = vqd.find_reference_from_hamiltonian_basis
    contexts = []
    for use_other in (False, True):
        def reference(problem, nucleus, **kwargs):
            result = original(problem, nucleus, **kwargs)
            if use_other:
                refs = get_hamiltonian_basis_references(nucleus, n_qubits=kwargs["n_qubits"], n_valence_particles=2)
                other = refs[-1]
                return replace(result, occupied_orbitals=list(other), basis_index=len(refs) - 1,
                               diagonal_energy=diagonal_energy(other, kwargs["qubit_hamiltonian"]))
            return result

        monkeypatch.setattr(vqd, "find_reference_from_hamiltonian_basis", reference)
        # Context construction must never apply ansatz circuits to filter the pool.
        monkeypatch.setattr(AnsatzBuilder, "state", lambda *args: pytest.fail("Reference activity probe"))
        contexts.append(vqd.build_context(
            ProblemConfig(shell="p", n_valence_protons=0, n_valence_neutrons=2),
            RunConfig(n_states=1, n_workers=1), NumericalConfig(n_jobs=1, use_model_space=False)))
    assert contexts[0].reference.occupied_orbitals != contexts[1].reference.occupied_orbitals
    assert contexts[0].base_operator_indices == contexts[1].base_operator_indices
    assert [op.terms for op in contexts[0].hamiltonian.qubit_generator_pool] == [
        op.terms for op in contexts[1].hamiltonian.qubit_generator_pool]


def test_initially_inactive_operator_remains_available(context):
    psi = context.ansatz_builder.state([], [])
    generators = context.hamiltonian.qubit_generator_pool
    initially_inactive = [mu for mu, g in enumerate(generators)
                          if np.linalg.norm(apply_qubit_operator_to_state(g, psi, context.hamiltonian.n_qubits)) < 1e-10]
    assert initially_inactive
    found = False
    for first in context.base_operator_indices:
        changed = context.ansatz_builder.state([0.3], [first])
        for mu in initially_inactive:
            if np.linalg.norm(apply_qubit_operator_to_state(generators[mu], changed, context.hamiltonian.n_qubits)) > 1e-5:
                assert mu in vqd.eligible_operator_indices(vqd.get_candidate_operators(context), [first])
                found = True
    assert found


def test_minimum_reference_and_canonical_ties_do_not_use_pool():
    mapping = OrbitalMapping({0: 0, 1: 1, 2: 2, 3: 3}, (1, 1, -1, -1), (1, -1, 1, -1), 1, 1)
    nucleus = SimpleNamespace(get_basis=lambda: SimpleNamespace(mb=lambda: [(1, 2), (0, 3)]),
                              get_space=lambda: SimpleNamespace(isospin=[1, 1, -1, -1]))
    options = dict(n_qubits=4, orbital_mapping=mapping)
    problem = ProblemConfig(shell="p", n_valence_protons=1, n_valence_neutrons=1)
    minimum = find_reference_from_hamiltonian_basis(problem, nucleus,
                qubit_hamiltonian=QubitOperator(((0, "Z"),), 1.0), **options)
    assert minimum.occupied_orbitals == [0, 3]
    assert minimum.diagonal_energy == -1.0
    # This H has an entirely disconnected, degenerate determinant space.
    for basis in ([(1, 2), (0, 3)], [(0, 3), (1, 2)]):
        nucleus.get_basis = lambda: SimpleNamespace(mb=lambda: basis)
        tied = find_reference_from_hamiltonian_basis(problem, nucleus, qubit_hamiltonian=QubitOperator((), -2), **options)
        assert tied.occupied_orbitals == [0, 3]
        assert tied.diagonal_energy == -2


def test_preparation_and_parameter_mapping(context):
    n = context.hamiltonian.n_qubits
    prepared = np.asarray(build_reference_circuit(n, [0, 3])().state())
    assert np.argmax(abs(prepared)) == (1 << (n - 1)) + (1 << (n - 4))
    theta, ops = [0.23, -0.41, 0.17], [6, 0, 6]
    psi = context.ansatz_builder.state(theta, ops)
    np.testing.assert_allclose(context.ansatz_builder.state(theta + [0], ops + [3]), psi, atol=1e-12)
    model = context.measurements.model_space
    np.testing.assert_allclose(model.state(theta, ops), psi[model.indices], atol=1e-12)
    with pytest.raises(ValueError, match="one parameter"):
        context.ansatz_builder.build(theta[:-1], ops)
    with pytest.raises(ValueError, match="one parameter"):
        model.state(theta[:-1], ops)


@pytest.mark.parametrize("deflated", [False, True])
@pytest.mark.parametrize("best_trial", [None, [0.1, 0.0]])
def test_reoptimization_keeps_lowest_cost_and_reevaluates_endpoint(monkeypatch, deflated, best_trial):
    def cost(context, thetas, ops, previous, betas, **kwargs):
        return (-thetas[0] + 10 * thetas[0] ** 2) if deflated else thetas[0] ** 2

    def minimize(objective, x0, **kwargs):
        np.testing.assert_array_equal(x0, [0.2, 0.0])
        if best_trial is not None:
            objective(np.array(best_trial))
        # Deliberately lie about fun; acceptance must evaluate the actual point.
        return OptimizeResult(x=np.array([0.8, 0.5]), fun=-100.0, success=True, message="done", nfev=2)

    monkeypatch.setattr(vqd, "vqd_cost", cost)
    monkeypatch.setattr(vqd, "minimize", minimize)
    previous = [{"thetas": [], "ops": []}] if deflated else []
    theta, value, result = vqd.optimize_ansatz(None, [0.2, 0.0], [6, 0], previous, [10.0] if deflated else [],
                    overlap_shots=1, energy_shots=1, optimizer_maxiter=10, n_jobs=1, cost_mode="statevector_exact")
    assert theta == (best_trial or [0.2, 0.0])
    assert value <= result.start_cost
    assert result.end_cost > value
    assert result.raw_fun == -100
    assert result.accepted_source == ("best_evaluated" if best_trial else "initial")


def test_nonconsecutive_reuse_and_more_layers_than_pool(context, monkeypatch):
    monkeypatch.setattr(context.measurements, "scan_candidate_gradients", fixed_scan)
    monkeypatch.setattr(vqd, "minimize", stationary_optimizer)
    pool = context.base_operator_indices
    result = vqd.run_adapt_vqd(context, max_adapt_iter=10, n_jobs=1, new_theta_tol=0, energy_improvement_tol=0)
    assert result["ops"] == [0, 1] * 5
    assert result["n_iterations"] == result["n_layers"] == 10 > len(pool)
    assert result["stop_reason"] == "max_iterations"
    assert context.base_operator_indices == pool
    assert [h["eligible_pool_size"] for h in result["history"]] == [8] + [7] * 9


@pytest.mark.parametrize("method", ["analytic", "finite_difference"])
def test_reuse_applies_to_both_existing_gradient_methods(context, monkeypatch, method):
    monkeypatch.setattr(context.measurements, "scan_candidate_gradients", fixed_scan)
    monkeypatch.setattr(vqd, "finite_difference_gradient", lambda task: (task[1], float(100 - task[1])))
    used = list(context.base_operator_indices)
    winner, _, scan = vqd.select_next_operator(
        context, [0.0] * len(used), used, [], [], eps=1e-6, max_candidates=None, n_jobs=1,
        overlap_shots=1, energy_shots=1, cost_mode="statevector_exact", gradient_method=method)
    assert winner == used[0]
    assert used[-1] not in scan.candidates
    assert set(scan.candidates) == set(used[:-1])


def test_existing_selection_modes_are_preserved(context, monkeypatch):
    def scan(theta, ops, candidates, *args):
        energy = dict.fromkeys(candidates, 0.0)
        total = dict.fromkeys(candidates, 0.0)
        energy[0], total[0], total[1] = 3.0, 1.0, 4.0
        return SimpleNamespace(candidates=candidates, g_energy=energy, g_total=total)

    monkeypatch.setattr(context.measurements, "scan_candidate_gradients", scan)
    options = dict(eps=1e-6, max_candidates=None, n_jobs=1, overlap_shots=1,
                   energy_shots=1, cost_mode="statevector_exact")
    assert vqd.select_next_operator(context, [], [], [], [], selection_criterion="energy_only", **options)[0] == 0
    assert vqd.select_next_operator(context, [], [], [], [], selection_criterion="full_vqd", **options)[0] == 1


def test_optimizer_exception_is_reported_without_losing_state(context, monkeypatch):
    def optimizer(*args, **kwargs):
        raise RuntimeError("optimizer aborted")

    monkeypatch.setattr(context.measurements, "scan_candidate_gradients", fixed_scan)
    monkeypatch.setattr(vqd, "minimize", optimizer)
    result = vqd.run_adapt_vqd(context, n_jobs=1)
    assert result["stop_reason"] == "optimizer_failure"
    assert result["n_iterations"] == 1
    assert result["n_layers"] == 0
    assert result["stop_details"]["accepted_candidate"] is False
    assert result["stop_details"]["optimizer_message"] == "optimizer aborted"


@pytest.mark.parametrize("options,reason,steps", [
    ({"max_adapt_iter": 0}, "max_iterations", 0),
    ({"max_adapt_iter": 1}, "max_iterations", 1),
    ({"grad_tol": 1e10}, "gradient_converged", 0),
    ({"max_candidates": 0}, "no_eligible_operators", 0),
    ({"max_candidates": 1, "new_theta_tol": 0, "energy_improvement_tol": 0}, "no_eligible_operators", 1),
])
def test_stopping_is_explicit_even_without_history(context, monkeypatch, options, reason, steps):
    monkeypatch.setattr(context.measurements, "scan_candidate_gradients", fixed_scan)
    monkeypatch.setattr(vqd, "minimize", stationary_optimizer)
    result = vqd.run_adapt_vqd(context, n_jobs=1, store_history=False, **options)
    assert result["stop_reason"] == reason
    assert result["n_iterations"] == steps
    assert result["history"] == []


@pytest.mark.parametrize("success,reason", [(True, "energy_stagnation"), (False, "optimizer_failure")])
def test_stagnation_and_optimizer_failure_are_distinct(context, monkeypatch, success, reason):
    def optimizer(objective, x0, **kwargs):
        return OptimizeResult(x=x0.copy(), fun=objective(x0), success=success, message="stuck", nfev=1)

    monkeypatch.setattr(context.measurements, "scan_candidate_gradients", fixed_scan)
    monkeypatch.setattr(vqd, "minimize", optimizer)
    result = vqd.run_adapt_vqd(context, n_jobs=1)
    assert result["stop_reason"] == reason
    assert result["history"][-1]["optimizer_message"] == "stuck"


def test_nonfinite_gradient_is_numerical_error(context, monkeypatch):
    def scan(theta, ops, candidates, *args):
        scores = dict.fromkeys(candidates, np.nan)
        return SimpleNamespace(candidates=candidates, g_energy=scores, g_total=scores)

    monkeypatch.setattr(context.measurements, "scan_candidate_gradients", scan)
    result = vqd.run_adapt_vqd(context, n_jobs=1)
    assert result["stop_reason"] == "numerical_error"
    assert result["n_layers"] == 0
    assert "Non-finite" in result["stop_details"]["error"]


def test_identity_violation_is_reported_without_adding_layer(context, monkeypatch):
    original = context.ansatz_builder.state

    def state(theta, ops):
        psi = original(theta, ops)
        return -psi if ops else psi

    monkeypatch.setattr(context.measurements, "scan_candidate_gradients", fixed_scan)
    monkeypatch.setattr(context.ansatz_builder, "state", state)
    result = vqd.run_adapt_vqd(context, n_jobs=1)
    assert result["stop_reason"] == "numerical_error"
    assert result["n_layers"] == 0
    assert result["stop_details"]["zero_parameter_state_error"] > 1


def test_existing_gradient_matches_actual_circuit_finite_difference(context):
    theta, ops = [0.31, -0.27], [6, 0]
    previous = [{"thetas": [0.23], "ops": [2]}, {"thetas": [-0.4], "ops": [4]}]
    for prev, betas in (([], []), (previous, [9.0, 13.0])):
        scan = context.measurements.scan_candidate_gradients(theta, ops, list(context.base_operator_indices), prev, betas)
        for mu in context.base_operator_indices:
            _, numeric = vqd.finite_difference_gradient((context, mu, theta, ops, prev, betas, 1e-6, 1, 1, "statevector_exact"))
            assert scan.g_total[mu] == pytest.approx(numeric, abs=3e-7, rel=2e-6)


@pytest.mark.parametrize("strategy", ["paper", "qwc"])
def test_sampled_measurement_interfaces_preserved(strategy):
    builder = SimpleNamespace(build=lambda *args: build_reference_circuit(1, []))
    engine = MeasurementEngine(ansatz_builder=builder, qubit_hamiltonian=QubitOperator(((0, "Z"),), 1),
                               n_qubits=1, available_workers=1, measurement_strategy=strategy)
    assert engine.sampled_energy([], [], nshots=20, n_jobs=1) == 1.0
    assert engine.destructive_swap_overlap([], [], [], [], nshots=20) == 1.0


@pytest.mark.parametrize("protons,neutrons", [(0, 2), (2, 0), (1, 1)])
def test_model_space_has_physical_orbitals_and_correct_matrix(protons, neutrons):
    ctx = vqd.build_context(ProblemConfig(shell="p", n_valence_protons=protons, n_valence_neutrons=neutrons),
                            RunConfig(n_states=3, n_workers=1), NumericalConfig(n_jobs=1))
    assert ctx.hamiltonian.n_qubits == 6 * (int(protons > 0) + int(neutrons > 0))
    np.testing.assert_allclose(ctx.measurements.model_space.hamiltonian_s, ctx.hamiltonian.matrix.toarray(), atol=1e-9)
    assert ctx.hamiltonian.pool_metadata["preserves"] == ["N_protons", "N_neutrons", "M"]
    assert ctx.hamiltonian.pool_metadata["exact_J_guaranteed"] is False


@pytest.mark.slow
def test_he6_three_states_with_pdf_settings_are_reproducible(tmp_path):
    from vqd_nuclear.io import save_outputs
    from vqd_nuclear.config import OutputConfig
    import json

    problem = ProblemConfig(shell="p", n_valence_protons=0, n_valence_neutrons=2)
    numerical = NumericalConfig(max_adapt_iter=20, optimizer_maxiter=1500, grad_tol=1e-5,
                                energy_improvement_tol=1e-6, new_theta_tol=1e-5, n_jobs=1)
    runs = [vqd.run_workflow(problem, RunConfig(n_states=3, n_workers=1, seed=47433), numerical, FullQuantumLimits())
            for _ in range(2)]
    for a, b in zip(runs[0]["statevector_results"], runs[1]["statevector_results"]):
        assert a["ops"] == b["ops"]
        assert a["energy"] == pytest.approx(b["energy"], abs=1e-12)
        assert a["pool_size_initial"] == a["pool_size_after_reference"] == 8
        assert all(x != y for x, y in zip(a["ops"], a["ops"][1:]))
        assert sum(a["exact_state_fidelities"]) == pytest.approx(1, abs=1e-9)
        assert a["stop_reason"] in {"energy_stagnation", "max_iterations", "gradient_converged"}
        layers = [h for h in a["history"] if h["layer_added"]]
        for i, h in enumerate(layers):
            assert h["vqd_cost_after"] <= h["vqd_cost_before"] + numerical.identity_tol
            assert abs(h["zero_parameter_objective_error"]) <= numerical.identity_tol
            assert h["zero_parameter_state_error"] <= numerical.identity_tol
            assert h["eligible_pool_size"] == (8 if i == 0 else 7)
            assert h["initial_thetas"][-1] == 0
            if i:
                assert h["initial_thetas"][:-1] == layers[i - 1]["thetas"]
    paths = save_outputs(OutputConfig(results_dir=tmp_path), run_id="he6", problem=problem,
                         run=runs[0]["context"].run, numerical=numerical, full_limits=FullQuantumLimits(), workflow_output=runs[0])
    raw = json.loads(paths["raw_json"].read_text(encoding="utf-8"), parse_constant=lambda x: pytest.fail(x))
    assert raw["pool"]["reference_filter_applied"] is False
    assert raw["reference"]["occupied_orbitals"] == [0, 3]
    assert "no_eligible_operators" not in [r["stop_reason"] for r in raw["statevector_results"]]
