"""Reduced ADAPT derivatives against dense evolution and finite differences."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from scipy.linalg import expm

from vqd_nuclear.measurements import MeasurementEngine, apply_qubit_operator_to_state
from vqd_nuclear.model_space import ModelSpaceEngine, ANSATZ_DERIVATIVE_PHASE
from vqd_nuclear.measurements import GRADIENT_PHASE


@pytest.fixture
def engines():
    def op(terms):
        return SimpleNamespace(terms=terms)

    h = op({((0, "Z"),): 0.7, ((1, "Z"),): -0.2})
    pool = [op({((0, "X"), (1, "X")): 0.5, ((0, "Y"), (1, "Y")): 0.5}),
            op({((0, "Z"),): 1}), op({((0, "X"),): 1})]
    matrices = [np.column_stack([apply_qubit_operator_to_state(g, v, 2) for v in np.eye(4)]) for g in pool]

    def state(theta, ops):
        psi = np.array([0, 0, 1, 0], dtype=complex)
        for t, mu in zip(theta, ops):
            psi = expm(-1j * t * matrices[mu]) @ psi
        return psi

    builder = SimpleNamespace(state=Mock(side_effect=state), cache_key=lambda t, ops: (tuple(t), tuple(ops)))
    ms = ModelSpaceEngine(qubit_hamiltonian=h, qubit_generator_pool=pool, indices=[2, 1],
                          reference_occupied_orbitals=[0], n_qubits=2, apply_operator=None)
    options = dict(ansatz_builder=builder, qubit_hamiltonian=h, qubit_generator_pool=pool,
                   n_qubits=2, available_workers=1)
    return MeasurementEngine(**options, model_space=ms), MeasurementEngine(**options), pool


@pytest.mark.parametrize("previous,betas", [([], []), ([{"thetas": [0.37], "ops": [0]}], [2.5])])
def test_scan_components_and_individual_gradients(engines, monkeypatch, previous, betas):
    reduced, full, pool = engines
    theta, ops = [0.17, -0.31, 0.23], [0, 1, 0]  # Nonconsecutive reuse.
    expected = full.scan_candidate_gradients(theta, ops, [0, 1], previous, betas)
    previous_states = [full.ansatz_builder.state(p["thetas"], p["ops"]) for p in previous]
    expected_individual = [full.statevector_vqd_gradient(theta, ops, g, previous_states, betas) for g in pool[:2]]
    reduced.ansatz_builder.state.reset_mock()
    reduced.ansatz_builder.state.side_effect = AssertionError("Full state reconstructed")
    monkeypatch.setattr("vqd_nuclear.measurements.apply_qubit_operator_to_state",
                        Mock(side_effect=AssertionError("Full operator applied")))
    actual = reduced.scan_candidate_gradients(theta, ops, [0, 1], previous, betas)
    for field in ["g_energy", "g_penalty", "g_total"]:
        np.testing.assert_allclose(list(getattr(actual, field).values()), list(getattr(expected, field).values()),
                                   atol=1e-10, rtol=1e-8)
    assert actual.max_overlap_amplitude == pytest.approx(expected.max_overlap_amplitude)
    for mu, g in enumerate(pool[:2]):
        np.testing.assert_allclose(reduced.statevector_vqd_gradient(theta, ops, g, previous_states, betas),
                                   expected_individual[mu], atol=1e-10, rtol=1e-8)
        assert reduced.analytic_vqd_gradient(theta, ops, g, previous, betas) == pytest.approx(actual.g_total[mu])
        assert reduced.statevector_energy_gradient(theta, ops, g) == pytest.approx(actual.g_energy[mu])
        eps = 1e-5
        def cost(t):
            return reduced.statevector_energy(theta + [t], ops + [mu]) + sum(
                beta * reduced.exact_overlap(p["thetas"], p["ops"], theta + [t], ops + [mu])
                for p, beta in zip(previous, betas))
        assert actual.g_total[mu] == pytest.approx((cost(eps) - cost(-eps)) / (2 * eps), abs=1e-9)
    reduced.ansatz_builder.state.assert_not_called()
    assert reduced.last_gradient_profile["gradient_backend"] == "model_space"
    assert GRADIENT_PHASE == ANSATZ_DERIVATIVE_PHASE
    ms = reduced.model_space
    np.testing.assert_allclose(ms.generator(0).effective_generator_action(ms.state(theta, ops)),
                               (ms.state(theta + [1e-6], ops + [0]) - ms.state(theta + [-1e-6], ops + [0])) / 2e-6,
                               atol=1e-9)


def test_zero_energy_gradient_can_have_large_penalty(engines):
    reduced, _, _ = engines
    scan = reduced.scan_candidate_gradients([], [], [0], [{"thetas": [0.37], "ops": [0]}], [2.5])
    assert abs(scan.g_energy[0]) < 1e-12
    assert abs(scan.g_penalty[0]) > 1


@pytest.mark.parametrize("where", ["candidate", "current", "previous", "hamiltonian"])
def test_incompatible_operators_use_complete_scan_without_filtering(engines, where):
    reduced, full, _ = engines
    theta, ops, candidates, previous, betas = [], [], [0, 1], [], []
    if where == "candidate":
        candidates.append(2)
    elif where == "current":
        theta, ops = [0.2], [2]
    elif where == "previous":
        previous, betas = [{"thetas": [0.4], "ops": [2]}], [2.0]
    else:
        reduced.model_space.available = False
    actual = reduced.scan_candidate_gradients(theta, ops, candidates, previous, betas)
    expected = full.scan_candidate_gradients(theta, ops, candidates, previous, betas)
    assert actual == expected
    assert actual.candidates == candidates
    assert reduced.last_gradient_profile["gradient_backend"] == "full_state"
    assert reduced.ansatz_builder.state.called


def test_diagnostics_reorders_reduced_basis(monkeypatch, engines):
    from vqd_nuclear import vqd
    reduced, full, _ = engines
    context = SimpleNamespace(measurements=reduced, ansatz_builder=full.ansatz_builder,
                              problem=SimpleNamespace(n_valence_protons=1, n_valence_neutrons=0),
                              hamiltonian=SimpleNamespace(exact_vectors=np.eye(2), nucleus=None,
                                                          n_qubits=2, orbital_mapping=None))
    monkeypatch.setattr(vqd, "get_hamiltonian_basis_references", lambda *a, **kw: [[1], [0]])
    expected = full.ansatz_builder.state([0.23], [0])[[1, 2]]
    full.ansatz_builder.state.side_effect = AssertionError("Full diagnostics")
    result = vqd.exact_state_diagnostics(context, [0.23], [0])
    np.testing.assert_allclose(result["exact_state_fidelities"], abs(expected) ** 2, atol=1e-12)


@pytest.mark.parametrize("excited", [False, True])
def test_nuclear_pool_gradients_match_at_identical_parameters(excited):
    pytest.importorskip("pynucshell")
    from vqd_nuclear import NumericalConfig, ProblemConfig, RunConfig, vqd

    context = vqd.build_context(ProblemConfig(shell="p", n_valence_protons=0, n_valence_neutrons=2),
                                RunConfig(n_states=2, n_workers=1), NumericalConfig(n_jobs=1))
    engine = context.measurements
    ms = engine.model_space
    theta, ops = [0.31, -0.27, 0.19], [6, 0, 6]
    previous = [{"thetas": [0.23, -0.4], "ops": [2, 4]}] if excited else []
    betas = [9.0] if excited else []
    candidates = list(context.base_operator_indices)
    reduced = engine.scan_candidate_gradients(theta, ops, candidates, previous, betas)
    engine.model_space = None
    full = engine.scan_candidate_gradients(theta, ops, candidates, previous, betas)
    engine.model_space = ms
    for field in ("g_energy", "g_penalty", "g_total"):
        np.testing.assert_allclose(list(getattr(reduced, field).values()), list(getattr(full, field).values()),
                                   atol=1e-10, rtol=1e-8)


@pytest.mark.slow
@pytest.mark.parametrize("criterion", ["full_vqd", "energy_only"])
def test_nuclear_adapt_trajectory_matches_full_route(monkeypatch, criterion):
    pytest.importorskip("pynucshell")
    from vqd_nuclear import NumericalConfig, ProblemConfig, RunConfig, vqd

    context = vqd.build_context(ProblemConfig(shell="p", n_valence_protons=0, n_valence_neutrons=2),
                                RunConfig(n_states=2, n_workers=1), NumericalConfig(n_jobs=1))
    ms = context.measurements.model_space
    options = dict(max_adapt_iter=3, optimizer_maxiter=250, n_jobs=1, selection_criterion=criterion,
                   grad_tol=1e-5, energy_improvement_tol=0, new_theta_tol=0,
                   cobyla_rhobeg=0.3, cobyla_rhoend=1e-8, collect_gradient_diagnostics=True)
    context.measurements.model_space = None
    full = []
    for state in range(2):
        full.append(vqd.run_adapt_vqd(context, previous_ansatze=full, betas=[5.0] * state, **options))
    context.measurements.clear_cache()
    context.measurements.model_space = ms
    monkeypatch.setattr(context.ansatz_builder, "state", Mock(side_effect=AssertionError("Full trajectory")))
    monkeypatch.setattr("vqd_nuclear.measurements.apply_qubit_operator_to_state",
                        Mock(side_effect=AssertionError("Full trajectory operator")))
    reduced = []
    for state in range(2):
        reduced.append(vqd.run_adapt_vqd(context, previous_ansatze=reduced, betas=[5.0] * state, **options))
    for actual, expected in zip(reduced, full):
        assert actual["ops"] == expected["ops"]
        assert actual["stop_reason"] == expected["stop_reason"]
        assert actual["energy"] == pytest.approx(expected["energy"], abs=1e-7)
        assert actual["vqd_objective"] == pytest.approx(expected["vqd_objective"], abs=1e-7)
        np.testing.assert_allclose(actual["exact_state_fidelities"], expected["exact_state_fidelities"], atol=1e-7)
        assert len(actual["history"]) == len(expected["history"])
        assert len(actual["gradient_diagnostics"]) == len(expected["gradient_diagnostics"])
        for a, b in zip(actual["gradient_diagnostics"], expected["gradient_diagnostics"]):
            assert a["winner_full_vqd"] == b["winner_full_vqd"]
            for field in ("g_energy", "g_penalty", "g_total"):
                # Independent COBYLA runs differ slightly at their endpoints;
                # identical-parameter derivatives are checked tightly above.
                np.testing.assert_allclose(np.fromstring(a[field], sep=";"), np.fromstring(b[field], sep=";"),
                                           atol=options["grad_tol"], rtol=1e-6)
        for a, b in zip(actual["history"], expected["history"]):
            assert a["selected_operator_id"] == b["selected_operator_id"]
            if "zero_parameter_state_error" in a:
                assert a["zero_parameter_state_error"] < 1e-10
            assert a["energy_after"] == pytest.approx(b["energy_after"], abs=1e-7)
            assert a["max_abs_gradient"] == pytest.approx(b["max_abs_gradient"], abs=options["grad_tol"])
    context.ansatz_builder.state.assert_not_called()
