"""Direct Pauli projection versus the original dense reference (small systems)."""
from itertools import combinations
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import expm

from vqd_nuclear.measurements import apply_qubit_operator_to_state, MeasurementEngine
from vqd_nuclear.model_space import LEAK_TOL, ModelSpaceEngine, ProjectedGenerator, project_operator


def dense_project_operator(operator, indices, n_qubits, apply_operator=apply_qubit_operator_to_state):
    """Original implementation, retained only as a small-system test oracle."""
    indices = np.asarray(indices, dtype=np.int64)
    dimension = 2 ** n_qubits
    projected = np.zeros((len(indices), len(indices)), dtype=complex)
    outside = np.ones(dimension, dtype=bool)
    outside[indices] = False
    vector = np.zeros(dimension, dtype=complex)
    leakage = 0.0
    for column, source in enumerate(indices):
        vector[:] = 0.0
        vector[int(source)] = 1.0
        image = apply_operator(operator, vector, n_qubits)
        projected[:, column] = image[indices]
        if outside.any():
            leakage = max(leakage, float(np.max(np.abs(image[outside]))))
    return projected, leakage


def operator(terms):
    return SimpleNamespace(terms=terms)


@pytest.mark.parametrize("pauli,matrix", [
    ("I", [[1, 0], [0, 1]]), ("X", [[0, 1], [1, 0]]),
    ("Y", [[0, -1j], [1j, 0]]), ("Z", [[1, 0], [0, -1]]),
])
def test_one_qubit_pauli_phases(pauli, matrix):
    projected, leakage = project_operator(operator({((0, pauli),): 1}), [0, 1], 1, None)
    np.testing.assert_array_equal(projected, matrix)
    assert leakage == 0.0


@pytest.mark.parametrize("indices", [[5, 0, 3], list(range(8)), [0], []])
def test_complex_sums_match_dense_projection_and_preserve_basis_order(indices):
    op = operator({(): 0.2 + 0.1j, ((0, "X"),): -0.7j,
                   ((0, "Y"), (2, "Z")): 0.4 + 0.2j,
                   ((0, "X"), (1, "Y"), (2, "Y")): -0.31,
                   ((1, "I"), (2, "Z")): 1.3})
    expected, leakage = dense_project_operator(op, indices, 3)
    actual, actual_leakage = project_operator(op, indices, 3, None)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    assert actual_leakage == pytest.approx(leakage, abs=1e-12)


def test_qibo_big_endian_and_multi_bit_flips():
    op = operator({((0, "Y"), (2, "X")): 1})
    projected, leakage = project_operator(op, [5, 0], 3, None)
    np.testing.assert_array_equal(projected, [[0, 1j], [-1j, 0]])
    assert leakage == 0


def test_external_amplitudes_cancel_before_measuring_leakage():
    # Both strings send |00> outside S to |10>, with opposite amplitudes.
    op = operator({((0, "X"),): 1, ((0, "X"), (1, "Z")): -1, (): 3})
    projected, leakage = project_operator(op, [0], 2, None)
    np.testing.assert_array_equal(projected, [[3]])
    assert leakage == 0


@pytest.mark.parametrize("amplitude", [np.nextafter(LEAK_TOL, 0), np.nextafter(LEAK_TOL, np.inf)])
def test_leak_threshold_decision_is_unchanged(amplitude):
    op = operator({((0, "X"),): amplitude})
    dense, leakage = dense_project_operator(op, [0], 1)
    projected, actual_leakage = project_operator(op, [0], 1, None)
    np.testing.assert_array_equal(projected, dense)
    assert actual_leakage == leakage
    generator = ProjectedGenerator.from_operator(op, [0], 1, None)
    assert generator.preserves_space == (leakage <= LEAK_TOL)


def test_hermiticity_and_projected_unitary_evolution():
    op = operator({((0, "X"), (1, "X")): 0.5, ((0, "Y"), (1, "Y")): 0.5})
    indices = np.array([2, 1])
    projected, leakage = project_operator(op, indices, 2, None)
    np.testing.assert_array_equal(projected, projected.conj().T)
    assert leakage == 0
    generator = ProjectedGenerator.from_operator(op, indices, 2, None)
    assert generator.preserves_space
    initial = np.array([1, 1j]) / np.sqrt(2)
    full, _ = dense_project_operator(op, np.arange(4), 2)
    full_state = np.zeros(4, dtype=complex)
    full_state[indices] = initial
    for theta in [-0.4, 0, 0.71]:
        actual = generator.apply(theta, initial)
        expected = (expm(-1j * theta * full) @ full_state)[indices]
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        assert np.vdot(actual, actual).real == pytest.approx(1, abs=1e-12)
    with pytest.raises(ValueError, match="hermitica"):
        ProjectedGenerator.from_operator(operator({((0, "Z"),): 1j}), [0, 1], 1, None)


def test_projection_never_uses_full_state_callback_or_arrays(monkeypatch):
    import vqd_nuclear.model_space as model_space

    def forbidden(*args, **kwargs):
        pytest.fail("Projection used a full-state operation")

    original_zeros = np.zeros

    def small_zeros(shape, *args, **kwargs):
        assert shape == (2, 2)
        return original_zeros(shape, *args, **kwargs)

    monkeypatch.setattr(model_space.np, "zeros", small_zeros)
    monkeypatch.setattr(model_space.np, "ones", forbidden)
    projected, leakage = project_operator(operator({((0, "Y"), (39, "X")): 2}),
                                         [0, (1 << 39) | 1], 40, forbidden)
    np.testing.assert_array_equal(projected, [[0, -2j], [2j, 0]])
    assert leakage == 0


@pytest.mark.parametrize("indices,n_qubits,terms", [
    ([0, 0], 2, {}), ([-1], 2, {}), ([4], 2, {}), ([[0]], 2, {}),
    ([0.5], 2, {}), ([0], -1, {}), ([0], 1.5, {}),
    ([0], 2, {((2, "X"),): 1}), ([0], 2, {((-1, "Z"),): 1}),
    ([0], 2, {((0.5, "X"),): 1}), ([0], 2, {((0, "A"),): 1}),
    ([0], 2, {(): np.inf}), ([0], 2, {(): complex(0, np.nan)}),
])
def test_invalid_projection_inputs_are_rejected(indices, n_qubits, terms):
    with pytest.raises(ValueError):
        project_operator(operator(terms), indices, n_qubits, None)


def test_leaking_hamiltonian_and_generator_keep_full_state_fallback():
    h = operator({((0, "Z"),): 1})
    leaking = operator({((0, "X"),): 1})
    model = ModelSpaceEngine(qubit_hamiltonian=h, qubit_generator_pool=[leaking], indices=[0, 3],
                             reference_occupied_orbitals=[], n_qubits=2, apply_operator=None)
    assert model.available
    assert not model.can_handle([0])
    calls = []

    def state(thetas, ops):
        calls.append(list(ops))
        return np.array([0, 0, 1, 0], dtype=complex)

    builder = SimpleNamespace(state=state, cache_key=lambda theta, ops: (tuple(theta), tuple(ops)))
    engine = MeasurementEngine(ansatz_builder=builder, qubit_hamiltonian=h, n_qubits=2,
                               available_workers=1, model_space=model)
    assert engine.statevector_energy([0.1], [0]) == -1
    assert calls == [[0]]
    unavailable = ModelSpaceEngine(qubit_hamiltonian=leaking, qubit_generator_pool=[], indices=[0],
                                   reference_occupied_orbitals=[], n_qubits=2, apply_operator=None)
    assert not unavailable.available
    assert not unavailable.can_handle([])


def test_nuclear_hamiltonian_generators_and_pauli_commutation():
    pytest.importorskip("pynucshell")
    from vqd_nuclear import NumericalConfig, ProblemConfig, RunConfig
    from vqd_nuclear.vqd import build_context

    context = build_context(ProblemConfig(shell="p", n_valence_protons=0, n_valence_neutrons=2),
                            RunConfig(n_states=3, n_workers=1), NumericalConfig(n_jobs=1))
    model = context.measurements.model_space
    for op in [context.hamiltonian.qubit_hamiltonian, *context.hamiltonian.qubit_generator_pool]:
        actual, leakage = project_operator(op, model.indices, model.n_qubits, None)
        expected, expected_leakage = dense_project_operator(op, model.indices, model.n_qubits)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)
        assert leakage == pytest.approx(expected_leakage, abs=1e-12)
        np.testing.assert_allclose(actual, actual.conj().T, atol=1e-12)
    for op in context.hamiltonian.qubit_generator_pool:
        for first, second in combinations(op.terms, 2):
            a, b = dict(first), dict(second)
            assert sum(a[q] != b[q] for q in a.keys() & b.keys() if a[q] != "I" and b[q] != "I") % 2 == 0


@pytest.mark.slow
def test_adapt_vqd_trajectory_matches_original_projection(monkeypatch):
    pytest.importorskip("pynucshell")
    from vqd_nuclear import FullQuantumLimits, NumericalConfig, ProblemConfig, RunConfig, vqd
    import vqd_nuclear.model_space as model_space

    problem = ProblemConfig(shell="p", n_valence_protons=0, n_valence_neutrons=2)
    options = NumericalConfig(max_adapt_iter=4, optimizer_maxiter=250, n_jobs=1,
                              collect_gradient_diagnostics=True, cobyla_rhobeg=0.3, cobyla_rhoend=1e-8)

    def run():
        return vqd.run_workflow(problem, RunConfig(n_states=3, n_workers=1, seed=47433), options, FullQuantumLimits())

    direct = run()
    monkeypatch.setattr(model_space, "project_operator", dense_project_operator)
    original = run()
    assert direct["beta_check"] == original["beta_check"]
    assert len(direct["statevector_results"]) == len(original["statevector_results"]) == 6
    for new, old in zip(direct["statevector_results"], original["statevector_results"]):
        assert new["ops"] == old["ops"]
        assert new["stop_reason"] == old["stop_reason"]
        assert new["energy"] == pytest.approx(old["energy"], abs=1e-10)
        assert new["vqd_objective"] == pytest.approx(old["vqd_objective"], abs=1e-10)
        np.testing.assert_allclose(new["thetas"], old["thetas"], atol=1e-10, rtol=0)
        for new_step, old_step in zip(new["history"], old["history"]):
            for field in ("energy_after", "vqd_cost_after", "max_abs_gradient"):
                assert new_step[field] == pytest.approx(old_step[field], abs=1e-10)
            np.testing.assert_allclose(new_step.get("overlaps_with_previous", []),
                                       old_step.get("overlaps_with_previous", []), atol=1e-10, rtol=0)
        for new_scan, old_scan in zip(new["gradient_diagnostics"], old["gradient_diagnostics"]):
            assert new_scan["winner_full_vqd"] == old_scan["winner_full_vqd"]
            for field in ("g_energy", "g_penalty", "g_total"):
                np.testing.assert_allclose(np.fromstring(new_scan[field], sep=";"),
                                           np.fromstring(old_scan[field], sep=";"), atol=1e-10, rtol=0)
