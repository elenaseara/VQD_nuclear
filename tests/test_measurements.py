import importlib.util

import numpy as np
import pytest

pytest.importorskip("qibo")

from vqd_nuclear.measurements import bitstring_to_array, statevector_pauli_expectation


def test_bitstring_to_array_string():
    assert np.array_equal(bitstring_to_array("0101", 4), np.array([0, 1, 0, 1]))


def test_statevector_pauli_expectation_z_on_zero():
    state = np.array([1.0, 0.0], dtype=complex)
    value = statevector_pauli_expectation(state, ((0, "Z"),), n_qubits=1)
    assert np.isclose(value, 1.0)


def test_statevector_pauli_expectation_z_on_one():
    state = np.array([0.0, 1.0], dtype=complex)
    value = statevector_pauli_expectation(state, ((0, "Z"),), n_qubits=1)
    assert np.isclose(value, -1.0)

from vqd_nuclear.measurements import group_qubit_wise_commuting_terms, group_measurement_basis


def test_qwc_grouping_merges_same_basis_terms():
    terms = [((0, "Z"),), ((0, "Z"), (1, "X")), ((2, "Y"),)]
    groups = group_qubit_wise_commuting_terms(terms)
    assert len(groups) == 1
    assert group_measurement_basis(groups[0]) == [(0, "Z"), (1, "X"), (2, "Y")]


def test_qwc_grouping_splits_incompatible_bases():
    terms = [((0, "X"),), ((0, "Y"),), ((0, "X"), (1, "Z"))]
    groups = group_qubit_wise_commuting_terms(terms)
    assert len(groups) == 2

class _SimpleQubitOperator:
    def __init__(self, terms):
        self.terms = terms


class _OneQubitXAnsatz:
    def cache_key(self, thetas, operator_indices):
        return (tuple(float(t) for t in thetas), tuple(int(i) for i in operator_indices))

    def clear_cache(self):
        pass

    def state(self, thetas, operator_indices):
        # Starts in |0> and applies exp(-i theta X) for every selected operator.
        total_theta = sum(float(theta) for theta, index in zip(thetas, operator_indices) if int(index) == 0)
        return np.array([np.cos(total_theta), -1j * np.sin(total_theta)], dtype=complex)


def test_analytic_vqd_gradient_includes_overlap_penalty():
    from vqd_nuclear.measurements import MeasurementEngine

    hamiltonian = _SimpleQubitOperator({((0, "Z"),): 1.0})
    generator = _SimpleQubitOperator({((0, "X"),): 1.0})
    engine = MeasurementEngine(
        ansatz_builder=_OneQubitXAnsatz(),
        qubit_hamiltonian=hamiltonian,
        n_qubits=1,
        available_workers=1,
        measurement_strategy="qwc",
    )

    alpha = 0.37
    beta = 2.5
    previous = [{"thetas": [alpha], "ops": [0]}]

    energy_only = engine.analytic_energy_gradient([], [], generator, previous, [beta])
    full_vqd = engine.analytic_vqd_gradient([], [], generator, previous, [beta])

    assert np.isclose(energy_only, 0.0)
    assert np.isclose(full_vqd, beta * np.sin(2.0 * alpha))


def test_analytic_vqd_gradient_matches_finite_difference_for_penalty():
    from vqd_nuclear.measurements import MeasurementEngine

    hamiltonian = _SimpleQubitOperator({((0, "Z"),): 1.0})
    generator = _SimpleQubitOperator({((0, "X"),): 1.0})
    engine = MeasurementEngine(
        ansatz_builder=_OneQubitXAnsatz(),
        qubit_hamiltonian=hamiltonian,
        n_qubits=1,
        available_workers=1,
        measurement_strategy="qwc",
    )

    alpha = 0.23
    beta = 1.7
    previous = [{"thetas": [alpha], "ops": [0]}]
    eps = 1e-6

    def objective(theta):
        energy = engine.statevector_energy([theta], [0])
        overlap = engine.exact_overlap([alpha], [0], [theta], [0])
        return energy + beta * overlap

    numeric = (objective(eps) - objective(-eps)) / (2.0 * eps)
    analytic = engine.analytic_vqd_gradient([], [], generator, previous, [beta])

    assert np.isclose(analytic, numeric, rtol=1e-6, atol=1e-8)
