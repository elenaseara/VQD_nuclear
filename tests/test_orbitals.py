"""Physical consistency of the shared many-body/JW orbital mapping."""
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pynucshell")
pytest.importorskip("qibo")

from pynucshell.operators import FermionicOperator
from vqd_nuclear.ansatz import get_hamiltonian_basis_references, diagonal_energy
from vqd_nuclear.config import ProblemConfig
from vqd_nuclear.hamiltonian import build_hamiltonian_data
from vqd_nuclear.orbitals import OrbitalMapping


def apply_to_determinant(operator, occupied):
    """Apply Pauli terms sparsely, without allocating a 2**n statevector."""
    initial = sum(1 << q for q in occupied)
    amplitudes = {}
    for term, coefficient in operator.terms.items():
        final, amplitude = initial, complex(coefficient)
        for q, pauli in term:
            bit = (initial >> int(q)) & 1
            if pauli in ("Y", "Z"):
                amplitude *= (-1) ** bit
            if pauli == "Y":
                amplitude *= 1j
            if pauli in ("X", "Y"):
                final ^= 1 << int(q)
        amplitudes[final] = amplitudes.get(final, 0j) + amplitude
    return {state: value for state, value in amplitudes.items() if abs(value) > 1e-9}


@pytest.mark.parametrize("shell,width,protons,neutrons", [
    ("p", 6, 2, 0), ("p", 6, 0, 2), ("p", 6, 1, 1), ("p", 6, 2, 2),
    ("sd", 12, 2, 0), ("sd", 12, 0, 2), ("sd", 12, 1, 1),
])
def test_mapping_energy_and_physical_sector(shell, width, protons, neutrons):
    data = build_hamiltonian_data(ProblemConfig(shell=shell, n_valence_protons=protons,
                                              n_valence_neutrons=neutrons, total_j=0), n_states=2)
    mapping = data.orbital_mapping
    assert data.n_qubits == width * (int(protons > 0) + int(neutrons > 0))
    references = get_hamiltonian_basis_references(
        data.nucleus, n_qubits=data.n_qubits, n_valence_particles=protons + neutrons,
        orbital_mapping=mapping,
    )
    inverse = {qubit: label for label, qubit in mapping.global_to_qubit.items()}
    basis = data.nucleus.get_basis().mb()
    physical_states = {sum(1 << q for q in occupied) for occupied in references}
    ordered_states = [sum(1 << q for q in occupied) for occupied in references]
    for index, occupied in enumerate(references):
        assert sorted(inverse[q] for q in occupied) == sorted(int(q) for q in basis[index])
        assert sum(mapping.qubit_species[q] == 1 for q in occupied) == protons
        assert sum(mapping.qubit_species[q] == -1 for q in occupied) == neutrons
        assert diagonal_energy(occupied, data.qubit_hamiltonian) == pytest.approx(data.matrix[index, index], abs=1e-9)
        action = apply_to_determinant(data.qubit_hamiltonian, occupied)
        np.testing.assert_allclose(
            [action.get(state, 0j) for state in ordered_states],
            data.matrix[:, index].toarray().ravel(), atol=1e-9,
        )
        for operator in [data.qubit_hamiltonian, *data.qubit_generator_pool]:
            assert set(apply_to_determinant(operator, occupied)).issubset(physical_states)


def test_complete_space_includes_modes_absent_from_pool_and_basis():
    # Only two of the six neutron orbitals occur in this artificial basis.
    nucleus = SimpleNamespace(
        N_protons=0, N_neutrons=2,
        get_space=lambda: SimpleNamespace(mapping=list(range(12)), isospin=[1] * 6 + [-1] * 6),
    )
    mapping = OrbitalMapping.from_nucleus(nucleus)
    assert mapping.n_qubits == 6
    assert mapping.determinant((6, 9)) == (0, 3)
    assert mapping.global_to_qubit[11] == 5


def test_mapping_rejects_wrong_species_duplicates_and_inactive_modes():
    mapping = OrbitalMapping({0: 0, 1: 1, 6: 2, 7: 3}, (1, 1, -1, -1), 1, 1)
    for occupied in [(0, 1), (6, 7), (0, 0), (0, 8)]:
        with pytest.raises(ValueError):
            mapping.determinant(occupied)
    with pytest.raises(ValueError, match="changes the proton or neutron"):
        mapping.fermionic_operator(FermionicOperator(((0, 1), (6, 0))))
    with pytest.raises(ValueError, match="inactive orbital"):
        mapping.fermionic_operator(FermionicOperator(((8, 1), (8, 0))))


def test_fermionic_relabeling_preserves_coefficients_actions_and_constants():
    mapping = OrbitalMapping({6: 0, 7: 1}, (-1, -1), 0, 1)
    operator = FermionicOperator(((6, 1), (7, 0)), 2j)
    operator += FermionicOperator((), 1.5)
    assert mapping.fermionic_operator(operator).terms == {((0, 1), (1, 0)): 2j, (): 1.5}


def test_he6_workflow_with_both_beta_methods():
    from vqd_nuclear import FullQuantumLimits, NumericalConfig, RunConfig, run_workflow

    output = run_workflow(
        ProblemConfig(shell="p", n_valence_protons=0, n_valence_neutrons=2, total_j=0),
        RunConfig(n_states=2, run_full_quantum_circuit=False, n_workers=1, seed=47433),
        NumericalConfig(max_adapt_iter=6, optimizer_maxiter=1000, n_jobs=1,
                        grad_tol=1e-5, new_theta_tol=1e-5, energy_improvement_tol=1e-6),
        FullQuantumLimits(),
    )
    assert output["context"].hamiltonian.n_qubits == 6
    results = output["statevector_results"]
    assert [r["beta_method"] for r in results] == ["exact", "exact", "approximate", "approximate"]
    for result in results:
        assert result["abs_energy_error"] < 1e-3
