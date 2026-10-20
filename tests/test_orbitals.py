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


@pytest.mark.parametrize("width", [6, 12])
@pytest.mark.parametrize("hamiltonian_local,pool_local", [(False, False), (True, True), (False, True), (True, False)])
def test_independent_operator_conventions_without_requiring_a_library_version(monkeypatch, width, hamiltonian_local, pool_local):
    from scipy.sparse import csr_matrix
    from pynucshell.circuit.encodings import jordan_wigner
    from vqd_nuclear import hamiltonian

    def hopping(offset):
        operator = FermionicOperator(((offset, 1), (offset + 1, 0)), 2j)
        operator += FermionicOperator(((offset + 1, 1), (offset, 0)), -2j)
        operator += FermionicOperator((), 1.5)
        return operator

    def excitation(offset):
        return (FermionicOperator(((offset, 1), (offset + 1, 0)))
                - FermionicOperator(((offset + 1, 1), (offset, 0))))

    nucleus = SimpleNamespace(
        N_protons=0, N_neutrons=1,
        get_space=lambda: SimpleNamespace(mapping=list(range(2 * width)), isospin=[1] * width + [-1] * width),
        get_fermionic=lambda prune: hopping(0 if hamiltonian_local else width),
        operator_pool=lambda: [excitation(0 if pool_local else width)],
    )
    monkeypatch.setattr(hamiltonian, "build_nuclear_hamiltonian", lambda config: nucleus)
    monkeypatch.setattr(hamiltonian, "exact_lowest_energies", lambda *args, **kwargs:
                        (np.array([1.5]), np.eye(1), csr_matrix([[1.5]])))
    data = hamiltonian.build_hamiltonian_data(ProblemConfig(), n_states=1)
    assert data.n_qubits == width
    assert data.orbital_mapping.determinant((width,)) == (0,)
    assert data.qubit_hamiltonian == jordan_wigner(hopping(0))
    assert data.fermionic_pool[0] == excitation(0)
    assert data.qubit_pool[0] == jordan_wigner(excitation(0))
    assert data.hamiltonian_index_mapping == {q if hamiltonian_local else q + width: q for q in range(width)}
    assert data.pool_index_mapping == {q if pool_local else q + width: q for q in range(width)}


def test_operator_convention_checks_the_whole_pool_and_rejects_ambiguity():
    # Nonidentity overlapping label spaces: the largest used index is not a safe rule.
    mapping = OrbitalMapping({1: 0, 3: 1}, (-1, -1), 0, 1)
    ambiguous = FermionicOperator(((1, 1), (1, 0)))
    with pytest.raises(ValueError, match="Ambiguous"):
        mapping.operator_to_qubit([ambiguous])
    global_only = FermionicOperator(((3, 1), (3, 0)))
    source_map = mapping.operator_to_qubit([ambiguous, global_only])
    assert source_map == {1: 0, 3: 1}
    assert mapping.fermionic_operator(ambiguous, index_mapping=source_map).terms == {((0, 1), (0, 0)): 1.0}
    local_only = FermionicOperator(((0, 1), (0, 0)))
    with pytest.raises(ValueError, match="neither"):
        mapping.operator_to_qubit([local_only, global_only])


def test_local_sparse_operator_retains_unused_modes_and_constant_terms():
    mapping = OrbitalMapping({q + 12: q for q in range(12)}, (-1,) * 12, 0, 1)
    operator = FermionicOperator(((5, 1), (5, 0)), 2.0)
    operator += FermionicOperator((), 1.5)
    source_map = mapping.operator_to_qubit([operator])
    assert source_map == {q: q for q in range(12)}
    assert mapping.fermionic_operator(operator, index_mapping=source_map) == operator
    assert mapping.n_qubits == 12


@pytest.mark.parametrize("protons,neutrons", [(0, 2), (2, 0)], ids=["He6", "Be6"])
def test_he6_workflow_with_both_beta_methods(protons, neutrons):
    from vqd_nuclear import FullQuantumLimits, NumericalConfig, RunConfig, run_workflow

    output = run_workflow(
        ProblemConfig(shell="p", n_valence_protons=protons, n_valence_neutrons=neutrons, total_j=0),
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
