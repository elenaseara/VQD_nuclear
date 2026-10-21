"""Minimum-diagonal reference selection, independent of pool connectivity."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import diags

pytest.importorskip("qibo")
pytest.importorskip("pynucshell")

from pynucshell.operators import QubitOperator
from vqd_nuclear.ansatz import find_reference_from_hamiltonian_basis
from vqd_nuclear.config import NumericalConfig, ProblemConfig, RunConfig
from vqd_nuclear.orbitals import OrbitalMapping
from vqd_nuclear import vqd


def select_reference(basis, energies):
    nucleus = SimpleNamespace(
        get_basis=lambda: SimpleNamespace(mb=lambda: basis),
        get_space=lambda: SimpleNamespace(isospin=[1, 1, -1, -1]),
    )
    return find_reference_from_hamiltonian_basis(
        ProblemConfig(shell="p", n_valence_protons=1, n_valence_neutrons=1), nucleus,
        n_qubits=4, orbital_mapping=OrbitalMapping({q: q for q in range(4)}, (1, 1, -1, -1), 1, 1),
        shell_diagonal=np.asarray(energies),
    )


def test_minimum_diagonal_in_supplied_sector():
    reference = select_reference([(0, 2), (1, 3), (0, 3)], [2.0, -3.0, 1.0])
    assert reference.occupied_orbitals == [1, 3]
    assert reference.diagonal_energy == -3.0
    assert reference.basis_index == 1


@pytest.mark.parametrize("basis", [[(1, 3), (0, 2)], {9: (1, 3), 2: (0, 2)}])
def test_ties_follow_basis_order_instead_of_orbital_order(basis):
    reference = select_reference(basis, [-2.0, -2.0])
    assert reference.occupied_orbitals == [1, 3]
    assert reference.basis_index == 0


def test_reference_rejects_determinant_outside_species_sector():
    with pytest.raises(ValueError, match="proton/neutron"):
        select_reference([(0, 1)], [-100.0])


@pytest.mark.parametrize("energies", [[-1.0], [np.nan, 0], [np.inf, 0], [1j, 0]])
def test_invalid_diagonal_is_rejected(energies):
    with pytest.raises(ValueError, match="diagonal"):
        select_reference([(0, 2), (1, 3)], energies)


@pytest.mark.parametrize("pool_policy,reason", [("reference_active", "pool_exhausted"), ("full", "gradient_converged")])
def test_disconnected_minimum_is_kept_and_adapt_stops(monkeypatch, pool_policy, reason):
    # Both determinants are in the one-proton sector. The pool has no action
    # on either: reference selection must still succeed and retain the minimum.
    nucleus = SimpleNamespace(
        get_basis=lambda: SimpleNamespace(mb=lambda: [(0,), (1,)]),
        get_space=lambda: SimpleNamespace(isospin=[1, 1]),
    )
    data = SimpleNamespace(
        nucleus=nucleus, n_qubits=2, matrix=diags([-1.0, -2.0]), exact_energies=np.array([-2.0, -1.0]),
        exact_vectors=None, qubit_generator_pool=[QubitOperator()], backend=object(),
        qubit_hamiltonian=QubitOperator((), -1.5) + QubitOperator(((1, "Z"),), 0.5),
        orbital_mapping=OrbitalMapping({0: 0, 1: 1}, (1, 1), 1, 0),
    )
    monkeypatch.setattr(vqd, "build_hamiltonian_data", lambda *args, **kwargs: data)
    context = vqd.build_context(
        ProblemConfig(shell="p", n_valence_protons=1, n_valence_neutrons=0),
        RunConfig(n_states=1, n_workers=1), NumericalConfig(n_jobs=1, measurement_strategy="qwc"),
    )
    assert context.reference.occupied_orbitals == [1]
    assert context.reference.basis_index == 1
    assert context.active_operator_indices == []
    result = vqd.run_adapt_vqd(context, n_jobs=1, pool_policy=pool_policy)
    assert result["stop_reason"] == reason
    assert result["n_iterations"] == result["n_layers"] == 0
    assert result["energy"] == pytest.approx(-2.0)
    assert context.reference.occupied_orbitals == [1]
