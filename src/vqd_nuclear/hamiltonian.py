"""Hamiltonian construction and exact reference energies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse.linalg import eigsh

import pynucshell as pns
from pynucshell.circuit.backends.qibo_backend import QiboCircuitBackend
from pynucshell.circuit.encodings import jordan_wigner

from .config import ProblemConfig
from .orbitals import OrbitalMapping


@dataclass(slots=True)
class HamiltonianData:
    """Container for the nuclear Hamiltonian and its qubit representation."""

    nucleus: Any
    matrix: Any
    exact_energies: np.ndarray
    exact_vectors: np.ndarray | None
    qubit_hamiltonian: Any
    fermionic_pool: list[Any]
    qubit_pool: list[Any]
    qubit_generator_pool: list[Any]
    n_qubits: int
    backend: QiboCircuitBackend
    encoding: Any
    orbital_mapping: OrbitalMapping


def build_nuclear_hamiltonian(config: ProblemConfig) -> Any:
    """Build a pynucshell p- or sd-shell Hamiltonian."""
    if config.shell == "p":
        return pns.p_Hamiltonian(
            N_protons=int(config.n_valence_protons),
            N_neutrons=int(config.n_valence_neutrons),
            total_J=int(config.total_j),
        )
    if config.shell == "sd":
        return pns.sd_Hamiltonian(
            N_protons=int(config.n_valence_protons),
            N_neutrons=int(config.n_valence_neutrons),
            total_J=int(config.total_j),
        )
    raise ValueError("ProblemConfig.shell must be 'p' or 'sd'.")


def exact_lowest_energies(nucleus: Any, n_states: int) -> tuple[np.ndarray, np.ndarray | None, Any]:
    """Compute the lowest shell-model reference energies.

    SciPy's sparse ``eigsh`` requires ``k < dim``. For very small model
    spaces, or when the requested number of eigenpairs approaches the full
    Hilbert-space dimension, this function falls back to dense diagonalization
    so that validation and beta selection still receive meaningful exact
    energies.
    """
    matrix = pns.Simulator(nucleus._ham, nucleus._basis).to_csr()
    dim = int(matrix.shape[0])
    requested = max(1, int(n_states))

    if dim == 0:
        raise RuntimeError("The shell-model Hamiltonian has zero dimension for this problem.")

    if dim <= 3 or requested >= dim:
        dense_matrix = matrix.toarray()
        values, vectors = np.linalg.eigh(dense_matrix)
        order = np.argsort(values)[: min(requested, dim)]
        return np.asarray(values[order], dtype=float), vectors[:, order], matrix

    k = min(requested, dim - 1)
    values, vectors = eigsh(matrix, k=k, which="SA")
    order = np.argsort(values)
    return np.asarray(values[order], dtype=float), vectors[:, order], matrix


def build_hamiltonian_data(config: ProblemConfig, n_states: int) -> HamiltonianData:
    """Construct all Hamiltonian objects needed by the VQD workflow."""
    nucleus = build_nuclear_hamiltonian(config)
    exact_energies, exact_vectors, matrix = exact_lowest_energies(nucleus, n_states=max(10, n_states))

    backend = QiboCircuitBackend()
    encoding = jordan_wigner
    orbital_mapping = OrbitalMapping.from_nucleus(nucleus)
    qubit_hamiltonian = encoding(orbital_mapping.fermionic_operator(nucleus.get_fermionic(prune=True)))
    fermionic_pool = [orbital_mapping.fermionic_operator(op) for op in nucleus.operator_pool()]
    qubit_pool = [encoding(operator) for operator in fermionic_pool]
    qubit_generator_pool = [(-1j) * operator for operator in qubit_pool]
    n_qubits = orbital_mapping.n_qubits
    for operator in [qubit_hamiltonian, *qubit_pool]:
        if any(not 0 <= int(q) < n_qubits for term in operator.terms for q, _ in term):
            raise ValueError("Encoded operator lies outside the mapped qubit register.")

    return HamiltonianData(
        nucleus=nucleus,
        matrix=matrix,
        exact_energies=exact_energies,
        exact_vectors=exact_vectors,
        qubit_hamiltonian=qubit_hamiltonian,
        fermionic_pool=fermionic_pool,
        qubit_pool=qubit_pool,
        qubit_generator_pool=qubit_generator_pool,
        n_qubits=n_qubits,
        backend=backend,
        encoding=encoding,
        orbital_mapping=orbital_mapping,
    )
