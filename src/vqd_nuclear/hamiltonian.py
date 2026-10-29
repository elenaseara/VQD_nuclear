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
    pool_metadata: dict[str, Any]


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
    matrix = nucleus.to_csr()
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
    # ARPACK's implicit random start is independent of the workflow seed in
    # some SciPy versions. Fix it so repeated reference gaps are reproducible.
    v0 = np.random.default_rng(0).standard_normal(dim)
    values, vectors = eigsh(matrix, k=k, which="SA", v0=v0)
    order = np.argsort(values)
    return np.asarray(values[order], dtype=float), vectors[:, order], matrix


def build_hamiltonian_data(config: ProblemConfig, n_states: int) -> HamiltonianData:
    """Construct all Hamiltonian objects needed by the VQD workflow."""
    nucleus = build_nuclear_hamiltonian(config)
    # Three reference eigenpairs by default; retain enough for larger VQD runs.
    exact_energies, exact_vectors, matrix = exact_lowest_energies(nucleus, n_states=max(3, n_states))

    backend = QiboCircuitBackend()
    encoding = jordan_wigner
    mapping = OrbitalMapping.from_nucleus(nucleus)
    source_hamiltonian = nucleus.get_fermionic(prune=True)
    source_pool = list(nucleus.operator_pool())
    h_mapping = mapping.operator_mapping([source_hamiltonian])
    pool_mapping = mapping.operator_mapping(source_pool)
    qubit_hamiltonian = encoding(mapping.operator(source_hamiltonian, h_mapping))
    fermionic_pool = [mapping.operator(operator, pool_mapping) for operator in source_pool]
    qubit_pool = [encoding(operator) for operator in fermionic_pool]
    qubit_generator_pool = [(-1j) * operator for operator in qubit_pool]
    n_qubits = mapping.n_qubits

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
        orbital_mapping=mapping,
        pool_metadata={
            "construction": "PyNucShell nucleus.operator_pool() over the entire many-body basis",
            "representation": "M_scheme", "requested_M": config.total_j,
            "preserves": ["N_protons", "N_neutrons", "M"], "exact_J_guaranteed": False,
            "pool_size_initial": len(source_pool), "pool_size_after_reference": len(source_pool),
            "reference_filter_applied": False, "structural_exclusions": [],
            "library_restriction": "two-body transitions present in H connecting determinants of the sector",
            "global_to_qubit": mapping.global_to_qubit,
            "hamiltonian_index_mapping": h_mapping, "pool_index_mapping": pool_mapping,
        },
    )
