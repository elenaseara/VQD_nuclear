"""Hamiltonian construction and exact reference energies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse.linalg import eigsh

import pynucshell as pns
from pynucshell.circuit.backends.qibo_backend import QiboCircuitBackend
from pynucshell.circuit.encodings import jordan_wigner
from pynucshell.circuit.solvers import ADAPTVQE

from .config import ProblemConfig


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


def infer_nqubits_from_pool(qubit_op_pool: list[Any]) -> int:
    """Infer the JW qubit count from the encoded operator pool."""
    max_qubit = -1
    for operator in qubit_op_pool:
        for term in operator.terms:
            for qubit, _pauli in term:
                max_qubit = max(max_qubit, int(qubit))
    if max_qubit < 0:
        raise RuntimeError("Could not infer the number of qubits from the operator pool.")
    return max_qubit + 1


def extract_qubit_hamiltonian(adaptvqe_helper: ADAPTVQE) -> Any:
    """Retrieve the qubit Hamiltonian generated internally by pynucshell."""
    for attr in ("qubit_hamiltonian", "qubit_ham", "observable", "hamiltonian", "ham_op"):
        if hasattr(adaptvqe_helper, attr):
            candidate = getattr(adaptvqe_helper, attr)
            if hasattr(candidate, "terms"):
                return candidate
    raise RuntimeError("Could not find a qubit Hamiltonian in the ADAPTVQE helper.")


def build_hamiltonian_data(config: ProblemConfig, n_states: int) -> HamiltonianData:
    """Construct all Hamiltonian objects needed by the VQD workflow."""
    nucleus = build_nuclear_hamiltonian(config)
    exact_energies, exact_vectors, matrix = exact_lowest_energies(nucleus, n_states=max(10, n_states))

    backend = QiboCircuitBackend()
    encoding = jordan_wigner
    helper = ADAPTVQE(encoding, backend)
    helper.load_nucleus(nucleus)
    qubit_hamiltonian = extract_qubit_hamiltonian(helper)

    fermionic_pool = list(nucleus.operator_pool())
    qubit_pool = [encoding(operator) for operator in fermionic_pool]
    qubit_generator_pool = [(-1j) * operator for operator in qubit_pool]
    n_qubits = infer_nqubits_from_pool(qubit_pool)

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
    )
