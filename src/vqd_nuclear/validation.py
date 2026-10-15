"""Validation helpers that do not require Qibo or pynucshell imports."""
from __future__ import annotations

from typing import Any
import numpy as np
from scipy.sparse import issparse


def approximate_betas(matrix: Any, n_states: int) -> dict[str, Any]:
    """Compute beta_i = |e_i - e_0 + 2B| in the many-body basis.

    Diagonal entries are sorted with multiplicities. B is the largest absolute
    off-diagonal row sum; sparse Hamiltonians never need to be densified.
    The ground-state run has no deflation penalty and uses beta_0 = 0.
    """
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("The Hamiltonian matrix must be square.")
    if not 1 <= int(n_states) <= matrix.shape[0]:
        raise ValueError("n_states must be between 1 and the Hamiltonian dimension.")
    if issparse(matrix):
        off_diagonal = matrix.tocsr(copy=True)
        off_diagonal.sum_duplicates()
        diagonal = np.asarray(off_diagonal.diagonal())
        off_diagonal.setdiag(0)
        off_diagonal.eliminate_zeros()
        row_sums = np.asarray(abs(off_diagonal).sum(axis=1)).ravel()
    else:
        off_diagonal = np.array(matrix, copy=True)
        diagonal = np.diag(off_diagonal).copy()
        np.fill_diagonal(off_diagonal, 0)
        row_sums = np.sum(np.abs(off_diagonal), axis=1)
    if not np.all(np.isfinite(diagonal)) or not np.all(np.isfinite(row_sums)):
        raise ValueError("Hamiltonian entries must be finite to determine beta.")
    if np.any(np.imag(diagonal) != 0):
        raise ValueError("The Hamiltonian diagonal must contain real energies.")
    energies = np.sort(np.real(diagonal).astype(float))
    bound = float(np.max(row_sums))
    betas = np.abs(energies[:int(n_states)] - energies[0] + 2.0 * bound)
    betas[0] = 0.0
    if not np.all(np.isfinite(betas)):
        raise ValueError("The approximate penalties are not finite.")
    return {
        "betas": betas.tolist(),
        "diagonal_energies": energies[:int(n_states)].tolist(),
        "B": bound,
        "formula": "beta_i = abs(e_i - e_0 + 2*B); beta_0 = 0 (unused)",
    }


def exact_betas(exact_energies: np.ndarray, n_states: int) -> dict[str, Any]:
    """Compute beta_i = 2*|E_i-E_0| from exact diagonalization energies."""
    if int(n_states) < 1:
        raise ValueError("At least one state must be requested.")
    exact_energies = np.asarray(exact_energies, dtype=float)
    if len(exact_energies) < int(n_states) or not np.all(np.isfinite(exact_energies[:int(n_states)])):
        raise ValueError("Finite exact energies are required for all requested states.")
    betas = 2.0 * np.abs(exact_energies[:int(n_states)] - exact_energies[0])
    if not np.all(np.isfinite(betas)):
        raise ValueError("The exact penalties are not finite.")
    return {"betas": betas.tolist(), "formula": "beta_i = 2*abs(E_i - E_0)"}


def resolve_beta_comparison(exact_energies: np.ndarray, matrix: Any, n_states: int) -> dict[str, Any]:
    """Calculate the exact and approximate per-target penalties."""
    exact = exact_betas(exact_energies, n_states)
    approximate = approximate_betas(matrix, n_states)
    return {
        "status": "automatic_comparison",
        "exact": exact,
        "approximate": approximate,
        "message": (
            f"Exact beta by state: {exact['betas']}; "
            f"approximate beta by state: {approximate['betas']} (B={approximate['B']:.12g}). "
            "Ground-state beta is unused."
        ),
    }
