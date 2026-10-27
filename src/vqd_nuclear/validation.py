"""Exact and approximate per-target VQD penalties."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.sparse import issparse


def exact_betas(exact_energies: np.ndarray, n_states: int) -> dict[str, Any]:
    """Return beta_i = 2*|E_i-E_0|; the ground-state run has no penalty."""
    energies = np.asarray(exact_energies, dtype=float)
    if int(n_states) != n_states or n_states < 1:
        raise ValueError("n_states must be a positive integer.")
    n_states = int(n_states)
    if energies.ndim != 1 or len(energies) < n_states or not np.all(np.isfinite(energies[:n_states])):
        raise ValueError("Finite exact energies are required for every requested state.")
    if np.any(np.diff(energies[:n_states]) < 0):
        raise ValueError("Exact energies must be ordered from lowest to highest.")
    betas = 2.0 * np.abs(energies[:n_states] - energies[0])
    if not np.all(np.isfinite(betas)):
        raise ValueError("Exact beta values must be finite.")
    return {"betas": betas.tolist(), "formula": "beta_i = 2*abs(E_i-E_0); beta_0 = 0 (unused)"}


def approximate_betas(matrix: Any, n_states: int) -> dict[str, Any]:
    """Return beta_i = |e_i-e_0+2B| from the many-body Hamiltonian.

    e_i are sorted diagonal entries, retaining multiplicities. B is the
    largest absolute off-diagonal row sum. Sparse matrices remain sparse.
    """
    if not issparse(matrix):
        matrix = np.asarray(matrix)
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("The Hamiltonian must be square.")
    if int(n_states) != n_states or not 1 <= n_states <= matrix.shape[0]:
        raise ValueError("n_states must be a positive integer no larger than the matrix dimension.")
    n_states = int(n_states)
    if issparse(matrix):
        off_diagonal = matrix.tocsr(copy=True)
        off_diagonal.sum_duplicates()
        diagonal = off_diagonal.diagonal().copy()
        off_diagonal.setdiag(0)
        off_diagonal.eliminate_zeros()
        row_sums = np.asarray(abs(off_diagonal).sum(axis=1)).ravel()
    else:
        off_diagonal = matrix.copy()
        diagonal = np.diag(off_diagonal).copy()
        np.fill_diagonal(off_diagonal, 0)
        row_sums = np.sum(np.abs(off_diagonal), axis=1)
    if not np.all(np.isfinite(diagonal)) or not np.all(np.isfinite(row_sums)):
        raise ValueError("The Hamiltonian entries must be finite.")
    if np.any(np.imag(diagonal) != 0):
        raise ValueError("The diagonal must contain real energies.")
    energies = np.sort(np.real(diagonal).astype(float))
    bound = float(np.max(row_sums))
    betas = np.abs(energies[:n_states] - energies[0] + 2.0 * bound)
    betas[0] = 0.0
    if not np.all(np.isfinite(betas)):
        raise ValueError("Approximate beta values must be finite.")
    return {
        "betas": betas.tolist(), "diagonal_energies": energies[:n_states].tolist(), "B": bound,
        "formula": "beta_i = abs(e_i-e_0+2*B); beta_0 = 0 (unused)",
    }


def resolve_beta_comparison(exact_energies: np.ndarray, matrix: Any, n_states: int) -> dict[str, Any]:
    """Calculate both automatic penalties for all requested target states."""
    exact = exact_betas(exact_energies, n_states)
    approximate = approximate_betas(matrix, n_states)
    return {
        "exact": exact, "approximate": approximate,
        "message": (f"Exact beta by state: {exact['betas']}; "
                    f"approximate beta by state: {approximate['betas']}; B={approximate['B']:.12g}. "
                    "Ground-state beta is unused."),
    }
