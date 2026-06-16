"""Validation helpers that do not require Qibo or pynucshell imports."""
from __future__ import annotations

from typing import Any
import numpy as np


def beta_from_largest_requested_gap(exact_energies: np.ndarray, n_states: int) -> float:
    """Return a VQD penalty large enough for all requested states.

    The deflation penalty must be larger than the energy separation between
    the ground state and any lower state that could otherwise be selected again.
    For multi-state VQD runs this uses

        beta = 2 * max_i |E_i - E_0|,

    where the maximum is taken over the exact energies of the requested states.
    This avoids the failure mode in which a higher excited-state calculation
    collapses back to the ground state because beta = 2*|E1 - E0| is too small.
    """
    exact_energies = np.asarray(exact_energies, dtype=float)
    n_requested = int(n_states)

    if n_requested < 2:
        return 0.0

    if len(exact_energies) < n_requested:
        raise ValueError(
            f"At least {n_requested} exact energies are required to set beta "
            "from the largest requested excitation gap."
        )

    ground_energy = float(exact_energies[0])
    requested_energies = exact_energies[:n_requested]
    gaps = np.abs(requested_energies - ground_energy)
    max_gap = float(np.max(gaps))

    if not np.isfinite(max_gap):
        raise ValueError("The requested excitation gaps are not finite, so beta cannot be determined.")

    return float(2.0 * max_gap)


def beta_from_first_excitation_gap(exact_energies: np.ndarray, n_states: int) -> float:
    """Return the legacy two-state penalty beta = 2*|E1 - E0|.

    This public helper is kept for backward compatibility and for tests that
    explicitly validate the original two-state rule. The workflow itself uses
    :func:`beta_from_largest_requested_gap` through :func:`resolve_beta`.
    """
    exact_energies = np.asarray(exact_energies, dtype=float)

    if int(n_states) < 2:
        return 0.0

    if len(exact_energies) < 2:
        raise ValueError("At least two exact energies are required to set beta = 2*max_i abs(E_i - E_0).")

    gap = abs(float(exact_energies[1]) - float(exact_energies[0]))

    if not np.isfinite(gap):
        raise ValueError("The exact first excitation gap is not finite, so beta cannot be determined.")

    return float(2.0 * gap)


def resolve_beta(requested_beta: float | None, exact_energies: np.ndarray, n_states: int) -> dict[str, Any]:
    """Resolve the beta used by the workflow.

    For n_states >= 2 the code intentionally ignores the user-provided beta and
    sets beta from the largest exact excitation gap among the requested states.
    The originally requested value is still returned for provenance in output
    reports.
    """
    exact_energies = np.asarray(exact_energies, dtype=float)
    n_requested = int(n_states)

    if n_requested < 2:
        return {
            "beta": 0.0,
            "requested_beta": None if requested_beta is None else float(requested_beta),
            "status": "not_needed",
            "gap_01": None,
            "max_requested_gap": None,
            "formula": "beta is not used for a single-state VQE/VQD run",
            "message": "beta is not used because only one state was requested.",
        }

    beta = beta_from_largest_requested_gap(exact_energies, n_requested)
    gap_01 = abs(float(exact_energies[1]) - float(exact_energies[0]))
    max_requested_gap = float(np.max(np.abs(exact_energies[:n_requested] - float(exact_energies[0]))))
    requested = None if requested_beta is None else float(requested_beta)

    return {
        "beta": beta,
        "requested_beta": requested,
        "status": "auto_from_largest_requested_gap",
        "gap_01": gap_01,
        "max_requested_gap": max_requested_gap,
        "formula": "beta = 2*max_i(abs(E_i - E_0)) for requested states",
        "message": (
            f"beta was set to 2*max_i(|Ei - E0|) = {beta:.12g} "
            f"using the largest requested gap {max_requested_gap:.12g}."
        ),
    }


def validate_beta(beta: float | None, exact_energies: np.ndarray, n_states: int) -> dict[str, Any]:
    """Backward-compatible alias for beta resolution."""
    return resolve_beta(beta, exact_energies, n_states)
