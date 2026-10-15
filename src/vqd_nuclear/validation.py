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


def resolve_beta(requested_beta, exact_energies, n_states) -> dict[str, Any]:
    """Resolve the beta used by the workflow.

    ``requested_beta`` may be ``None`` or ``"auto"`` to derive beta from the
    exact spectrum, or an explicit non-negative number, which is respected.
    An explicit value below the largest requested gap is flagged as unsafe but
    is NOT overridden: the caller asked for it deliberately.
    """
    exact_energies = np.asarray(exact_energies, dtype=float)
    n_requested = int(n_states)

    is_auto = requested_beta is None or (
        isinstance(requested_beta, str) and requested_beta.strip().lower() == "auto"
    )
    requested = None if is_auto else float(requested_beta)

    if requested is not None and (requested < 0.0 or not np.isfinite(requested)):
        raise ValueError(f"beta must be a finite non-negative number, got {requested_beta!r}.")

    if n_requested < 2:
        return {
            "beta": 0.0,
            "requested_beta": "auto" if is_auto else requested,
            "status": "not_needed",
            "gap_01": None,
            "max_requested_gap": None,
            "auto_beta": None,
            "warning": None,
            "formula": "beta is not used for a single-state VQE/VQD run",
            "message": "beta is not used because only one state was requested.",
        }

    auto_beta = beta_from_largest_requested_gap(exact_energies, n_requested)
    gap_01 = abs(float(exact_energies[1]) - float(exact_energies[0]))
    max_requested_gap = float(
        np.max(np.abs(exact_energies[:n_requested] - float(exact_energies[0])))
    )

    if is_auto:
        return {
            "beta": auto_beta,
            "requested_beta": "auto",
            "status": "auto_from_largest_requested_gap",
            "gap_01": gap_01,
            "max_requested_gap": max_requested_gap,
            "auto_beta": auto_beta,
            "warning": None,
            "formula": "beta = 2*max_i(abs(E_i - E_0)) for requested states",
            "message": (
                f"beta was set automatically to 2*max_i(|Ei - E0|) = {auto_beta:.12g} "
                f"using the largest requested gap {max_requested_gap:.12g}."
            ),
        }

    warning = None
    if requested < max_requested_gap:
        warning = (
            f"beta = {requested:.12g} is below the largest requested gap "
            f"{max_requested_gap:.12g}. VQD requires beta_i > E_k - E_i, so an excited "
            "state may collapse onto a lower one. Check the final overlaps."
        )

    return {
        "beta": requested,
        "requested_beta": requested,
        "status": "user_supplied",
        "gap_01": gap_01,
        "max_requested_gap": max_requested_gap,
        "auto_beta": auto_beta,
        "warning": warning,
        "formula": "beta supplied explicitly by the user",
        "message": (
            f"beta = {requested:.12g} was supplied explicitly "
            f"(the automatic value would have been {auto_beta:.12g})."
            + (f" WARNING: {warning}" if warning else "")
        ),
    }


def validate_beta(beta: float | None, exact_energies: np.ndarray, n_states: int) -> dict[str, Any]:
    """Backward-compatible alias for beta resolution."""
    return resolve_beta(beta, exact_energies, n_states)
