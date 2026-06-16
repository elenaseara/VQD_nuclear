"""Nuclear ADAPT-VQD simulations with pynucshell and Qibo."""

from .config import FullQuantumLimits, NumericalConfig, OutputConfig, ProblemConfig, RunConfig
from .validation import (
    beta_from_first_excitation_gap,
    beta_from_largest_requested_gap,
    resolve_beta,
    validate_beta,
)


def run_workflow(*args, **kwargs):
    """Lazy wrapper around :func:`vqd_nuclear.vqd.run_workflow`."""
    from .vqd import run_workflow as _run_workflow

    return _run_workflow(*args, **kwargs)


__all__ = [
    "ProblemConfig",
    "RunConfig",
    "NumericalConfig",
    "FullQuantumLimits",
    "OutputConfig",
    "run_workflow",
    "beta_from_first_excitation_gap",
    "beta_from_largest_requested_gap",
    "resolve_beta",
    "validate_beta",
]
