"""Configuration objects for nuclear ADAPT-VQD runs."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import os


@dataclass(slots=True)
class ProblemConfig:
    """Nuclear shell-model problem definition."""

    shell: str = "sd"
    n_valence_protons: int = 2
    n_valence_neutrons: int = 0
    total_j: int = 0


@dataclass(slots=True)
class RunConfig:
    """High-level VQD execution settings."""

    n_states: int = 2
    beta: float | str = "auto"
    run_statevector: bool = True
    run_full_quantum_circuit: bool = False
    n_workers: int | str = "auto"
    seed: int = 12345


@dataclass(slots=True)
class NumericalConfig:
    """Numerical controls for ADAPT-VQD."""

    max_adapt_iter: int = 20
    grad_tol: float = 5e-4
    max_candidates: int | None = None
    optimizer_maxiter: int = 1500
    cobyla_rhobeg: float = 0.03
    cobyla_rhoend: float = 5e-3
    overlap_shots: int = 200000
    energy_shots: int = 200000
    new_theta_tol: float = 1e-3
    energy_improvement_tol: float = 1e-4
    gradient_method: str = "analytic"
    measurement_strategy: str = "paper"
    n_jobs: int | str = "all"
    selection_criterion: str = "full_vqd"
    collect_gradient_diagnostics: bool = False


@dataclass(slots=True)
class FullQuantumLimits:
    """Optional caps applied only to the sampled circuit route."""

    max_adapt_iter: int | None = None
    max_candidates: int | None = None
    optimizer_maxiter: int | None = None
    overlap_shots: int | None = None
    energy_shots: int | None = None
    n_jobs: int | str = "all"


@dataclass(slots=True)
class OutputConfig:
    """Output paths and file-format controls."""

    results_dir: Path = Path("vqd_results")
    run_label: str | None = None
    save_text_outputs: bool = True
    save_machine_outputs: bool = True


def env_int(name: str, default: int) -> int:
    """Read a positive integer from the environment."""
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return int(default)
    try:
        return max(1, int(value))
    except ValueError:
        return int(default)


def detect_hpc_workers() -> int:
    """Detect CPU workers from common HPC schedulers or local CPU count."""
    for name in ("VQD_N_WORKERS", "SLURM_CPUS_PER_TASK", "PBS_NP", "NSLOTS"):
        if os.environ.get(name):
            return env_int(name, os.cpu_count() or 1)
    return max(1, os.cpu_count() or 1)


def dataclass_to_dict(obj: Any) -> dict[str, Any]:
    """Return a JSON-serializable dictionary for a configuration dataclass."""
    data = asdict(obj)
    for key, value in list(data.items()):
        if isinstance(value, Path):
            data[key] = str(value)
    return data


def apply_optional_caps(options: dict[str, Any], caps: dict[str, Any]) -> dict[str, Any]:
    """Apply optional upper bounds to runtime-sensitive options."""
    capped = dict(options)
    for key, cap_value in caps.items():
        if cap_value is None:
            continue
        current = capped.get(key)
        if isinstance(cap_value, int) and isinstance(current, int):
            capped[key] = min(current, cap_value)
        elif key not in capped:
            capped[key] = cap_value
    return capped
