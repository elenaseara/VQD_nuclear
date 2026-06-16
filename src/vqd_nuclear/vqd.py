"""ADAPT-VQD driver for ground and first-excited nuclear states."""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any
import logging

import numpy as np
from scipy.optimize import minimize

import qibo

from .ansatz import AnsatzBuilder, find_reference_from_hamiltonian_basis, prepare_generator_term_pool
from .config import FullQuantumLimits, NumericalConfig, ProblemConfig, RunConfig, apply_optional_caps, dataclass_to_dict, detect_hpc_workers
from .hamiltonian import HamiltonianData, build_hamiltonian_data
from .measurements import MeasurementEngine
from .parallel import parallel_map, resolve_n_jobs
from .validation import resolve_beta

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class VQDContext:
    """All mutable and immutable objects required by an ADAPT-VQD run."""

    problem: ProblemConfig
    run: RunConfig
    numerical: NumericalConfig
    hamiltonian: HamiltonianData
    reference: Any
    ansatz_builder: AnsatzBuilder
    measurements: MeasurementEngine
    active_operator_indices: list[int]
    available_workers: int


def set_reproducible_seed(seed: int) -> None:
    """Set the random seed for NumPy and Qibo when available."""
    np.random.seed(int(seed))
    if hasattr(qibo, "set_seed"):
        qibo.set_seed(int(seed))
    elif hasattr(qibo, "config") and hasattr(qibo.config, "set_seed"):
        qibo.config.set_seed(int(seed))


def configure_qibo_backend(backend_name: str = "numpy", seed: int | None = None) -> None:
    """Configure Qibo with a backend and optional seed."""
    qibo.set_backend(backend_name)
    if seed is not None:
        set_reproducible_seed(seed)


def build_context(problem: ProblemConfig, run: RunConfig, numerical: NumericalConfig, *, qibo_backend: str = "numpy") -> VQDContext:
    """Build Hamiltonian, reference state, ansatz builder, and active pool."""
    configure_qibo_backend(qibo_backend, seed=run.seed)
    available_workers = detect_hpc_workers() if str(run.n_workers).lower() == "auto" else int(run.n_workers)
    hamiltonian = build_hamiltonian_data(problem, n_states=run.n_states)
    reference = find_reference_from_hamiltonian_basis(
        problem,
        hamiltonian.nucleus,
        n_qubits=hamiltonian.n_qubits,
        qubit_hamiltonian=hamiltonian.qubit_hamiltonian,
        qubit_generator_pool=hamiltonian.qubit_generator_pool,
        backend=hamiltonian.backend,
    )
    prepared_pool = prepare_generator_term_pool(hamiltonian.qubit_generator_pool)
    ansatz_builder = AnsatzBuilder(
        n_qubits=hamiltonian.n_qubits,
        backend=hamiltonian.backend,
        prepared_generator_term_pool=prepared_pool,
        reference_occupied_orbitals=reference.occupied_orbitals,
    )
    measurements = MeasurementEngine(
        ansatz_builder=ansatz_builder,
        qubit_hamiltonian=hamiltonian.qubit_hamiltonian,
        n_qubits=hamiltonian.n_qubits,
        available_workers=available_workers,
        measurement_strategy=numerical.measurement_strategy,
    )
    active_operator_indices = find_active_operators(
        ansatz_builder,
        operator_count=len(hamiltonian.qubit_generator_pool),
        n_jobs=numerical.n_jobs,
        available_workers=available_workers,
    )
    return VQDContext(
        problem=problem,
        run=run,
        numerical=numerical,
        hamiltonian=hamiltonian,
        reference=reference,
        ansatz_builder=ansatz_builder,
        measurements=measurements,
        active_operator_indices=active_operator_indices,
        available_workers=available_workers,
    )


def find_active_operators(ansatz_builder: AnsatzBuilder, *, operator_count: int, n_jobs: int | str | None, available_workers: int) -> list[int]:
    """Return operator indices that change the selected reference state."""
    reference_state = ansatz_builder.state([], [])

    def probe(operator_index: int) -> int | None:
        test_state = ansatz_builder.state([0.1], [int(operator_index)])
        diff = np.linalg.norm(test_state - reference_state)
        return int(operator_index) if diff > 1e-8 else None

    # Threads avoid multiprocessing issues with Qibo objects while still parallelizing light probes.
    values = parallel_map(probe, range(operator_count), n_jobs=n_jobs, available_workers=available_workers)
    return [int(value) for value in values if value is not None]


def get_candidate_operators(context: VQDContext, max_candidates: int | None = None) -> list[int]:
    """Return active candidate operators, optionally truncated for quick tests."""
    candidates = list(context.active_operator_indices)
    if max_candidates is not None:
        candidates = candidates[: min(int(max_candidates), len(candidates))]
    if not candidates:
        raise RuntimeError("No active ADAPT operators available for the selected reference.")
    return candidates


def vqd_cost(
    context: VQDContext,
    thetas: list[float],
    operator_indices: list[int],
    previous_ansatze: list[dict[str, Any]] | None,
    betas: list[float] | None,
    *,
    overlap_shots: int,
    energy_shots: int,
    n_jobs: int | str | None,
    cost_mode: str,
) -> float:
    """Dispatch the VQD objective to exact-statevector or sampled-circuit mode."""
    previous_ansatze = previous_ansatze or []
    betas = betas or []

    if cost_mode == "statevector_exact":
        energy = context.measurements.statevector_energy(thetas, operator_indices)
        overlap_fn = lambda previous: context.measurements.exact_overlap(previous["thetas"], previous["ops"], thetas, operator_indices)
    elif cost_mode == "full_quantum_circuit":
        energy = context.measurements.sampled_energy(thetas, operator_indices, nshots=energy_shots, n_jobs=n_jobs)
        overlap_fn = lambda previous: context.measurements.destructive_swap_overlap(previous["thetas"], previous["ops"], thetas, operator_indices, nshots=overlap_shots)
    else:
        raise ValueError("cost_mode must be 'statevector_exact' or 'full_quantum_circuit'.")

    overlaps = parallel_map(overlap_fn, previous_ansatze, n_jobs=n_jobs, available_workers=context.available_workers) if previous_ansatze else []
    penalty = float(np.sum([float(beta) * float(overlap) for beta, overlap in zip(betas, overlaps)]))
    return float(energy + penalty)


def finite_difference_gradient(task: tuple[VQDContext, int, list[float], list[int], list[dict[str, Any]], list[float], float, int, int, str]) -> tuple[int, float]:
    """Evaluate one finite-difference gradient candidate as a compatibility fallback."""
    context, operator_index, thetas, operator_indices, previous_ansatze, betas, eps, overlap_shots, energy_shots, cost_mode = task
    trial_indices = operator_indices + [int(operator_index)]
    cost_plus = vqd_cost(context, thetas + [eps], trial_indices, previous_ansatze, betas, overlap_shots=overlap_shots, energy_shots=energy_shots, cost_mode=cost_mode, n_jobs=1)
    cost_minus = vqd_cost(context, thetas + [-eps], trial_indices, previous_ansatze, betas, overlap_shots=overlap_shots, energy_shots=energy_shots, cost_mode=cost_mode, n_jobs=1)
    return int(operator_index), float((cost_plus - cost_minus) / (2 * eps))


def analytic_gradient(task):
    """Evaluate one ADAPT candidate gradient from the full VQD objective."""
    context, operator_index, thetas, operator_indices, previous_ansatze, betas = task
    generator = context.hamiltonian.qubit_generator_pool[int(operator_index)]
    gradient = context.measurements.analytic_vqd_gradient(
        thetas,
        operator_indices,
        generator,
        previous_ansatze,
        betas,
    )
    return int(operator_index), float(gradient)


def select_next_operator(context: VQDContext, thetas: list[float], operator_indices: list[int], previous_ansatze: list[dict[str, Any]], betas: list[float], *, eps: float, max_candidates: int | None, n_jobs: int | str | None, overlap_shots: int, energy_shots: int, cost_mode: str, gradient_method: str = "analytic") -> tuple[int | None, float]:
    """Select the unused candidate with the largest absolute ADAPT gradient.

    The default uses the analytic gradient of the full VQD objective. For
    excited states, this includes the derivative of the overlap-penalty terms.
    The legacy finite-difference path remains available for comparisons with
    ``gradient_method=finite_difference``.
    """
    used = set(operator_indices)
    candidates = [idx for idx in get_candidate_operators(context, max_candidates) if idx not in used]
    if not candidates:
        return None, 0.0
    if gradient_method == "analytic":
        tasks = [(context, int(candidate), list(thetas), list(operator_indices), list(previous_ansatze), list(betas)) for candidate in candidates]
        gradients = parallel_map(analytic_gradient, tasks, n_jobs=n_jobs, available_workers=context.available_workers)
    elif gradient_method == "finite_difference":
        tasks = [(context, int(candidate), list(thetas), list(operator_indices), list(previous_ansatze), list(betas), eps, int(overlap_shots), int(energy_shots), cost_mode) for candidate in candidates]
        gradients = parallel_map(finite_difference_gradient, tasks, n_jobs=n_jobs, available_workers=context.available_workers)
    else:
        raise ValueError("gradient_method must be 'analytic' or 'finite_difference'.")
    return max(gradients, key=lambda item: abs(item[1]))


def optimize_ansatz(context: VQDContext, thetas: list[float], operator_indices: list[int], previous_ansatze: list[dict[str, Any]], betas: list[float], *, overlap_shots: int, energy_shots: int, optimizer_maxiter: int, n_jobs: int | str | None, cost_mode: str):
    """Optimize the current ADAPT-VQD ansatz using COBYLA."""
    def objective(x: np.ndarray) -> float:
        return vqd_cost(
            context,
            list(x),
            operator_indices,
            previous_ansatze,
            betas,
            overlap_shots=overlap_shots,
            energy_shots=energy_shots,
            n_jobs=n_jobs,
            cost_mode=cost_mode,
        )

    result = minimize(
        objective,
        np.asarray(thetas, dtype=float),
        method="COBYLA",
        options={"maxiter": int(optimizer_maxiter), "rhobeg": 0.03, "tol": 5e-3},
    )
    return list(result.x), float(result.fun), result


def compute_previous_overlaps(context: VQDContext, thetas: list[float], operator_indices: list[int], previous_ansatze: list[dict[str, Any]], *, n_jobs: int | str | None, cost_mode: str, overlap_shots: int) -> list[float]:
    """Compute overlaps with previous states using the selected overlap method."""
    if cost_mode == "statevector_exact":
        one_overlap = lambda previous: context.measurements.exact_overlap(previous["thetas"], previous["ops"], thetas, operator_indices)
    else:
        one_overlap = lambda previous: context.measurements.destructive_swap_overlap(previous["thetas"], previous["ops"], thetas, operator_indices, nshots=overlap_shots)
    return parallel_map(one_overlap, previous_ansatze, n_jobs=n_jobs, available_workers=context.available_workers)


def evaluate_energy(context: VQDContext, thetas: list[float], operator_indices: list[int], *, cost_mode: str, energy_shots: int, n_jobs: int | str | None) -> float:
    """Evaluate the physical energy in the requested mode."""
    if cost_mode == "full_quantum_circuit":
        return context.measurements.sampled_energy(thetas, operator_indices, nshots=energy_shots, n_jobs=n_jobs)
    return context.measurements.statevector_energy(thetas, operator_indices)


def run_adapt_vqd(context: VQDContext, *, previous_ansatze: list[dict[str, Any]] | None = None, betas: list[float] | None = None, max_adapt_iter: int = 20, grad_tol: float = 2e-4, max_candidates: int | None = None, n_jobs: int | str | None = None, overlap_shots: int = 200000, energy_shots: int = 200000, optimizer_maxiter: int = 200, new_theta_tol: float = 1e-3, energy_improvement_tol: float = 1e-4, cost_mode: str = "statevector_exact", gradient_method: str = "analytic", store_history: bool = True) -> dict[str, Any]:
    """Run ADAPT-VQD for one state."""
    previous_ansatze = previous_ansatze or []
    betas = betas or []
    workers = resolve_n_jobs(n_jobs, context.available_workers)
    thetas: list[float] = []
    operator_indices: list[int] = []
    history: list[dict[str, Any]] = []
    max_layers = min(int(max_adapt_iter), len(get_candidate_operators(context, max_candidates)))
    previous_objective: float | None = None
    last_improvement: float | None = None

    for iteration in range(max_layers):
        operator_index, gradient = select_next_operator(
            context,
            thetas,
            operator_indices,
            previous_ansatze,
            betas,
            eps=1e-2,
            max_candidates=max_candidates,
            n_jobs=workers,
            overlap_shots=overlap_shots,
            energy_shots=energy_shots,
            cost_mode=cost_mode,
            gradient_method=gradient_method,
        )
        if operator_index is None:
            break
        operator_indices.append(int(operator_index))
        thetas.append(0.0)
        thetas, objective_value, result = optimize_ansatz(
            context,
            thetas,
            operator_indices,
            previous_ansatze,
            betas,
            overlap_shots=overlap_shots,
            energy_shots=energy_shots,
            optimizer_maxiter=optimizer_maxiter,
            n_jobs=workers,
            cost_mode=cost_mode,
        )
        energy = evaluate_energy(context, thetas, operator_indices, cost_mode=cost_mode, energy_shots=energy_shots, n_jobs=workers)
        overlaps = compute_previous_overlaps(context, thetas, operator_indices, previous_ansatze, n_jobs=workers, cost_mode=cost_mode, overlap_shots=overlap_shots)
        state_index = len(previous_ansatze)
        exact_target_energy = float(context.hamiltonian.exact_energies[state_index]) if len(context.hamiltonian.exact_energies) > state_index else np.nan
        energy_error = float(energy - exact_target_energy) if np.isfinite(exact_target_energy) else np.nan
        record = {
            "iteration": int(iteration),
            "selected_operator": int(operator_index),
            "adapt_gradient": float(gradient),
            "physical_energy": float(energy),
            "vqd_objective": float(objective_value),
            "exact_target_energy": exact_target_energy,
            "energy_error": energy_error,
            "abs_energy_error": abs(energy_error) if np.isfinite(energy_error) else np.nan,
            "n_layers": int(len(operator_indices)),
            "n_parameters": int(len(thetas)),
            "optimizer_success": bool(getattr(result, "success", False)),
            "optimizer_message": str(getattr(result, "message", "")),
            "optimizer_nfev": int(getattr(result, "nfev", -1)),
            "overlaps_with_previous": [float(x) for x in overlaps],
        }
        if store_history:
            history.append(record)
        LOGGER.info(
            "%s state=%d iter=%d exact=%.12g energy=%.12g abs_error=%.6g op=%d thetas=%s",
            cost_mode,
            state_index,
            iteration,
            exact_target_energy,
            energy,
            record["abs_energy_error"],
            operator_index,
            np.array2string(np.asarray(thetas), precision=6),
        )
        if previous_objective is not None:
            # For excited-state VQD, the physical energy can increase while the
            # penalized objective decreases because the orthogonality penalty is
            # being reduced. Therefore convergence must be assessed with the
            # VQD objective, not with the physical energy alone.
            last_improvement = float(previous_objective) - float(objective_value)

        small_gradient = abs(float(gradient)) < float(grad_tol)
        stagnant_objective = last_improvement is not None and last_improvement < float(energy_improvement_tol)
        tiny_new_parameter = len(thetas) > 0 and abs(float(thetas[-1])) < float(new_theta_tol)

        # Do not stop only because the last theta is tiny: COBYLA and shot-based
        # objectives can return a near-zero new parameter before the existing
        # ansatz has fully relaxed. Require objective stagnation and a small
        # analytic gradient as well.
        if iteration > 0 and small_gradient:
            break
        if iteration > 0 and stagnant_objective and (small_gradient or tiny_new_parameter):
            break
        previous_objective = float(objective_value)

    final_energy = evaluate_energy(context, thetas, operator_indices, cost_mode=cost_mode, energy_shots=energy_shots, n_jobs=workers)
    state_index = len(previous_ansatze)
    exact_target_energy = float(context.hamiltonian.exact_energies[state_index]) if len(context.hamiltonian.exact_energies) > state_index else np.nan
    energy_error = float(final_energy - exact_target_energy) if np.isfinite(exact_target_energy) else np.nan
    return {
        "state_index": int(state_index),
        "energy": float(final_energy),
        "exact_energy": exact_target_energy,
        "energy_error": energy_error,
        "abs_energy_error": abs(energy_error) if np.isfinite(energy_error) else np.nan,
        "thetas": [float(theta) for theta in thetas],
        "ops": [int(op) for op in operator_indices],
        "n_iterations": int(len(history)),
        "n_layers": int(len(operator_indices)),
        "n_parameters": int(len(thetas)),
        "cost_mode": str(cost_mode),
        "gradient_method": str(gradient_method),
        "overlap_shots": int(overlap_shots),
        "energy_shots": int(energy_shots),
        "optimizer_maxiter": int(optimizer_maxiter),
        "max_adapt_iter": int(max_adapt_iter),
        "max_candidates": None if max_candidates is None else int(max_candidates),
        "grad_tol": float(grad_tol),
        "new_theta_tol": float(new_theta_tol),
        "energy_improvement_tol": float(energy_improvement_tol),
        "beta": float(betas[0]) if betas else 0.0,
        "history": history,
    }


def run_vqd_family(context: VQDContext, *, cost_mode: str, options: dict[str, Any]) -> list[dict[str, Any]]:
    """Run all requested VQD states for one objective-evaluation route."""
    context.measurements.clear_cache()
    results: list[dict[str, Any]] = []
    for state_index in range(int(context.run.n_states)):
        result = run_adapt_vqd(
            context,
            previous_ansatze=results,
            betas=[float(context.run.beta)] * state_index,
            cost_mode=cost_mode,
            **options,
        )
        results.append(result)
    return results


def run_workflow(problem: ProblemConfig, run: RunConfig, numerical: NumericalConfig, full_limits: FullQuantumLimits, *, qibo_backend: str = "numpy") -> dict[str, Any]:
    """Execute the complete VQD workflow and return all numerical results."""
    context = build_context(problem, run, numerical, qibo_backend=qibo_backend)
    beta_check = resolve_beta(run.beta, context.hamiltonian.exact_energies, run.n_states)
    run.beta = float(beta_check["beta"])

    statevector_results: list[dict[str, Any]] = []
    full_quantum_circuit_results: list[dict[str, Any]] = []
    numerical_options = dataclass_to_dict(numerical)
    # measurement_strategy configures MeasurementEngine and is not an ADAPT optimizer option.
    numerical_options.pop("measurement_strategy", None)

    if run.run_statevector:
        statevector_results = run_vqd_family(context, cost_mode="statevector_exact", options=dict(numerical_options))
    if run.run_full_quantum_circuit:
        full_options = apply_optional_caps(dict(numerical_options), dataclass_to_dict(full_limits))
        full_quantum_circuit_results = run_vqd_family(context, cost_mode="full_quantum_circuit", options=full_options)

    return {
        "context": context,
        "beta_check": beta_check,
        "statevector_results": statevector_results,
        "full_quantum_circuit_results": full_quantum_circuit_results,
    }
