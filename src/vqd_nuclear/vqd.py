"""ADAPT-VQD driver for ground and excited nuclear states."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import logging

import numpy as np
from scipy.optimize import minimize

import qibo

from .ansatz import AnsatzBuilder, find_reference_from_hamiltonian_basis, get_hamiltonian_basis_references, prepare_generator_term_pool
from .config import FullQuantumLimits, NumericalConfig, ProblemConfig, RunConfig, apply_optional_caps, dataclass_to_dict, detect_hpc_workers
from .hamiltonian import HamiltonianData, build_hamiltonian_data
from .measurements import MeasurementEngine
from .parallel import parallel_map, resolve_n_jobs
from .validation import resolve_beta_comparison

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
        orbital_mapping=hamiltonian.orbital_mapping,
        shell_diagonal=hamiltonian.matrix.diagonal(),
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


def get_candidate_operators(context: VQDContext, max_candidates: int | None = None, *, pool_policy: str = "reference_active") -> list[int]:
    """Return active candidate operators, optionally truncated for quick tests."""
    if pool_policy not in {"reference_active", "full"}:
        raise ValueError("pool_policy must be 'reference_active' or 'full'.")
    candidates = (list(context.active_operator_indices) if pool_policy == "reference_active"
                  else list(range(len(context.hamiltonian.qubit_generator_pool))))
    if max_candidates is not None:
        candidates = candidates[: min(int(max_candidates), len(candidates))]
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


def _eligible_operator_indices(candidates: list[int], operator_indices: list[int]) -> list[int]:
    """Allow reuse except for the operator selected in the preceding iteration."""
    last_operator = operator_indices[-1] if operator_indices else None
    return [idx for idx in candidates if idx != last_operator]


def select_next_operator(context: VQDContext, thetas: list[float], operator_indices: list[int], previous_ansatze: list[dict[str, Any]], betas: list[float], *, eps: float, max_candidates: int | None, n_jobs: int | str | None, overlap_shots: int, energy_shots: int, cost_mode: str, gradient_method: str = "analytic", pool_policy: str = "reference_active") -> tuple[int | None, float]:
    """Select the eligible candidate with the largest absolute ADAPT gradient.

    The default uses the analytic gradient of the full VQD objective. For
    excited states, this includes the derivative of the overlap-penalty terms.
    The legacy finite-difference path remains available for comparisons with
    ``gradient_method=finite_difference``.
    """
    candidates = _eligible_operator_indices(
        get_candidate_operators(context, max_candidates, pool_policy=pool_policy), operator_indices,
    )
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
    if not all(np.isfinite(value) for _, value in gradients):
        raise ValueError("Non-finite ADAPT gradient.")
    return max(gradients, key=lambda item: abs(item[1]))


def optimize_ansatz(context: VQDContext, thetas: list[float], operator_indices: list[int], previous_ansatze: list[dict[str, Any]], betas: list[float], *, overlap_shots: int, energy_shots: int, optimizer_maxiter: int, n_jobs: int | str | None, cost_mode: str, optimizer_tol: float = 5e-3):
    """Optimize all parameters, retaining the best evaluated point, including x0.

    In sampled mode this protects the measured objective only; independent
    measurements still fluctuate and do not imply exact monotonicity.
    """
    best_x = np.asarray(thetas, dtype=float).copy()
    best_value = np.inf

    def objective(x: np.ndarray) -> float:
        nonlocal best_x, best_value
        value = vqd_cost(
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
        if np.isfinite(value) and value < best_value:
            best_x, best_value = np.asarray(x, dtype=float).copy(), float(value)
        return value

    initial_value = objective(best_x)
    if not np.isfinite(initial_value):
        raise ValueError("Non-finite initial VQD objective.")
    result = minimize(
        objective,
        np.asarray(thetas, dtype=float),
        method="COBYLA",
        options={"maxiter": int(optimizer_maxiter), "rhobeg": 0.03, "tol": float(optimizer_tol)},
    )
    result.raw_x = np.asarray(result.x).copy()
    result.raw_fun = float(result.fun)
    result.initial_fun = float(initial_value)
    result.used_best_evaluated = not np.array_equal(result.raw_x, best_x)
    result.x, result.fun = best_x, best_value
    return list(best_x), float(best_value), result


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


def exact_state_diagnostics(context: VQDContext, thetas: list[float], operator_indices: list[int]) -> dict[str, Any]:
    """Project the simulator state onto the ordered shell-model eigenbasis.

    These are simulator validation quantities, not sampled measurements. For
    degeneracies, individual eigenvector fidelities depend on the chosen basis.
    """
    data = context.hamiltonian
    if data.exact_vectors is None:
        return {}
    references = get_hamiltonian_basis_references(
        data.nucleus, n_qubits=data.n_qubits,
        n_valence_particles=context.problem.n_valence_protons + context.problem.n_valence_neutrons,
        orbital_mapping=data.orbital_mapping,
    )
    indices = [sum(1 << (data.n_qubits - 1 - q) for q in occupied) for occupied in references]
    state = np.asarray(context.ansatz_builder.state(thetas, operator_indices))[indices]
    fidelities = np.abs(data.exact_vectors.conj().T @ state) ** 2
    target = int(np.argmax(fidelities))
    h_state = data.matrix @ state
    energy = np.vdot(state, h_state).real
    return {"exact_state_fidelities": fidelities.tolist(),
            "best_matching_exact_state": int(target),
            "shell_sector_probability": float(np.vdot(state, state).real),
            "energy_variance": float(max(0.0, np.vdot(h_state, h_state).real - energy ** 2))}


def run_adapt_vqd(context: VQDContext, *, previous_ansatze: list[dict[str, Any]] | None = None, betas: list[float] | None = None, max_adapt_iter: int = 20, grad_tol: float = 2e-4, max_candidates: int | None = None, n_jobs: int | str | None = None, overlap_shots: int = 200000, energy_shots: int = 200000, optimizer_maxiter: int = 200, new_theta_tol: float = 1e-3, energy_improvement_tol: float = 1e-4, cost_mode: str = "statevector_exact", gradient_method: str = "analytic", store_history: bool = True, optimizer_tol: float = 5e-3, pool_policy: str = "reference_active") -> dict[str, Any]:
    """Run ADAPT-VQD for one state."""
    previous_ansatze = previous_ansatze or []
    betas = betas or []
    if len(previous_ansatze) != len(betas):
        raise ValueError("VQD requires one beta per previous state.")
    if max_adapt_iter < 0 or optimizer_maxiter < 1 or optimizer_tol <= 0 or grad_tol < 0:
        raise ValueError("Invalid ADAPT/optimizer iteration limit or tolerance.")
    if new_theta_tol < 0 or energy_improvement_tol < 0:
        raise ValueError("Stagnation tolerances must be nonnegative; zero disables them.")
    workers = resolve_n_jobs(n_jobs, context.available_workers)
    thetas: list[float] = []
    operator_indices: list[int] = []
    history: list[dict[str, Any]] = []
    candidates = get_candidate_operators(context, max_candidates, pool_policy=pool_policy)
    objective_options = dict(overlap_shots=overlap_shots, energy_shots=energy_shots, n_jobs=workers, cost_mode=cost_mode)
    previous_objective = vqd_cost(context, thetas, operator_indices, previous_ansatze, betas, **objective_options)
    last_improvement: float | None = None
    stop_reason = "max_iterations"
    stop_details: dict[str, Any] = {
        "pool_size": len(context.hamiltonian.qubit_generator_pool),
        "reference_active_count": len(context.active_operator_indices),
        "candidate_count": len(candidates), "eligible_count": len(candidates),
        "selected_operator": None, "max_abs_gradient": None, "grad_tol": float(grad_tol),
    }
    n_iterations = 0
    monotonicity_tol = 1e-10

    for iteration in range(int(max_adapt_iter)):
        eligible = _eligible_operator_indices(candidates, operator_indices)
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
            pool_policy=pool_policy,
        )
        stop_details = {
            "pool_size": len(context.hamiltonian.qubit_generator_pool),
            "reference_active_count": len(context.active_operator_indices),
            "candidate_count": len(candidates),
            "eligible_count": len(eligible),
            "selected_operator": operator_index,
            "max_abs_gradient": abs(float(gradient)),
            "grad_tol": float(grad_tol),
        }
        if operator_index is None:
            stop_reason = "pool_exhausted"
            break
        if abs(float(gradient)) < float(grad_tol):
            stop_reason = "gradient_converged"
            break
        energy_before = evaluate_energy(context, thetas, operator_indices, cost_mode=cost_mode, energy_shots=energy_shots, n_jobs=workers)
        old_thetas, old_ops = list(thetas), list(operator_indices)
        operator_indices.append(int(operator_index))
        thetas.append(0.0)
        initial_thetas = list(thetas)
        initial_objective = vqd_cost(context, thetas, operator_indices, previous_ansatze, betas, **objective_options)
        embedding_error = float(initial_objective - previous_objective)
        if cost_mode == "statevector_exact":
            if (abs(embedding_error) > monotonicity_tol or not np.allclose(
                context.ansatz_builder.state(old_thetas, old_ops),
                context.ansatz_builder.state(thetas, operator_indices), rtol=0, atol=monotonicity_tol,
            )):
                raise RuntimeError("Appending a zero parameter changed the exact ansatz/objective.")
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
            optimizer_tol=optimizer_tol,
        )
        if not np.isfinite(objective_value) or objective_value > initial_objective:
            thetas, objective_value = initial_thetas, initial_objective
        # Include the reference objective in the first improvement as well.
        last_improvement = float(previous_objective - objective_value)
        n_iterations += 1
        energy = evaluate_energy(context, thetas, operator_indices, cost_mode=cost_mode, energy_shots=energy_shots, n_jobs=workers)
        overlaps = compute_previous_overlaps(context, thetas, operator_indices, previous_ansatze, n_jobs=workers, cost_mode=cost_mode, overlap_shots=overlap_shots)
        state_index = len(previous_ansatze)
        exact_target_energy = float(context.hamiltonian.exact_energies[state_index]) if len(context.hamiltonian.exact_energies) > state_index else np.nan
        energy_error = float(energy - exact_target_energy) if np.isfinite(exact_target_energy) else np.nan
        record = {
            **stop_details,
            "iteration": int(iteration),
            "selected_operator": int(operator_index),
            "adapt_gradient": float(gradient),
            "physical_energy": float(energy),
            "vqd_objective": float(objective_value),
            "physical_energy_before": float(energy_before),
            "vqd_objective_before": float(previous_objective),
            "initial_vqd_objective": float(initial_objective),
            "zero_parameter_objective_error": embedding_error,
            "objective_improvement": last_improvement,
            "initial_thetas": initial_thetas,
            "thetas": list(thetas),
            "new_theta": float(thetas[-1]),
            "optimizer_raw_objective": float(result.raw_fun),
            "optimizer_raw_thetas": [float(x) for x in result.raw_x],
            "optimizer_used_best_evaluated": bool(result.used_best_evaluated),
            "energy_improvement_tol": float(energy_improvement_tol),
            "new_theta_tol": float(new_theta_tol),
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
        if cost_mode == "statevector_exact":
            record.update(exact_state_diagnostics(context, thetas, operator_indices))
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
        stagnant_objective = energy_improvement_tol > 0 and last_improvement < float(energy_improvement_tol)
        tiny_new_parameter = new_theta_tol > 0 and abs(float(thetas[-1])) < float(new_theta_tol)
        record["stagnant_objective"] = bool(stagnant_objective)
        record["tiny_new_parameter"] = bool(tiny_new_parameter)
        # A budget-limited optimizer can still yield a useful, accepted step.
        # Report failure only when it also failed to improve the objective.
        if not result.success and last_improvement <= monotonicity_tol:
            stop_reason = "optimization_failure"
        elif iteration > 0 and stagnant_objective and tiny_new_parameter:
            stop_reason = "energy_stagnation"
        else:
            stop_reason = "max_iterations"
        record["stop_reason"] = stop_reason if stop_reason != "max_iterations" else None
        previous_objective = float(objective_value)
        if stop_reason != "max_iterations":
            break

    stop_details.update({"energy_improvement_tol": float(energy_improvement_tol), "new_theta_tol": float(new_theta_tol),
                         "last_improvement": last_improvement, "new_theta": float(thetas[-1]) if thetas else None})
    if history:
        history[-1]["stop_reason"] = stop_reason
    LOGGER.info("ADAPT state=%d stop_reason=%s diagnostics=%s", len(previous_ansatze), stop_reason, stop_details)
    final_energy = evaluate_energy(context, thetas, operator_indices, cost_mode=cost_mode, energy_shots=energy_shots, n_jobs=workers)
    state_index = len(previous_ansatze)
    exact_target_energy = float(context.hamiltonian.exact_energies[state_index]) if len(context.hamiltonian.exact_energies) > state_index else np.nan
    energy_error = float(final_energy - exact_target_energy) if np.isfinite(exact_target_energy) else np.nan
    return {
        **(exact_state_diagnostics(context, thetas, operator_indices) if cost_mode == "statevector_exact" else {}),
        "state_index": int(state_index),
        "energy": float(final_energy),
        "exact_energy": exact_target_energy,
        "energy_error": energy_error,
        "abs_energy_error": abs(energy_error) if np.isfinite(energy_error) else np.nan,
        "thetas": [float(theta) for theta in thetas],
        "ops": [int(op) for op in operator_indices],
        "n_iterations": n_iterations,
        "stop_reason": stop_reason,
        "stop_details": stop_details,
        "vqd_objective": float(previous_objective),
        "monotonicity_tol": monotonicity_tol,
        "pool_policy": pool_policy,
        "pool_indices": list(range(len(context.hamiltonian.qubit_generator_pool))),
        "reference_active_operator_indices": list(context.active_operator_indices),
        "candidate_operator_indices": candidates,
        "n_layers": int(len(operator_indices)),
        "n_parameters": int(len(thetas)),
        "cost_mode": str(cost_mode),
        "gradient_method": str(gradient_method),
        "overlap_shots": int(overlap_shots),
        "energy_shots": int(energy_shots),
        "optimizer_maxiter": int(optimizer_maxiter),
        "optimizer_tol": float(optimizer_tol),
        "max_adapt_iter": int(max_adapt_iter),
        "max_candidates": None if max_candidates is None else int(max_candidates),
        "grad_tol": float(grad_tol),
        "new_theta_tol": float(new_theta_tol),
        "energy_improvement_tol": float(energy_improvement_tol),
        "beta": float(betas[0]) if betas else 0.0,
        "history": history,
    }


def run_vqd_family(context: VQDContext, *, cost_mode: str, options: dict[str, Any],
                   beta_method: str, state_betas: list[float]) -> list[dict[str, Any]]:
    """Run all requested VQD states for one objective-evaluation route."""
    context.measurements.clear_cache()
    results: list[dict[str, Any]] = []
    for state_index in range(int(context.run.n_states)):
        result = run_adapt_vqd(
            context,
            previous_ansatze=results,
            betas=[float(state_betas[state_index])] * state_index,
            cost_mode=cost_mode,
            **options,
        )
        result["beta_method"] = beta_method
        results.append(result)
    return results


def run_workflow(problem: ProblemConfig, run: RunConfig, numerical: NumericalConfig, full_limits: FullQuantumLimits, *, qibo_backend: str = "numpy") -> dict[str, Any]:
    """Execute the complete VQD workflow and return all numerical results."""
    context = build_context(problem, run, numerical, qibo_backend=qibo_backend)
    beta_check = resolve_beta_comparison(
        context.hamiltonian.exact_energies, context.hamiltonian.matrix, run.n_states,
    )

    statevector_results: list[dict[str, Any]] = []
    full_quantum_circuit_results: list[dict[str, Any]] = []
    numerical_options = dataclass_to_dict(numerical)
    # measurement_strategy configures MeasurementEngine and is not an ADAPT optimizer option.
    numerical_options.pop("measurement_strategy", None)

    for beta_method in ("exact", "approximate"):
        state_betas = beta_check[beta_method]["betas"]
        if run.run_statevector:
            set_reproducible_seed(run.seed)
            statevector_results.extend(run_vqd_family(
                context, cost_mode="statevector_exact", options=dict(numerical_options),
                beta_method=beta_method, state_betas=state_betas,
            ))
        if run.run_full_quantum_circuit:
            set_reproducible_seed(run.seed)
            full_options = apply_optional_caps(dict(numerical_options), dataclass_to_dict(full_limits))
            full_quantum_circuit_results.extend(run_vqd_family(
                context, cost_mode="full_quantum_circuit", options=full_options,
                beta_method=beta_method, state_betas=state_betas,
            ))

    return {
        "context": context,
        "beta_check": beta_check,
        "statevector_results": statevector_results,
        "full_quantum_circuit_results": full_quantum_circuit_results,
    }
