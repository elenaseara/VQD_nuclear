"""ADAPT-VQD driver for ground and first-excited nuclear states."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import logging

import numpy as np
from scipy.optimize import minimize

import qibo

from .ansatz import AnsatzBuilder, find_reference_from_hamiltonian_basis, prepare_generator_term_pool, get_hamiltonian_basis_references
from .config import FullQuantumLimits, NumericalConfig, ProblemConfig, RunConfig, apply_optional_caps, dataclass_to_dict, detect_hpc_workers
from .hamiltonian import HamiltonianData, build_hamiltonian_data
from .measurements import MeasurementEngine, apply_qubit_operator_to_state
from .parallel import parallel_map, resolve_n_jobs
from .validation import resolve_beta_comparison
from .model_space import ModelSpaceEngine, model_space_indices


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
    base_operator_indices: tuple[int, ...]
    available_workers: int

    @property
    def active_operator_indices(self) -> list[int]:
        """Compatibility name: contains the full pool, without an activity filter."""
        return list(self.base_operator_indices)


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
    """Build the physical Hamiltonian, minimum-diagonal reference and full pool."""
    configure_qibo_backend(qibo_backend, seed=run.seed)
    available_workers = detect_hpc_workers() if str(run.n_workers).lower() == "auto" else int(run.n_workers)
    hamiltonian = build_hamiltonian_data(problem, n_states=run.n_states)
    reference = find_reference_from_hamiltonian_basis(
        problem,
        hamiltonian.nucleus,
        n_qubits=hamiltonian.n_qubits,
        qubit_hamiltonian=hamiltonian.qubit_hamiltonian,
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
    model_space = None
    if numerical.use_model_space:
        referencias = get_hamiltonian_basis_references(
            hamiltonian.nucleus,
            n_qubits=hamiltonian.n_qubits,
            n_valence_particles=problem.n_valence_protons + problem.n_valence_neutrons,
        )
        model_space = ModelSpaceEngine(
            qubit_hamiltonian=hamiltonian.qubit_hamiltonian,
            qubit_generator_pool=hamiltonian.qubit_generator_pool,
            indices=model_space_indices(referencias, hamiltonian.n_qubits),
            reference_occupied_orbitals=reference.occupied_orbitals,
            n_qubits=hamiltonian.n_qubits,
            apply_operator=apply_qubit_operator_to_state,
        )
    measurements = MeasurementEngine(
        ansatz_builder=ansatz_builder,
        qubit_hamiltonian=hamiltonian.qubit_hamiltonian,
        n_qubits=hamiltonian.n_qubits,
        qubit_generator_pool=hamiltonian.qubit_generator_pool,
        available_workers=available_workers,
        measurement_strategy=numerical.measurement_strategy,
        model_space=model_space,
    )
    # Canonical IDs refer to the entire physically constructed pool. They never
    # depend on a probe of this particular reference or on previous selections.
    base_operator_indices = tuple(range(len(hamiltonian.qubit_generator_pool)))
    return VQDContext(
        problem=problem,
        run=run,
        numerical=numerical,
        hamiltonian=hamiltonian,
        reference=reference,
        ansatz_builder=ansatz_builder,
        measurements=measurements,
        base_operator_indices=base_operator_indices,
        available_workers=available_workers,
    )


def get_candidate_operators(context: VQDContext, max_candidates: int | None = None) -> list[int]:
    """Return base IDs; an explicit max_candidates cap is only a diagnostic limit."""
    candidates = list(context.base_operator_indices)
    if max_candidates is not None:
        candidates = candidates[: min(int(max_candidates), len(candidates))]
    return candidates


def eligible_operator_indices(candidates: list[int], operator_indices: list[int]) -> list[int]:
    """Only the immediately preceding operator is excluded from this iteration."""
    last = operator_indices[-1] if operator_indices else None
    return [mu for mu in candidates if mu != last]


@dataclass
class FiniteDifferenceScan:
    candidates: list[int]
    scores: dict[int, float]


def selection_scores(scan, selection_criterion: str) -> dict[int, float]:
    if isinstance(scan, FiniteDifferenceScan):
        return scan.scores
    return scan.g_energy if selection_criterion == "energy_only" else scan.g_total


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


def select_next_operator(context: VQDContext, thetas: list[float], operator_indices: list[int],
                          previous_ansatze: list[dict[str, Any]], betas: list[float], *, eps: float,
                          max_candidates: int | None, n_jobs: int | str | None, overlap_shots: int, energy_shots: int,
                          cost_mode: str, gradient_method: str = "analytic", selection_criterion: str = "full_vqd",
                          ) -> tuple[int | None, float, Any | None]:
    """Select the eligible candidate, allowing nonconsecutive operator reuse.

    The default uses the analytic gradient of the full VQD objective. For
    excited states, this includes the derivative of the overlap-penalty terms.
    The legacy finite-difference path remains available for comparisons with
    ``gradient_method=finite_difference``.
    """
    candidates = eligible_operator_indices(get_candidate_operators(context, max_candidates), operator_indices)

    if not candidates:
        return None, 0.0, None
    
    if gradient_method == "analytic":
        scan = context.measurements.scan_candidate_gradients(
            thetas, operator_indices, candidates, previous_ansatze, betas,
        )
        scores = scan.g_energy if selection_criterion == "energy_only" else scan.g_total
        if selection_criterion not in ("full_vqd", "energy_only"):
            raise ValueError("selection_criterion must be 'full_vqd' or 'energy_only'.")
        if not all(np.isfinite(scores[mu]) for mu in scan.candidates):
            raise FloatingPointError("Non-finite candidate gradient.")

        best = max(abs(scores[mu]) for mu in scan.candidates)
        treshold = best * (1.0 - 1e-9)

        winner = min(mu for mu in scan.candidates if abs(scores[mu]) >= treshold)
        #winner = max(scan.candidates, key=lambda mu: abs(scores[mu]))

        return int(winner), float(scan.g_total[winner]), scan
    
    if gradient_method == "finite_difference":
        tasks = [(context, int(candidate), list(thetas), list(operator_indices),
                  list(previous_ansatze), list(betas), eps,
                  int(overlap_shots), int(energy_shots), cost_mode)
                 for candidate in candidates]
        gradients = parallel_map(finite_difference_gradient, tasks,
                                 n_jobs=n_jobs, available_workers=context.available_workers)
        if not all(np.isfinite(value) for _, value in gradients):
            raise FloatingPointError("Non-finite candidate gradient.")
        best = max(gradients, key=lambda item: abs(item[1]))
        return int(best[0]), float(best[1]), FiniteDifferenceScan(candidates, dict(gradients))

    raise ValueError("gradient_method must be 'analytic' or 'finite_difference'.")


def optimize_ansatz(context: VQDContext, thetas: list[float], operator_indices: list[int],
                    previous_ansatze: list[dict[str, Any]], betas: list[float], *, overlap_shots: int,
                    energy_shots: int, optimizer_maxiter: int, cobyla_rhobeg=0.03, cobyla_rhoend=5e-3,
                    n_jobs: int | str | None, cost_mode: str):
    """Optimize every parameter while retaining the lowest evaluated full cost.

    In sampled mode this protects measured values, not independently resampled
    energies. No statistical monotonicity guarantee is inferred from it.
    """
    best_x = np.asarray(thetas, dtype=float).copy()
    best_cost = np.inf

    def objective(x: np.ndarray) -> float:
        nonlocal best_x, best_cost
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
        if np.isfinite(value) and value < best_cost:
            best_x, best_cost = np.asarray(x, dtype=float).copy(), float(value)
        return value

    if not float(cobyla_rhobeg) > float(cobyla_rhoend) > 0.0:
        raise ValueError(
            f"COBYLA exige rhobeg > rhoend > 0; recibidos: "
            f"rhobeg={cobyla_rhobeg}, rhoend={cobyla_rhoend}."
        )


    initial_cost = objective(np.asarray(thetas, dtype=float))
    if not np.isfinite(initial_cost):
        raise FloatingPointError("Non-finite initial VQD objective.")
    result = minimize(
        objective,
        np.asarray(thetas, dtype=float),
        method="COBYLA",
        options={"maxiter": int(optimizer_maxiter), 
                 "rhobeg": float(cobyla_rhobeg), 
                 "tol": float(cobyla_rhoend)},
    )
    # Use the actual cost function, not the optimizer's potentially stale fun.
    result.raw_x = np.asarray(result.x, dtype=float).copy()
    result.raw_fun = float(result.fun)
    result.end_cost = float(objective(result.raw_x))
    result.start_cost = float(objective(np.asarray(thetas, dtype=float)))
    result.initial_cost = float(initial_cost)
    result.accepted_source = ("initial" if np.array_equal(best_x, thetas) else
                              "optimizer_result" if np.array_equal(best_x, result.raw_x) else "best_evaluated")
    result.x, result.fun = best_x.copy(), float(best_cost)
    return [float(x) for x in best_x], float(best_cost), result


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


def exact_state_diagnostics(context, thetas, operator_indices):
    """Validation only: exact eigenvectors never influence operator selection."""
    data = context.hamiltonian
    if data.exact_vectors is None:
        return {}
    references = get_hamiltonian_basis_references(
        data.nucleus, n_qubits=data.n_qubits,
        n_valence_particles=context.problem.n_valence_protons + context.problem.n_valence_neutrons,
        orbital_mapping=data.orbital_mapping,
    )
    indices = model_space_indices(references, data.n_qubits)
    state = np.asarray(context.ansatz_builder.state(thetas, operator_indices))[indices]
    fidelities = np.abs(data.exact_vectors.conj().T @ state) ** 2
    return {"exact_state_fidelities": fidelities.tolist(),
            "best_matching_exact_state": int(np.argmax(fidelities)),
            "shell_sector_probability": float(np.vdot(state, state).real)}


def run_adapt_vqd(context: VQDContext, *, previous_ansatze: list[dict[str, Any]] | None = None, betas: list[float] | None = None,
                  max_adapt_iter: int = 20, grad_tol: float = 2e-4, max_candidates: int | None = None, n_jobs: int | str | None = None,
                  overlap_shots: int = 200000, energy_shots: int = 200000, optimizer_maxiter: int = 200, cobyla_rhobeg: float = 0.03,
                  cobyla_rhoend: float = 5e-3, new_theta_tol: float = 1e-3, energy_improvement_tol: float = 1e-4,
                  cost_mode: str = "statevector_exact", gradient_method: str = "analytic", selection_criterion: str = "full_vqd",
                  collect_gradient_diagnostics: bool = False, store_history: bool = True,
                  identity_tol: float = 1e-10, acceptance_tol: float = 1e-12) -> dict[str, Any]:
    """Run ADAPT with a fixed base pool, safe extension and explicit termination.

    The existing gradient formulas and selection modes are preserved. Gradient
    convergence is assessed before extension over all eligible candidates.
    Stagnation stops only in combination with a small new parameter; either
    threshold set to zero disables that combined criterion.
    """
    # Copy only the fixed ansatz data, not mutable previous result histories.
    previous_ansatze = [{"thetas": list(p["thetas"]), "ops": list(p["ops"])} for p in (previous_ansatze or [])]
    betas = [float(beta) for beta in (betas or [])]
    if len(betas) != len(previous_ansatze) or not all(np.isfinite(b) and b >= 0 for b in betas):
        raise ValueError("One finite, nonnegative beta is required per previous state.")
    if max_adapt_iter < 0 or optimizer_maxiter < 1 or (max_candidates is not None and max_candidates < 0):
        raise ValueError("Invalid iteration/candidate limit.")
    tolerances = [grad_tol, new_theta_tol, energy_improvement_tol, identity_tol, acceptance_tol]
    if not all(np.isfinite(t) and t >= 0 for t in tolerances):
        raise ValueError("Numerical tolerances must be finite and nonnegative.")
    if not np.isfinite(cobyla_rhobeg) or not float(cobyla_rhobeg) > float(cobyla_rhoend) > 0:
        raise ValueError("COBYLA requires finite rhobeg > rhoend > 0.")
    if cost_mode not in {"statevector_exact", "full_quantum_circuit"}:
        raise ValueError("Unknown objective mode.")
    if selection_criterion not in {"full_vqd", "energy_only"} or gradient_method not in {"analytic", "finite_difference"}:
        raise ValueError("Unknown gradient selection mode.")
    workers = resolve_n_jobs(n_jobs, context.available_workers)
    thetas, operator_indices, history, diagnostics = [], [], [], []
    state_index = len(previous_ansatze)
    pool = tuple(context.base_operator_indices)
    candidates = get_candidate_operators(context, max_candidates)
    common = dict(overlap_shots=overlap_shots, energy_shots=energy_shots, n_jobs=workers, cost_mode=cost_mode)

    def cost(theta, ops):
        return vqd_cost(context, theta, ops, previous_ansatze, betas, **common)

    def energy(theta, ops):
        return evaluate_energy(context, theta, ops, cost_mode=cost_mode, energy_shots=energy_shots, n_jobs=workers)

    reference_id = "[" + ",".join(str(q) for q in context.reference.occupied_orbitals) + "]"
    base_record = dict(reference_id=reference_id, reference_diagonal_energy=float(context.reference.diagonal_energy),
                       pool_size_initial=len(pool), pool_size_after_reference=len(pool),
                       candidate_pool_size=len(candidates), pool_cap_applied=max_candidates is not None,
                       objective_mode=cost_mode, selection_criterion=selection_criterion,
                       grad_tol=float(grad_tol), identity_tol=float(identity_tol), acceptance_tol=float(acceptance_tol),
                       energy_improvement_tol=float(energy_improvement_tol), new_theta_tol=float(new_theta_tol))
    stop_reason, n_iterations = "max_iterations", 0
    accepted_cost = float(cost(thetas, operator_indices))
    stop_details = dict(base_record, eligible_pool_size=len(candidates), eligible_operator_ids=list(candidates))
    if not np.isfinite(accepted_cost):
        stop_reason = "numerical_error"
        stop_details["error"] = "Non-finite reference objective."

    for iteration in range(int(max_adapt_iter)) if stop_reason != "numerical_error" else []:
        eligible = eligible_operator_indices(candidates, operator_indices)
        current_energy = float(energy(thetas, operator_indices))
        record = dict(base_record, iteration=iteration, eligible_pool_size=len(eligible), eligible_operator_ids=eligible,
                      selected_operator_id=None, selected_operator=None, selected_gradient=None,
                      max_abs_gradient=0.0, grad_norm=0.0, selected_operator_history=list(operator_indices),
                      energy_before=current_energy, energy_after=current_energy, vqd_cost_before=float(accepted_cost),
                      vqd_cost_after=float(accepted_cost), physical_energy=current_energy, vqd_objective=float(accepted_cost),
                      optimizer_success=None, optimizer_message="", accepted_candidate=False, new_theta=None,
                      layer_added=False, n_layers=len(operator_indices), n_parameters=len(thetas), stop_reason=None)
        if not np.isfinite(current_energy):
            stop_reason = "numerical_error"
            record["error"] = "Non-finite physical energy."
        elif not eligible:
            stop_reason = "no_eligible_operators"
        else:
            try:
                operator_index, gradient, scan = select_next_operator(
                    context, thetas, operator_indices, previous_ansatze, betas,
                    eps=1e-2, max_candidates=max_candidates, n_jobs=workers,
                    overlap_shots=overlap_shots, energy_shots=energy_shots,
                    cost_mode=cost_mode, gradient_method=gradient_method, selection_criterion=selection_criterion,
                )
                scores = selection_scores(scan, selection_criterion)
                values = np.asarray([scores[mu] for mu in eligible], dtype=float)
                record.update(selected_operator_id=int(operator_index), selected_operator=int(operator_index),
                              selected_gradient=float(scores[operator_index]), adapt_gradient=float(gradient),
                              max_abs_gradient=float(np.max(np.abs(values))), grad_norm=float(np.linalg.norm(values)))
                if collect_gradient_diagnostics and gradient_method == "analytic":
                    from .gradient_diagnostics import build_diagnostics
                    diagnostic = build_diagnostics(scan, state_index=state_index, iteration=iteration)
                    diagnostics.append(diagnostic)
                    record.update(gradient_ranking_changed=diagnostic.ranking_changed, gradient_regret=diagnostic.regret,
                                  winner_energy_only=diagnostic.winner_energy_only)
                if record["max_abs_gradient"] <= grad_tol:
                    stop_reason = "gradient_converged"
                else:
                    old_cost, energy_before = float(cost(thetas, operator_indices)), float(energy(thetas, operator_indices))
                    initial_thetas, new_ops = list(thetas) + [0.0], list(operator_indices) + [int(operator_index)]
                    start_cost = float(cost(initial_thetas, new_ops))
                    record.update(energy_before=energy_before, vqd_cost_before=old_cost,
                                  initial_thetas=initial_thetas, initial_vqd_cost=start_cost,
                                  zero_parameter_objective_error=start_cost - old_cost)
                    if cost_mode == "statevector_exact":
                        old_state = context.ansatz_builder.state(thetas, operator_indices)
                        start_state = context.ansatz_builder.state(initial_thetas, new_ops)
                        state_error = float(np.linalg.norm(start_state - old_state))
                        record["zero_parameter_state_error"] = state_error
                        if (not np.isfinite(start_cost) or abs(start_cost - old_cost) > identity_tol or
                                not np.isfinite(state_error) or state_error > identity_tol):
                            raise FloatingPointError("Zero-parameter extension changed the state or VQD objective.")
                    n_iterations += 1
                    try:
                        optimized, objective_value, result = optimize_ansatz(
                            context, initial_thetas, new_ops, previous_ansatze, betas,
                            optimizer_maxiter=optimizer_maxiter, cobyla_rhobeg=cobyla_rhobeg,
                            cobyla_rhoend=cobyla_rhoend, **common,
                        )
                    except RuntimeError as exc:
                        stop_reason = "optimizer_failure"
                        record["optimizer_success"], record["optimizer_message"] = False, str(exc)
                    else:
                        # Reevaluation uses the same fixed objective as both baselines.
                        checked_cost = float(cost(optimized, new_ops))
                        if not np.isfinite(checked_cost):
                            raise FloatingPointError("Non-finite optimized objective.")
                        if cost_mode == "statevector_exact" and abs(checked_cost - objective_value) > acceptance_tol:
                            raise FloatingPointError("Optimizer point and stored objective disagree.")
                        if checked_cost > start_cost:
                            optimized, checked_cost = initial_thetas, start_cost
                            result.accepted_source = "initial"
                        improvement = float(old_cost - checked_cost)
                        if cost_mode == "statevector_exact" and checked_cost > accepted_cost + identity_tol + acceptance_tol:
                            raise FloatingPointError("Accepted VQD objective would increase.")
                        thetas, operator_indices, accepted_cost = list(optimized), new_ops, checked_cost
                        physical_energy = float(energy(thetas, operator_indices))
                        overlaps = compute_previous_overlaps(context, thetas, operator_indices, previous_ansatze,
                                                           n_jobs=workers, cost_mode=cost_mode, overlap_shots=overlap_shots)
                        target_energy = (float(context.hamiltonian.exact_energies[state_index])
                                         if len(context.hamiltonian.exact_energies) > state_index else np.nan)
                        record.update(layer_added=True, selected_operator_history=list(operator_indices),
                                      energy_after=physical_energy, physical_energy=physical_energy,
                                      vqd_cost_after=accepted_cost, vqd_objective=accepted_cost,
                                      objective_improvement=improvement, thetas=list(thetas), new_theta=float(thetas[-1]),
                                      n_layers=len(operator_indices), n_parameters=len(thetas),
                                      optimizer_success=bool(result.success), optimizer_message=str(result.message),
                                      optimizer_nfev=int(getattr(result, "nfev", -1)),
                                      optimizer_raw_objective=float(result.raw_fun), optimizer_end_cost=float(result.end_cost),
                                      optimizer_start_cost=float(result.start_cost), optimizer_raw_thetas=result.raw_x.tolist(),
                                      accepted_source=result.accepted_source,
                                      accepted_candidate=result.accepted_source != "initial",
                                      overlaps_with_previous=[float(x) for x in overlaps],
                                      exact_target_energy=target_energy, energy_error=physical_energy - target_energy,
                                      abs_energy_error=abs(physical_energy - target_energy))
                        stagnant = energy_improvement_tol > 0 and improvement < energy_improvement_tol
                        tiny = new_theta_tol > 0 and abs(thetas[-1]) < new_theta_tol
                        record.update(stagnant_objective=bool(stagnant), small_new_theta=bool(tiny))
                        if not np.isfinite(result.end_cost):
                            stop_reason = "numerical_error"
                        elif not result.success and improvement <= acceptance_tol:
                            stop_reason = "optimizer_failure"
                        elif iteration > 0 and stagnant and tiny:
                            stop_reason = "energy_stagnation"
            except (FloatingPointError, np.linalg.LinAlgError) as exc:
                stop_reason = "numerical_error"
                record["error"] = str(exc)
        if stop_reason != "max_iterations" or iteration == int(max_adapt_iter) - 1:
            record["stop_reason"] = stop_reason
        stop_details = dict(record)
        if store_history:
            history.append(record)
        LOGGER.info("ADAPT state=%d iter=%d eligible=%d op=%s grad_max=%s E=%s F=%s accepted=%s stop=%s",
                    state_index, iteration, len(eligible), record["selected_operator_id"], record["max_abs_gradient"],
                    record["energy_after"], record["vqd_cost_after"], record["accepted_candidate"], record["stop_reason"])
        if stop_reason != "max_iterations":
            break
    LOGGER.info("ADAPT state=%d stop_reason=%s details=%s", state_index, stop_reason, stop_details)
    final_energy = float(energy(thetas, operator_indices))
    exact_energy = (float(context.hamiltonian.exact_energies[state_index])
                    if len(context.hamiltonian.exact_energies) > state_index else np.nan)
    validation = exact_state_diagnostics(context, thetas, operator_indices) if cost_mode == "statevector_exact" else {}
    return dict(validation, state_index=state_index, energy=final_energy, exact_energy=exact_energy,
                energy_error=final_energy - exact_energy, abs_energy_error=abs(final_energy - exact_energy),
                thetas=[float(t) for t in thetas], ops=list(operator_indices), n_iterations=n_iterations,
                n_layers=len(operator_indices), n_parameters=len(thetas), stop_reason=stop_reason,
                stop_details=stop_details, vqd_objective=float(accepted_cost),
                reference_id=reference_id, reference_diagonal_energy=float(context.reference.diagonal_energy),
                pool_size_initial=len(pool), pool_size_after_reference=len(pool), base_operator_ids=list(pool),
                candidate_operator_ids=candidates, operator_reuse_policy="nonconsecutive",
                identity_tol=float(identity_tol), acceptance_tol=float(acceptance_tol),
                cost_mode=cost_mode, gradient_method=gradient_method, selection_criterion=selection_criterion,
                overlap_shots=int(overlap_shots), energy_shots=int(energy_shots), optimizer_maxiter=int(optimizer_maxiter),
                cobyla_rhobeg=float(cobyla_rhobeg), cobyla_rhoend=float(cobyla_rhoend),
                max_adapt_iter=int(max_adapt_iter), max_candidates=max_candidates, grad_tol=float(grad_tol),
                new_theta_tol=float(new_theta_tol), energy_improvement_tol=float(energy_improvement_tol),
                beta=betas[0] if betas else 0.0, history=history,
                gradient_diagnostics=[d.as_row() for d in diagnostics])


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
    # same with use_model_space to create ModelSpaceEngine
    numerical_options.pop("measurement_strategy", None)
    numerical_options.pop("use_model_space", None)

    for beta_method in ("exact", "approximate"):
        state_betas = beta_check[beta_method]["betas"]
        LOGGER.info("Starting beta_method=%s with beta by state=%s", beta_method, state_betas)
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
