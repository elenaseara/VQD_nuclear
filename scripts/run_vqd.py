#!/usr/bin/env python3
"""Command-line entry point for nuclear ADAPT-VQD simulations."""
from __future__ import annotations

import argparse
import logging
import os
import re
from datetime import datetime
from pathlib import Path

from vqd_nuclear import FullQuantumLimits, NumericalConfig, OutputConfig, ProblemConfig, RunConfig, run_workflow


def parse_bool(value: str) -> bool:
    """Parse a command-line boolean value."""
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the ground state and excited states of a shell-model nuclear Hamiltonian "
            "using ADAPT-VQD with Qibo circuit simulation."
        )
    )
    parser.add_argument("--shell", choices=["p", "sd"], default="sd")
    parser.add_argument("--n-valence-protons", "--protons", dest="n_valence_protons", type=int, default=2)
    parser.add_argument("--n-valence-neutrons", "--neutrons", dest="n_valence_neutrons", type=int, default=0)
    parser.add_argument("--total-j", "--J", dest="total_j", type=int, default=0)
    parser.add_argument("--n-states", type=int, default=2)
    parser.add_argument(
    "--beta",
    type=parse_beta,
    default="auto",
    help=(
        "VQD penalty coefficient. 'auto' (default) derives it from the exact "
        "spectrum as 2*max_i|E_i - E_0|. An explicit value is respected; if it "
        "is below the largest requested gap you get a warning, not an override."
    ),
)
    parser.add_argument(
        "--mode",
        choices=["statevector_exact", "full_quantum_circuit", "both"],
        default=None,
        help=(
            "Objective-evaluation route. Use statevector_exact for exact simulator energies/overlaps, "
            "full_quantum_circuit for sampled Pauli energies and destructive-SWAP overlaps, or both. "
            "If omitted, the legacy --statevector and --full-quantum-circuit flags are used."
        ),
    )
    parser.add_argument(
        "--statevector",
        type=parse_bool,
        default=True,
        help="Legacy flag. Ignored when --mode is supplied.",
    )
    parser.add_argument(
        "--full-quantum-circuit",
        type=parse_bool,
        default=True,
        help="Legacy flag. Ignored when --mode is supplied.",
    )
    parser.add_argument("--max-adapt-iter", type=int, default=20)
    parser.add_argument("--grad-tol", type=float, default=5e-4)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--optimizer-maxiter", type=int, default=1500)
    parser.add_argument("--cobyla-rhobeg", type=float, default=0.03,
                        help="Radio inicial del simplex de COBYLA.")
    parser.add_argument("--cobyla-rhoend", type=float, default=5e-3,
                        help="Radio final de COBYLA (el 'tol' de SciPy).")
    parser.add_argument("--overlap-shots", type=int, default=200000)
    parser.add_argument("--energy-shots", type=int, default=200000)
    parser.add_argument("--new-theta-tol", type=float, default=1e-3)
    parser.add_argument("--energy-improvement-tol", type=float, default=1e-4)
    parser.add_argument(
        "--gradient-method",
        choices=["analytic", "finite_difference"],
        default="analytic",
        help=(
            "Operator-pool gradient used for ADAPT selection. The default analytic method "
            "uses the full VQD objective, including overlap-penalty derivatives for "
            "excited states, and is more stable than shot-based finite differences."
        ),
    )
    parser.add_argument(
        "--measurement-strategy",
        choices=["paper", "qwc"],
        default="paper",
        help=(
            "Energy-measurement strategy for full_quantum_circuit. 'paper' uses the "
            "shell-model basis-change circuits for diagonal, single-hopping and double-hopping "
            "Hamiltonian terms; 'qwc' uses generic qubit-wise commuting Pauli grouping."
        ),
    )
    parser.add_argument("--no-model-space", dest="use_model_space",
        action="store_false",
        help="Deactivates the fast path that uses the projection of the space"
        "and forces the complete simulation of the circuits for statevector.")
    parser.add_argument("--selection-criterion",
                        choices=["full_vqd", "energy_only"],
                        default="full_vqd",
                        help=(
                            "Selects the criterion used to select operators, between "
                            "g^E + g^penalty (full_vqd) or g^E (energy_only)."
                        ),
    )
    parser.add_argument("--gradient-diagnostics", type=bool, default=False,
        help="Stores, for each ADAPT iteration, the operator that would have"
            "been selected by using g^E alone, instead of g^E + g^penalty.")
    
    parser.add_argument("--n-jobs", default="all")
    parser.add_argument("--n-workers", default="auto")
    parser.add_argument("--qibo-backend", default=os.environ.get("QIBO_BACKEND", "numpy"))
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("runs"),
        help=(
            "Root directory where run folders are created. By default each execution writes all "
            "outputs inside runs/<run-label>/ so the whole folder can be downloaded directly."
        ),
    )
    parser.add_argument(
        "--flat-results",
        action="store_true",
        help="Write output files directly in --results-dir instead of creating a run-specific subfolder.",
    )
    parser.add_argument("--run-label", default=None)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def resolve_run_routes(args: argparse.Namespace) -> tuple[bool, bool]:
    """Resolve the requested execution routes from --mode or legacy boolean flags."""
    if args.mode is None:
        return bool(args.statevector), bool(args.full_quantum_circuit)
    return (
        args.mode in {"statevector_exact", "both"},
        args.mode in {"full_quantum_circuit", "both"},
    )


def safe_path_name(value: str) -> str:
    """Return a filesystem-safe name for a run folder."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._-")
    return cleaned or datetime.now().strftime("run_%Y%m%d_%H%M%S")


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")

    run_statevector, run_full_quantum_circuit = resolve_run_routes(args)

    run_label = args.run_label or f"{args.shell}p{args.n_valence_protons}n{args.n_valence_neutrons}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    problem = ProblemConfig(
        shell=args.shell,
        n_valence_protons=args.n_valence_protons,
        n_valence_neutrons=args.n_valence_neutrons,
        total_j=args.total_j,
    )
    run = RunConfig(
        n_states=args.n_states,
        beta=args.beta,
        run_statevector=run_statevector,
        run_full_quantum_circuit=run_full_quantum_circuit,
        n_workers=args.n_workers,
        seed=args.seed,
    )
    numerical = NumericalConfig(
        max_adapt_iter=args.max_adapt_iter,
        grad_tol=args.grad_tol,
        max_candidates=args.max_candidates,
        optimizer_maxiter=args.optimizer_maxiter,
        cobyla_rhobeg=args.cobyla_rhobeg,
        cobyla_rhoend=args.cobyla_rhoend,
        overlap_shots=args.overlap_shots,
        energy_shots=args.energy_shots,
        new_theta_tol=args.new_theta_tol,
        energy_improvement_tol=args.energy_improvement_tol,
        gradient_method=args.gradient_method,
        measurement_strategy=args.measurement_strategy,
        n_jobs=args.n_jobs,
        selection_criterion=args.selection_criterion,
        collect_gradient_diagnostics=args.gradient_diagnostics,
        use_model_space=args.use_model_space
    )
    full_limits = FullQuantumLimits()
    if args.flat_results:
        results_dir = args.results_dir
    else:
        results_dir = args.results_dir / safe_path_name(run_label)
    output = OutputConfig(results_dir=results_dir, run_label=run_label)

    from vqd_nuclear.io import save_outputs

    workflow_output = run_workflow(problem, run, numerical, full_limits, qibo_backend=args.qibo_backend)
    paths = save_outputs(
        output,
        run_id=run_label,
        problem=problem,
        run=run,
        numerical=numerical,
        full_limits=full_limits,
        workflow_output=workflow_output,
    )
    logging.info("Beta setting: %s", workflow_output["beta_check"].get("message", ""))
    if workflow_output["beta_check"].get("warning"):
        logging.warning("%s", workflow_output["beta_check"]["warning"])
    logging.info("Run folder: %s", output.results_dir)
    for name, path in paths.items():
        logging.info("Saved %s: %s", name, path)
    return 0


def parse_beta(value: str):
    if value.strip().lower() == "auto":
        return "auto"
    try:
        beta = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"beta must be a number or 'auto', got {value!r}")
    if beta < 0.0:
        raise argparse.ArgumentTypeError("beta must be non-negative")
    return beta


if __name__ == "__main__":
    raise SystemExit(main())
