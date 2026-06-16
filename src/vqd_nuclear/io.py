"""Result serialization for VQD runs."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
import json

import numpy as np
import pandas as pd

from .config import OutputConfig, ProblemConfig, RunConfig, NumericalConfig, FullQuantumLimits, dataclass_to_dict
if TYPE_CHECKING:
    from .vqd import VQDContext


def as_clean_string(values: list[Any] | tuple[Any, ...] | None, precision: int = 10) -> str:
    """Format lists of operators or parameters without NumPy wrappers."""
    if values is None:
        return ""
    cleaned: list[str] = []
    for value in list(values):
        if isinstance(value, (np.integer, int)):
            cleaned.append(str(int(value)))
        elif isinstance(value, (np.floating, float)):
            cleaned.append(f"{float(value):.{precision}g}")
        else:
            cleaned.append(str(value))
    return "[" + ", ".join(cleaned) + "]"


def serializable_result(result: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe version of one result dictionary."""
    clean: dict[str, Any] = {}
    for key, value in result.items():
        if key == "circuit":
            continue
        if isinstance(value, np.ndarray):
            clean[key] = value.tolist()
        elif isinstance(value, (np.integer,)):
            clean[key] = int(value)
        elif isinstance(value, (np.floating,)):
            clean[key] = float(value)
        else:
            clean[key] = value
    return clean


def build_summary_dataframe(results: list[dict[str, Any]], label: str, run_id: str) -> pd.DataFrame:
    """Build one row per final VQD state."""
    rows: list[dict[str, Any]] = []
    for result in results:
        rows.append({
            "run_id": run_id,
            "method": label,
            "cost_mode": result.get("cost_mode", ""),
            "state": int(result.get("state_index", len(rows))),
            "exact_energy": float(result.get("exact_energy", np.nan)),
            "computed_energy": float(result.get("energy", np.nan)),
            "signed_error": float(result.get("energy_error", np.nan)),
            "abs_error": float(result.get("abs_energy_error", np.nan)),
            "n_iterations": int(result.get("n_iterations", 0)),
            "n_layers": int(result.get("n_layers", 0)),
            "n_parameters": int(result.get("n_parameters", 0)),
            "overlap_shots": int(result.get("overlap_shots", 0)),
            "energy_shots": int(result.get("energy_shots", 0)),
            "optimizer_maxiter": int(result.get("optimizer_maxiter", 0)),
            "max_adapt_iter": int(result.get("max_adapt_iter", 0)),
            "max_candidates": result.get("max_candidates", None),
            "grad_tol": float(result.get("grad_tol", np.nan)),
            "operators": as_clean_string(result.get("ops", []), precision=0),
            "thetas": as_clean_string(result.get("thetas", []), precision=12),
        })
    return pd.DataFrame(rows)


def build_energy_sorted_dataframe(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Compare computed states with exact energies after sorting by energy.

    Sequential VQD can return mutually orthogonal states in an order that differs
    from the exact-energy ordering when the first ADAPT run does not converge to
    the true ground state. This diagnostic keeps the original VQD state labels but
    matches the computed energies to the exact energies by energy rank within each
    method/cost-mode block.
    """
    if summary_df is None or summary_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    group_columns = [column for column in ("run_id", "method", "cost_mode") if column in summary_df.columns]

    for group_values, group in summary_df.groupby(group_columns, dropna=False, sort=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        group_data = dict(zip(group_columns, group_values))

        computed_sorted = group.sort_values("computed_energy", kind="mergesort").reset_index(drop=True)
        exact_sorted = group[["state", "exact_energy"]].sort_values("exact_energy", kind="mergesort").reset_index(drop=True)
        n_rows = min(len(computed_sorted), len(exact_sorted))

        for rank in range(n_rows):
            computed_row = computed_sorted.iloc[rank]
            exact_row = exact_sorted.iloc[rank]
            computed_energy = float(computed_row["computed_energy"])
            exact_energy = float(exact_row["exact_energy"])
            signed_error = computed_energy - exact_energy
            row = {
                **group_data,
                "energy_rank": int(rank),
                "computed_state": int(computed_row["state"]),
                "matched_exact_state": int(exact_row["state"]),
                "exact_energy_sorted": exact_energy,
                "computed_energy_sorted": computed_energy,
                "signed_error_sorted": float(signed_error),
                "abs_error_sorted": abs(float(signed_error)),
                "original_exact_state_for_computed_state": int(computed_row["state"]),
                "original_exact_energy_for_computed_state": float(computed_row["exact_energy"]),
                "operators": computed_row.get("operators", ""),
                "thetas": computed_row.get("thetas", ""),
            }
            rows.append(row)

    return pd.DataFrame(rows)



def build_energy_matched_validation_dataframe(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Validate computed states by optimal energy matching, not by VQD label.

    The raw VQD state index is an algorithmic label: state 0 is the first run,
    state 1 is the first deflated run, and so on. It is not guaranteed to be the
    same as the exact eigenvalue rank when ADAPT is truncated, when two levels are
    close or degenerate, or when the excited-state penalty is not strong enough.

    This diagnostic builds the absolute-error matrix between every computed
    physical energy and every available exact target energy in each
    method/cost-mode block, then solves the one-to-one assignment that minimizes
    the total absolute energy error. The resulting table is the main validation
    table to inspect before judging whether a calculation is physically good.
    """
    if summary_df is None or summary_df.empty:
        return pd.DataFrame()

    try:
        from scipy.optimize import linear_sum_assignment
    except Exception as exc:  # pragma: no cover - scipy is a project dependency
        raise RuntimeError("scipy is required to build the matched validation summary.") from exc

    rows: list[dict[str, Any]] = []
    group_columns = [column for column in ("run_id", "method", "cost_mode") if column in summary_df.columns]

    for group_values, group in summary_df.groupby(group_columns, dropna=False, sort=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        group_data = dict(zip(group_columns, group_values))

        computed = group.reset_index(drop=True)
        exact = group[["state", "exact_energy"]].drop_duplicates().reset_index(drop=True)
        computed_energies = computed["computed_energy"].astype(float).to_numpy()
        exact_energies = exact["exact_energy"].astype(float).to_numpy()

        valid_computed = np.isfinite(computed_energies)
        valid_exact = np.isfinite(exact_energies)
        if not np.any(valid_computed) or not np.any(valid_exact):
            continue

        computed_indices = np.flatnonzero(valid_computed)
        exact_indices = np.flatnonzero(valid_exact)
        cost = np.abs(computed_energies[computed_indices, None] - exact_energies[exact_indices][None, :])
        assigned_computed_local, assigned_exact_local = linear_sum_assignment(cost)

        for c_local, e_local in zip(assigned_computed_local, assigned_exact_local):
            c_idx = int(computed_indices[int(c_local)])
            e_idx = int(exact_indices[int(e_local)])
            computed_row = computed.iloc[c_idx]
            exact_row = exact.iloc[e_idx]
            computed_energy = float(computed_row["computed_energy"])
            exact_energy = float(exact_row["exact_energy"])
            matched_error = computed_energy - exact_energy
            raw_error = computed_energy - float(computed_row["exact_energy"])
            row = {
                **group_data,
                "computed_state": int(computed_row["state"]),
                "matched_exact_state": int(exact_row["state"]),
                "computed_energy": computed_energy,
                "matched_exact_energy": exact_energy,
                "matched_signed_error": float(matched_error),
                "matched_abs_error": abs(float(matched_error)),
                "raw_index_exact_energy": float(computed_row["exact_energy"]),
                "raw_index_signed_error": float(raw_error),
                "raw_index_abs_error": abs(float(raw_error)),
                "state_label_changed": bool(int(computed_row["state"]) != int(exact_row["state"])),
                "n_layers": int(computed_row.get("n_layers", 0)),
                "n_iterations": int(computed_row.get("n_iterations", 0)),
                "operators": computed_row.get("operators", ""),
                "thetas": computed_row.get("thetas", ""),
            }
            rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        [column for column in ("run_id", "method", "cost_mode", "matched_exact_state") if column in rows[0]],
        kind="mergesort",
    ).reset_index(drop=True)

def build_history_dataframe(results: list[dict[str, Any]], label: str, run_id: str) -> pd.DataFrame:
    """Build one row per ADAPT iteration."""
    rows: list[dict[str, Any]] = []
    for result in results:
        state_index = int(result.get("state_index", 0))
        for record in result.get("history", []):
            row = dict(record)
            row["run_id"] = run_id
            row["method"] = label
            row["state"] = state_index
            row["overlaps_with_previous"] = as_clean_string(row.get("overlaps_with_previous", []), precision=8)
            rows.append(row)
    return pd.DataFrame(rows)


def build_parameters_dataframe(results: list[dict[str, Any]], label: str, run_id: str) -> pd.DataFrame:
    """Build one row per final variational parameter."""
    rows: list[dict[str, Any]] = []
    for result in results:
        state_index = int(result.get("state_index", 0))
        ops = list(result.get("ops", []))
        thetas = list(result.get("thetas", []))
        for parameter_index, theta in enumerate(thetas):
            rows.append({
                "run_id": run_id,
                "method": label,
                "state": state_index,
                "parameter_index": parameter_index,
                "operator_index": int(ops[parameter_index]) if parameter_index < len(ops) else np.nan,
                "theta": float(theta),
                "abs_theta": abs(float(theta)),
            })
    return pd.DataFrame(rows)


def build_overlap_matrix(context: "VQDContext", results: list[dict[str, Any]]) -> pd.DataFrame:
    """Build exact statevector overlap matrix for final statevector ansatze."""
    n_results = len(results)
    matrix = np.zeros((n_results, n_results))
    for i in range(n_results):
        for j in range(i, n_results):
            ri = results[i]
            rj = results[j]
            value = context.measurements.exact_overlap(ri["thetas"], ri["ops"], rj["thetas"], rj["ops"])
            matrix[i, j] = value
            matrix[j, i] = value
    labels = [f"state_{i}" for i in range(n_results)]
    return pd.DataFrame(matrix, index=labels, columns=labels)


def format_table(df: pd.DataFrame, columns: list[str] | None = None, float_format="{:.12g}".format) -> str:
    """Return a readable fixed-width table for text reports."""
    if df is None or df.empty:
        return "No data generated."
    table = df.copy(deep=False)
    if columns is not None:
        table = table[[column for column in columns if column in table.columns]]
    return table.to_string(index=False, na_rep="", float_format=float_format)


def write_single_readable_txt_report(path: Path, *, run_id: str, context: "VQDContext", beta_check: dict[str, Any], problem: ProblemConfig, run: RunConfig, numerical: NumericalConfig, full_limits: FullQuantumLimits, summary_df: pd.DataFrame, history_df: pd.DataFrame, parameters_df: pd.DataFrame, overlap_df: pd.DataFrame, energy_sorted_df: pd.DataFrame | None = None, matched_validation_df: pd.DataFrame | None = None) -> None:
    """Write one clean txt file containing all relevant information from a run."""
    sections: list[str] = []

    def add_section(title: str, body: str) -> None:
        sections.append(title)
        sections.append("=" * len(title))
        sections.append(str(body).rstrip())
        sections.append("")

    metadata_lines = [
        f"Run ID: {run_id}",
        f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Detected CPU workers: {context.available_workers}",
        f"n_qubits: {context.hamiltonian.n_qubits}",
        f"Reference occupied orbitals: {context.reference.occupied_orbitals}",
        f"Reference diagonal energy: {context.reference.diagonal_energy:.12f}",
        f"First active operator: {context.reference.first_active_operator}",
        f"Beta check: {beta_check.get('message', '')}",
        f"Measurement strategy: {context.measurements.measurement_strategy}",
        f"Measurement circuits: {context.measurements.measurement_summary}",
    ]
    add_section("VQD RUN METADATA", "\n".join(metadata_lines))
    add_section("PROBLEM CONFIGURATION", "\n".join(f"{k}: {v}" for k, v in dataclass_to_dict(problem).items()))
    add_section("RUN CONFIGURATION", "\n".join(f"{k}: {v}" for k, v in dataclass_to_dict(run).items()))
    add_section("VQD NUMERICAL CONFIGURATION", "\n".join(f"{k}: {v}" for k, v in dataclass_to_dict(numerical).items()))
    add_section("FULL-QUANTUM CIRCUIT LIMITS", "\n".join(f"{k}: {v}" for k, v in dataclass_to_dict(full_limits).items()))
    exact_lines = [f"state {idx}: {float(e):.12f}" for idx, e in enumerate(context.hamiltonian.exact_energies[: int(run.n_states)])]
    add_section("EXACT REFERENCE ENERGIES", "\n".join(exact_lines))
    add_section("FINAL ENERGY SUMMARY", format_table(summary_df))
    if matched_validation_df is not None and not matched_validation_df.empty:
        add_section("MATCHED VALIDATION SUMMARY", format_table(matched_validation_df))
    if energy_sorted_df is not None and not energy_sorted_df.empty:
        add_section("ENERGY-SORTED SUMMARY", format_table(energy_sorted_df))
    add_section("ADAPT ITERATION HISTORY", format_table(history_df))
    add_section("FINAL VARIATIONAL PARAMETERS", format_table(parameters_df))
    add_section("EXACT STATEVECTOR OVERLAP MATRIX", overlap_df.to_string(float_format="{:.12g}".format) if not overlap_df.empty else "No statevector states were generated.")
    path.write_text("\n".join(sections), encoding="utf-8")


def save_outputs(output: OutputConfig, *, run_id: str, problem: ProblemConfig, run: RunConfig, numerical: NumericalConfig, full_limits: FullQuantumLimits, workflow_output: dict[str, Any]) -> dict[str, Path]:
    """Save text, CSV, and JSON outputs for a workflow run."""
    results_dir = Path(output.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    context: VQDContext = workflow_output["context"]
    statevector_results = workflow_output["statevector_results"]
    full_quantum_circuit_results = workflow_output["full_quantum_circuit_results"]

    summary_frames = []
    history_frames = []
    parameter_frames = []
    if statevector_results:
        summary_frames.append(build_summary_dataframe(statevector_results, "statevector", run_id))
        history_frames.append(build_history_dataframe(statevector_results, "statevector", run_id))
        parameter_frames.append(build_parameters_dataframe(statevector_results, "statevector", run_id))
    if full_quantum_circuit_results:
        summary_frames.append(build_summary_dataframe(full_quantum_circuit_results, "full_quantum_circuit", run_id))
        history_frames.append(build_history_dataframe(full_quantum_circuit_results, "full_quantum_circuit", run_id))
        parameter_frames.append(build_parameters_dataframe(full_quantum_circuit_results, "full_quantum_circuit", run_id))

    summary_df = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    history_df = pd.concat(history_frames, ignore_index=True) if history_frames else pd.DataFrame()
    parameters_df = pd.concat(parameter_frames, ignore_index=True) if parameter_frames else pd.DataFrame()
    energy_sorted_df = build_energy_sorted_dataframe(summary_df) if not summary_df.empty else pd.DataFrame()
    matched_validation_df = build_energy_matched_validation_dataframe(summary_df) if not summary_df.empty else pd.DataFrame()
    overlap_df = build_overlap_matrix(context, statevector_results) if statevector_results else pd.DataFrame()

    paths: dict[str, Path] = {}
    if output.save_text_outputs:
        txt_path = results_dir / f"vqd_results_{run_id}.txt"
        write_single_readable_txt_report(
            txt_path,
            run_id=run_id,
            context=context,
            beta_check=workflow_output["beta_check"],
            problem=problem,
            run=run,
            numerical=numerical,
            full_limits=full_limits,
            summary_df=summary_df,
            history_df=history_df,
            parameters_df=parameters_df,
            overlap_df=overlap_df,
            energy_sorted_df=energy_sorted_df,
            matched_validation_df=matched_validation_df,
        )
        paths["txt_report"] = txt_path
    if output.save_machine_outputs:
        summary_path = results_dir / f"summary_{run_id}.csv"
        history_path = results_dir / f"history_{run_id}.csv"
        parameters_path = results_dir / f"parameters_{run_id}.csv"
        energy_sorted_path = results_dir / f"energy_sorted_summary_{run_id}.csv"
        matched_validation_path = results_dir / f"matched_validation_summary_{run_id}.csv"
        overlap_path = results_dir / f"overlap_matrix_{run_id}.csv"
        json_path = results_dir / f"raw_results_{run_id}.json"
        summary_df.to_csv(summary_path, index=False)
        history_df.to_csv(history_path, index=False)
        parameters_df.to_csv(parameters_path, index=False)
        energy_sorted_df.to_csv(energy_sorted_path, index=False)
        matched_validation_df.to_csv(matched_validation_path, index=False)
        overlap_df.to_csv(overlap_path)
        raw = {
            "problem": dataclass_to_dict(problem),
            "run": dataclass_to_dict(run),
            "numerical": dataclass_to_dict(numerical),
            "full_limits": dataclass_to_dict(full_limits),
            "beta_check": workflow_output["beta_check"],
            "statevector_results": [serializable_result(r) for r in statevector_results],
            "full_quantum_circuit_results": [serializable_result(r) for r in full_quantum_circuit_results],
        }
        json_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
        paths.update({
            "summary_csv": summary_path,
            "history_csv": history_path,
            "parameters_csv": parameters_path,
            "energy_sorted_summary_csv": energy_sorted_path,
            "matched_validation_summary_csv": matched_validation_path,
            "overlap_csv": overlap_path,
            "raw_json": json_path,
        })
    return paths
