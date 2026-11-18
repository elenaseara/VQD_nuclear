#!/usr/bin/env python3
"""Fidelity of every ADAPT iteration with the exact eigenstates, from the JSON.

For each route, VQD state k and iteration, the ansatz recorded in the history
(`ops`, `thetas`) is rebuilt WITHOUT noise and compared with the exact
eigenvectors of H in the model space:

    F_k = |<phi_k|psi>|^2   (summed over the eigenspace if E_k is degenerate)

This tells whether an iteration really improves the state even when the
(reported or exact) energy gets worse, which the energy alone cannot show.
Also printed per iteration: exact energy E(theta), the energy reported by the
run (noisy in the sampled route), the largest contamination, and the exact
overlaps / objective against the FINAL previous states of the same route.

The Hamiltonian is rebuilt with build_context (needs pynucshell); the run itself
is not repeated. Requires a JSON written after 18/11/2026, whose history
records `ops` and `thetas` per iteration.

Usage:
    python fidelity_per_iteration.py raw_results.json
    python fidelity_per_iteration.py raw_results.json --route full_quantum_circuit --csv fid.csv
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

ROUTES = {"statevector": "statevector_results",
          "full_quantum_circuit": "full_quantum_circuit_results"}
DEGENERACY_TOL = 1e-8


# ---------------------------------------------------------------- the basis

class ExactBasis:
    """Model-space Hamiltonian, its eigenvectors and a state builder, all in
    the SAME determinant order.

    Uses the ModelSpaceEngine of the context when there is one (its H_S and its
    propagation share the order of `indices` by construction). Otherwise, or
    for an ansatz it cannot propagate, the full statevector is restricted to
    the model-space indices.
    """

    def __init__(self, ctx, n_valence_particles: int) -> None:
        self.ctx = ctx
        self.ms = getattr(ctx.measurements, "model_space", None)
        if self.ms is not None and getattr(self.ms, "available", False):
            self.indices = np.asarray(self.ms.indices)
            h_s = np.asarray(self.ms.hamiltonian_s)
        else:
            self.ms = None
            from vqd_nuclear.ansatz import get_hamiltonian_basis_references
            from vqd_nuclear.measurements import apply_qubit_operator_to_state
            nq = ctx.hamiltonian.n_qubits
            refs = get_hamiltonian_basis_references(ctx.hamiltonian.nucleus, n_qubits=nq,
                                                    n_valence_particles=n_valence_particles)
            self.indices = np.asarray(sorted({sum(1 << (nq - 1 - int(q)) for q in occ)
                                              for occ in refs}))
            d = len(self.indices)
            h_s = np.zeros((d, d), dtype=complex)
            for b, ib in enumerate(self.indices):
                v = np.zeros(2 ** nq, dtype=complex)
                v[ib] = 1.0
                h_s[:, b] = apply_qubit_operator_to_state(ctx.hamiltonian.qubit_hamiltonian, v, nq)[self.indices]
        self.h_s = h_s
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(h_s)

    @property
    def dimension(self) -> int:
        return len(self.indices)

    def state(self, thetas, ops) -> np.ndarray:
        if self.ms is not None and self.ms.can_handle(ops):
            psi = np.asarray(self.ms.state(thetas, ops))
        else:
            full = np.asarray(self.ctx.ansatz_builder.state(thetas, ops)).ravel()
            psi = full[self.indices]
            leak = 1.0 - float(np.vdot(psi, psi).real)
            if leak > 1e-8:
                print(f"  WARNING: {leak:.2e} of the norm is outside the model space.")
        return psi

    def energy(self, psi) -> float:
        return float(np.vdot(psi, self.h_s @ psi).real)

    def weights(self, psi) -> np.ndarray:
        """|<phi_j|psi>|^2 for every exact eigenvector."""
        return np.abs(self.eigenvectors.conj().T @ psi) ** 2

    def target_level(self, k: int) -> list[int]:
        """Eigenvector indices of the k-th level (more than one if degenerate)."""
        return [j for j, e in enumerate(self.eigenvalues)
                if abs(e - self.eigenvalues[k]) < DEGENERACY_TOL]


# ---------------------------------------------------------------- analysis

def analyse_route(basis: ExactBasis, results: list[dict]) -> list[dict]:
    """One row per (state, iteration), plus the final state of each VQD state."""
    rows: list[dict] = []
    finals = [basis.state(r["thetas"], r["ops"]) for r in results]
    for k, result in enumerate(results):
        history = result.get("history", [])
        if history and ("thetas" not in history[0] or "ops" not in history[0]):
            raise KeyError("missing_ansatz")
        target = basis.target_level(k)
        beta = float(result.get("beta", 0.0))
        previous_f = None
        previous_e = None
        for record in history:
            psi = basis.state(record["thetas"], record["ops"])
            w = basis.weights(psi)
            fidelity = float(sum(w[j] for j in target))
            others = [j for j in range(len(w)) if j not in target]
            worst = max(others, key=lambda j: w[j]) if others else None
            e_exact = basis.energy(psi)
            overlaps = [float(abs(np.vdot(prev, psi)) ** 2) for prev in finals[:k]]
            rows.append({
                "state": k,
                "iteration": int(record["iteration"]),
                "operator": int(record["selected_operator"]),
                "n_layers": int(record["n_layers"]),
                "fidelity": fidelity,
                "delta_fidelity": None if previous_f is None else fidelity - previous_f,
                "exact_energy_theta": e_exact,
                "delta_exact_energy": None if previous_e is None else e_exact - previous_e,
                "reported_energy": record.get("physical_energy"),
                "target_eigenvalue": float(basis.eigenvalues[k]),
                "main_contamination": None if worst is None else f"phi_{worst}",
                "main_contamination_weight": None if worst is None else float(w[worst]),
                "exact_overlaps_with_previous": overlaps,
                "exact_objective": e_exact + beta * sum(overlaps),
                "weights": [float(x) for x in w],
            })
            previous_f, previous_e = fidelity, e_exact
    return rows


def print_route(name: str, rows: list[dict], basis: ExactBasis) -> None:
    print(f"\n{'=' * 78}\nRoute: {name}\n{'=' * 78}")
    for k in sorted({r["state"] for r in rows}):
        state_rows = [r for r in rows if r["state"] == k]
        print(f"\nState {k}  (target eigenvalue {basis.eigenvalues[k]:.5f})")
        print(f"{'it':>3} {'op':>3} {'F':>9} {'dF':>10} {'E(theta)':>10} {'dE':>10} "
              f"{'E reported':>11} {'contamination':>17} {'obj exact':>10}  note")
        for r in state_rows:
            df = "" if r["delta_fidelity"] is None else f"{r['delta_fidelity']:+.2e}"
            de = "" if r["delta_exact_energy"] is None else f"{r['delta_exact_energy']:+.2e}"
            rep = "" if r["reported_energy"] is None else f"{r['reported_energy']:.5f}"
            cont = ("" if r["main_contamination"] is None
                    else f"{r['main_contamination']} {r['main_contamination_weight']:.5f}")
            note = ""
            if r["delta_fidelity"] is not None:
                if r["delta_fidelity"] < -1e-6:
                    note = "F worse"
                elif r["delta_fidelity"] > 1e-6 and r["delta_exact_energy"] > 1e-9:
                    note = "F better, E up"
            print(f"{r['iteration']:>3} {r['operator']:>3} {r['fidelity']:>9.5f} {df:>10} "
                  f"{r['exact_energy_theta']:>10.5f} {de:>10} {rep:>11} {cont:>17} "
                  f"{r['exact_objective']:>10.5f}  {note}")
        fids = [r["fidelity"] for r in state_rows]
        if fids:
            best = int(np.argmax(fids))
            worse = sum(1 for r in state_rows if r["delta_fidelity"] is not None
                        and r["delta_fidelity"] < -1e-6)
            print(f"   final F = {fids[-1]:.5f}; best F = {fids[best]:.5f} at iteration "
                  f"{state_rows[best]['iteration']}; iterations where F got worse: {worse}")


def write_csv(path: str, rows_by_route: dict[str, list[dict]]) -> None:
    import pandas as pd
    flat = []
    for route, rows in rows_by_route.items():
        for r in rows:
            row = {"route": route, **{k: v for k, v in r.items() if k != "weights"}}
            row["exact_overlaps_with_previous"] = ";".join(f"{x:.6e}" for x in r["exact_overlaps_with_previous"])
            for j, w in enumerate(r["weights"]):
                row[f"w_phi_{j}"] = w
            flat.append(row)
    pd.DataFrame(flat).to_csv(path, index=False)
    print(f"\nCSV written: {path}")


def run(data: dict, ctx, *, routes: list[str], csv: str | None = None) -> dict[str, list[dict]]:
    """Analysis given an already-built context (main() builds the real one)."""
    p = data["problem"]
    basis = ExactBasis(ctx, int(p["n_valence_protons"]) + int(p["n_valence_neutrons"]))
    exact = np.asarray(ctx.hamiltonian.exact_energies, dtype=float)
    n = min(len(exact), basis.dimension)
    gap = float(np.max(np.abs(np.sort(basis.eigenvalues)[:n] - exact[:n]))) if n else 0.0
    print(f"Model space: d = {basis.dimension}; eigenvalues "
          + ", ".join(f"{e:.4f}" for e in basis.eigenvalues[:min(basis.dimension, 8)])
          + (" ..." if basis.dimension > 8 else ""))
    if gap > 1e-6:
        print(f"WARNING: the model-space spectrum differs from exact_energies by {gap:.2e}.")

    out: dict[str, list[dict]] = {}
    for route in routes:
        results = data.get(ROUTES[route]) or []
        if not results:
            continue
        try:
            out[route] = analyse_route(basis, results)
        except KeyError:
            print(f"\nRoute {route}: the history has no `ops`/`thetas` per iteration "
                  "(JSON written before 18/11/2026). Run again to use this script.")
            continue
        print_route(route, out[route], basis)
    if csv and out:
        write_csv(csv, out)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("json_path")
    parser.add_argument("--route", choices=["both", *ROUTES], default="both")
    parser.add_argument("--csv", default=None, help="optional CSV with one row per iteration")
    args = parser.parse_args()

    from vqd_nuclear.config import NumericalConfig, ProblemConfig, RunConfig
    from vqd_nuclear.vqd import build_context

    data = json.load(open(args.json_path, encoding="utf-8"))
    p = data["problem"]
    problem = ProblemConfig(shell=p["shell"], n_valence_protons=p["n_valence_protons"],
                            n_valence_neutrons=p["n_valence_neutrons"], total_j=p["total_j"])
    ctx = build_context(problem, RunConfig(n_states=int(data["run"]["n_states"]), n_workers=1),
                        NumericalConfig(n_jobs=1, use_model_space=True))
    routes = list(ROUTES) if args.route == "both" else [args.route]
    run(data, ctx, routes=routes, csv=args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
