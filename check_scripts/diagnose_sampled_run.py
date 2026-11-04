#!/usr/bin/env python3
"""Reevalua SIN RUIDO una ejecucion full_quantum_circuit.

Toma los thetas/ops finales de cada estado del JSON y calcula con statevector:
energia exacta, solapamientos exactos con los estados previos (los de la propia
ejecucion muestreada) y objetivo VQD exacto. Separa dos causas de error:

  - ruido de estimacion: el estado es bueno, pero la energia reportada no;
  - optimizacion: el estado mismo es malo (objetivo exacto alto).

Ademas imprime los gradientes exactos en la referencia, para comprobar que el
primer operador elegido con shots es el que elegiria statevector.

Uso:
    python diagnose_sampled_run.py raw_results_full_quantum_circuit.json
"""
from __future__ import annotations

import json
import sys

import numpy as np

from vqd_nuclear.config import NumericalConfig, ProblemConfig, RunConfig
from vqd_nuclear.vqd import build_context


def main(path: str) -> int:
    data = json.load(open(path, encoding="utf-8"))
    p = data["problem"]
    problem = ProblemConfig(shell=p["shell"], n_valence_protons=p["n_valence_protons"],
                            n_valence_neutrons=p["n_valence_neutrons"], total_j=p["total_j"])
    n_states = int(data["run"]["n_states"])
    ctx = build_context(problem, RunConfig(n_states=n_states, n_workers=1), NumericalConfig(n_jobs=1))
    eng = ctx.measurements
    beta = float(data["run"]["beta"])
    resultados = data["full_quantum_circuit_results"]
    exactas = ctx.hamiltonian.exact_energies

    print("\nGradientes exactos en la referencia (estado 0, iteracion 0):")
    ref = eng.ansatz_builder.state([], [])
    for mu in ctx.active_operator_indices:
        g = eng.statevector_energy_gradient([], [], ctx.hamiltonian.qubit_generator_pool[int(mu)], psi=ref)
        print(f"   op {int(mu):>2}: {g:+.6f}")
    print(f"   muestreado eligio op {resultados[0]['history'][0]['selected_operator']} "
          f"con g = {resultados[0]['history'][0]['adapt_gradient']:+.4f}")

    print(f"\n{'estado':>6} {'E reportada':>12} {'E exacta(theta)':>16} {'E objetivo':>11} "
          f"{'solapes exactos':>22} {'obj. exacto':>12} {'obj. ideal':>11}")
    for k, r in enumerate(resultados):
        th, ops = r["thetas"], r["ops"]
        e_exacta = eng.statevector_energy(th, ops)
        solapes = [eng.exact_overlap(prev["thetas"], prev["ops"], th, ops) for prev in resultados[:k]]
        obj = e_exacta + beta * sum(solapes)
        print(f"{k:>6} {r['energy']:>12.5f} {e_exacta:>16.5f} {float(exactas[k]):>11.5f} "
              f"{str([round(s, 5) for s in solapes]):>22} {obj:>12.5f} {float(exactas[k]):>11.5f}")
    print("\n'obj. ideal' = autovalor exacto. OJO: solo es una cota valida si los")
    print("estados PREVIOS son autoestados exactos. Si un estado previo es malo, el")
    print("siguiente puede bajar de esa cota (se cuela en el autoestado que el previo")
    print("no supo representar). La tabla de fidelidades de abajo no tiene ese problema.")

    # ---------------------------------------------------------------------
    # Fidelidad con los autovectores EXACTOS: no depende de los estados previos.
    # ---------------------------------------------------------------------
    from vqd_nuclear.ansatz import get_hamiltonian_basis_references
    from vqd_nuclear.measurements import apply_qubit_operator_to_state

    nq = ctx.hamiltonian.n_qubits
    refs = get_hamiltonian_basis_references(
        ctx.hamiltonian.nucleus, n_qubits=nq,
        n_valence_particles=p["n_valence_protons"] + p["n_valence_neutrons"])
    indices = sorted({sum(1 << (nq - 1 - int(q)) for q in occ) for occ in refs})
    d = len(indices)
    HS = np.zeros((d, d), dtype=complex)
    for b, ib in enumerate(indices):
        v = np.zeros(2 ** nq, dtype=complex); v[ib] = 1.0
        HS[:, b] = apply_qubit_operator_to_state(ctx.hamiltonian.qubit_hamiltonian, v, nq)[indices]
    autoval, autovec = np.linalg.eigh(HS)

    print(f"\nPeso de cada estado VQD sobre los autovectores exactos |<phi_j|psi_k>|^2"
          f" (espacio del modelo, d = {d}):")
    print(f"{'estado':>6} " + " ".join(f"{'phi_' + str(j):>9}" for j in range(d)))
    print(f"{'':>6} " + " ".join(f"{e:>9.3f}" for e in autoval))
    for k, r in enumerate(resultados):
        psi = eng.ansatz_builder.state(r["thetas"], r["ops"])
        pesos = np.abs(autovec.conj().T @ psi[indices]) ** 2
        print(f"{k:>6} " + " ".join(f"{w:>9.5f}" for w in pesos)
              + f"    fidelidad con phi_{k}: {pesos[k]:.5f}")
    print("\nUn estado VQD perfecto tiene peso 1 en su phi_k y 0 en el resto.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
