#!/usr/bin/env python3
"""Contraste DENSO de la conmutacion entre operadores del pool.

check_degeneracy.py calcula [A,B] con algebra de Pauli, que es lo correcto para
nucleos grandes. Este script hace la misma pregunta con matrices densas, por un
camino completamente distinto, para descartar un fallo en esa algebra.

Motivo concreto: en ⁶Be los seis pares de operadores activos dieron
||[A,B]||_1 = 1.000e+00 EXACTAMENTE. Puede ser real (los coeficientes de
Jordan-Wigner son diadicos, asi que las sumas salen redondas), pero una
coincidencia tan limpia merece una comprobacion independiente.

Lo que importa es nulo frente a NO nulo, no el valor: las dos normas son
distintas (suma de modulos de coeficientes vs norma del maximo de la matriz).

Uso:
    python check_commutators_dense.py --shell p --protons 2 --neutrons 0 --J 0
"""
from __future__ import annotations

import argparse
import itertools
import sys

import numpy as np

_MATRIZ = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def densa(qubit_operator, n_qubits: int) -> np.ndarray:
    """Matriz densa de un operador de qubits.

    Convenio de bits: el qubit q pesa 1 << (n-1-q), o sea q=0 es el mas
    significativo. El mismo que apply_pauli_string_to_state.
    """
    dimension = 2 ** int(n_qubits)
    total = np.zeros((dimension, dimension), dtype=complex)
    for cadena, coeficiente in qubit_operator.terms.items():
        mapa = dict(cadena)
        bloque = np.array([[1.0 + 0j]])
        for q in range(int(n_qubits)):
            bloque = np.kron(bloque, _MATRIZ[mapa.get(q, "I")])
        total += complex(coeficiente) * bloque
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shell", choices=["p", "sd"], default="p")
    ap.add_argument("--protons", type=int, default=2)
    ap.add_argument("--neutrons", type=int, default=0)
    ap.add_argument("--J", type=int, default=0)
    ap.add_argument("--max-qubits", type=int, default=12,
                    help="por encima de esto las matrices densas no caben")
    args = ap.parse_args()

    from vqd_nuclear.config import ProblemConfig, RunConfig, NumericalConfig
    from vqd_nuclear.vqd import build_context

    problem = ProblemConfig(shell=args.shell, n_valence_protons=args.protons,
                            n_valence_neutrons=args.neutrons, total_j=args.J)
    ctx = build_context(problem, RunConfig(n_states=2, n_workers=1),
                        NumericalConfig(n_jobs=1))

    n = ctx.hamiltonian.n_qubits
    pool = ctx.hamiltonian.qubit_generator_pool
    activos = [int(c) for c in ctx.active_operator_indices]

    print(f"\n{args.shell}-shell  n_qubits={n}  activos: {activos}")

    if n > args.max_qubits:
        memoria = (2 ** n) ** 2 * 16 / 2 ** 20
        print(f"\n{n} qubits: cada matriz densa ocuparia {memoria:.0f} MB. Se omite.")
        print("Para nucleos de este tamano, fiate de check_degeneracy.py, que usa")
        print("algebra de Pauli y no construye matrices.")
        return 0

    matrices = {mu: densa(pool[mu], n) for mu in activos}

    # Control de sanidad del propio script: los generadores deben ser hermiticos
    # (coeficientes reales sobre cadenas de Pauli).
    for mu in activos:
        asimetria = float(np.max(np.abs(matrices[mu] - matrices[mu].conj().T)))
        if asimetria > 1e-12:
            print(f"  AVISO: el generador {mu} no es hermitico ({asimetria:.2e}). "
                  "Revisa el pool antes de interpretar nada.")

    print(f"\n{'par':>10} {'max|[A,B]|':>14} {'||[A,B]||_F':>14}  conmutan")
    conmutan_todos = True
    for a, b in itertools.combinations(activos, 2):
        conmutador = matrices[a] @ matrices[b] - matrices[b] @ matrices[a]
        maximo = float(np.max(np.abs(conmutador)))
        frobenius = float(np.linalg.norm(conmutador))
        conmuta = maximo < 1e-10
        conmutan_todos &= conmuta
        print(f"{f'({a},{b})':>10} {maximo:>14.3e} {frobenius:>14.3e}  "
              f"{'SI' if conmuta else 'no'}")

    print("\nLECTURA")
    print("  Compara la columna 'conmutan' con la de check_degeneracy.py. Deben")
    print("  coincidir par a par. Los VALORES no tienen por que coincidir: alli se")
    print("  suman modulos de coeficientes de Pauli y aqui se mide la matriz.")
    if conmutan_todos:
        print("  Todos conmutan: el orden del ansatz es irrelevante en este nucleo.")
    else:
        print("  No todos conmutan: el orden del ansatz produce parametrizaciones")
        print("  distintas. Eso NO implica estados finales distintos; si el")
        print("  autovalor no esta degenerado, ambas convergen al mismo autovector.")
        print("  Compruebalo con compare_runs.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
