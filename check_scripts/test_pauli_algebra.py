#!/usr/bin/env python3
"""Tests de las fases A y B de la ruta muestreada.

Los bloques 1-4 son autocontenidos: comparan contra matrices densas construidas
a mano y no necesitan el proyecto. El bloque 5 es el que de verdad importa y si
necesita el proyecto: comprueba que el gradiente medido como <psi|G_mu|psi>
coincide EN VALOR Y EN SIGNO con commutator_energy_gradient.

Uso:
    python test_pauli_algebra.py              # bloques 1-4
    python test_pauli_algebra.py --con-motor  # ademas el 5 (necesita vqd_nuclear)
"""
from __future__ import annotations

import argparse
import itertools
import sys

import numpy as np

from vqd_nuclear.conventions import ANSATZ_DERIVATIVE_PHASE
from vqd_nuclear.pauli_algebra import (PauliSum, build_gradient_operator,
                                       multiply_pauli_strings, pauli_commute)

_MATRIZ = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}

_fallos = 0


def marca(ok: bool, texto: str) -> None:
    global _fallos
    if not ok:
        _fallos += 1
    print(f"  {'[OK]  ' if ok else '[FALLO]'} {texto}")


def densa(cadena: tuple, n: int) -> np.ndarray:
    """Matriz densa de una cadena de Pauli.

    Convenio de bits: el qubit q pesa 1 << (n-1-q), es decir el qubit 0 es el
    mas significativo. Es el mismo que apply_pauli_string_to_state.
    """
    mapa = dict(cadena)
    salida = np.array([[1.0 + 0j]])
    for q in range(n):
        salida = np.kron(salida, _MATRIZ[mapa.get(q, "I")])
    return salida


def todas_las_cadenas(n: int):
    for letras in itertools.product("IXYZ", repeat=n):
        yield tuple(sorted((q, p) for q, p in enumerate(letras) if p != "I"))


def qwc(p: tuple, q: tuple) -> bool:
    """QWC de referencia, para contrastar con pauli_commute."""
    mapa_q = dict(q)
    for qubit, letra_p in p:
        letra_q = mapa_q.get(qubit)
        if letra_q is not None and letra_q != letra_p:
            return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--con-motor", action="store_true")
    ap.add_argument("--shell", default="p")
    ap.add_argument("--protons", type=int, default=2)
    ap.add_argument("--neutrons", type=int, default=0)
    ap.add_argument("--J", type=int, default=0)
    args = ap.parse_args()

    n = 3
    cadenas = list(todas_las_cadenas(n))

    # =================================================================
    print("\n1. multiply_pauli_strings contra matrices densas "
          f"({len(cadenas)**2} pares a {n} qubits)")
    # =================================================================
    peor = 0.0
    for p in cadenas:
        for q in cadenas:
            fase, r = multiply_pauli_strings(p, q)
            peor = max(peor, float(np.max(np.abs(
                densa(p, n) @ densa(q, n) - fase * densa(r, n)))))
    marca(peor < 1e-12, f"P Q = fase * R   (error maximo {peor:.2e})")

    # =================================================================
    print("\n2. pauli_commute contra el conmutador denso")
    # =================================================================
    discrepancias = 0
    for p in cadenas:
        for q in cadenas:
            mp, mq = densa(p, n), densa(q, n)
            conmuta_real = bool(np.max(np.abs(mp @ mq - mq @ mp)) < 1e-12)
            if pauli_commute(p, q) != conmuta_real:
                discrepancias += 1
    marca(discrepancias == 0, f"coincide en todos los pares ({discrepancias} fallos)")

    # Dos cadenas de Pauli o conmutan o anticonmutan: nunca hay un caso mixto.
    mixtos = 0
    for p in cadenas:
        for q in cadenas:
            mp, mq = densa(p, n), densa(q, n)
            if not (np.max(np.abs(mp @ mq - mq @ mp)) < 1e-12
                    or np.max(np.abs(mp @ mq + mq @ mp)) < 1e-12):
                mixtos += 1
    marca(mixtos == 0, "todo par conmuta o anticonmuta, nunca otra cosa")

    # =================================================================
    print("\n3. pauli_commute NO es qwc_commutes")
    # =================================================================
    total = distintos = 0
    for p in cadenas:
        for q in cadenas:
            total += 1
            if pauli_commute(p, q) != qwc(p, q):
                distintos += 1
    print(f"   diferen en {distintos}/{total} pares ({100*distintos/total:.1f}%)")
    marca(distintos > 0, "son criterios distintos (si esto falla, una llama a la otra)")
    # QWC es mas fuerte: si P y Q son QWC, necesariamente conmutan.
    implicacion = all(pauli_commute(p, q) for p in cadenas for q in cadenas if qwc(p, q))
    marca(implicacion, "QWC implica conmutacion (pero no al reves)")

    # =================================================================
    print("\n4. build_gradient_operator contra PHASE * [H,A] denso")
    # =================================================================
    rng = np.random.default_rng(20261020)
    peor = 0.0
    vacios = 0
    for _ in range(200):
        cuerpo = [c for c in cadenas if c != ()]
        idx_h = rng.choice(len(cuerpo), size=4, replace=False)
        idx_a = rng.choice(len(cuerpo), size=3, replace=False)
        h = PauliSum({cuerpo[int(i)]: complex(rng.normal(), 0.0) for i in idx_h})
        a = PauliSum({cuerpo[int(i)]: complex(rng.normal(), 0.0) for i in idx_a})

        g = build_gradient_operator(h, a)
        if not g:
            vacios += 1

        mh = sum(c * densa(s, n) for s, c in h.terms.items())
        ma = sum(c * densa(s, n) for s, c in a.terms.items())
        esperado = ANSATZ_DERIVATIVE_PHASE * (mh @ ma - ma @ mh)
        obtenido = (sum(c * densa(s, n) for s, c in g.terms.items())
                    if g else np.zeros((2 ** n, 2 ** n), dtype=complex))
        peor = max(peor, float(np.max(np.abs(esperado - obtenido))))

    marca(peor < 1e-10, f"G_mu reconstruye el conmutador (error maximo {peor:.2e})")
    print(f"   ({vacios}/200 dieron G_mu vacio, es decir [H,A] = 0)")

    # Hermiticidad: coeficientes reales.
    h = PauliSum({((0, "X"), (1, "Y")): 0.7 + 0j, ((1, "Z"),): -0.3 + 0j})
    a = PauliSum({((0, "Y"),): 0.5 + 0j, ((0, "Z"), (2, "X")): 0.2 + 0j})
    g = build_gradient_operator(h, a)
    marca(all(abs(c.imag) < 1e-12 for c in g.terms.values()),
          "los coeficientes de G_mu son reales (G_mu hermitico)")

    # =================================================================
    if args.con_motor:
        print("\n5. SIGNO: <psi|G_mu|psi> contra commutator_energy_gradient")
        # =============================================================
        from vqd_nuclear.config import ProblemConfig, RunConfig, NumericalConfig
        from vqd_nuclear.vqd import build_context
        from vqd_nuclear import measurements as M

        problem = ProblemConfig(shell=args.shell, n_valence_protons=args.protons,
                                n_valence_neutrons=args.neutrons, total_j=args.J)
        ctx = build_context(problem, RunConfig(n_states=2, n_workers=1),
                            NumericalConfig(n_jobs=1))
        eng = ctx.measurements
        nq = ctx.hamiltonian.n_qubits

        thetas = [0.31, -0.17]
        ops = [int(i) for i in ctx.active_operator_indices[:2]]
        psi = eng.ansatz_builder.state(thetas, ops)
        h_psi = M.apply_qubit_operator_to_state(ctx.hamiltonian.qubit_hamiltonian, psi, nq)

        peor = 0.0
        print(f"   {'op':>4} {'via G_mu':>16} {'via conmutador':>16} {'dif':>11}")
        for mu in ctx.active_operator_indices:
            generator = ctx.hamiltonian.qubit_generator_pool[int(mu)]
            t_psi = M.apply_qubit_operator_to_state(generator, psi, nq)
            por_conmutador = M.commutator_energy_gradient(h_psi, t_psi)

            g_mu = build_gradient_operator(ctx.hamiltonian.qubit_hamiltonian, generator)
            if not g_mu:
                print(f"   {mu:>4}   G_mu vacio: [H,A] = 0   "
                      f"(conmutador da {por_conmutador:+.3e})")
                marca(abs(por_conmutador) < 1e-10,
                      f"op {mu}: G_mu vacio implica gradiente de energia nulo")
                continue

            g_psi = M.apply_qubit_operator_to_state(g_mu, psi, nq)
            por_observable = float(np.real(np.vdot(psi, g_psi)))

            dif = abs(por_observable - por_conmutador)
            peor = max(peor, dif)
            print(f"   {mu:>4} {por_observable:>16.9f} {por_conmutador:>16.9f} {dif:>11.2e}")

        marca(peor < 1e-9, f"mismo valor Y mismo signo (dif maxima {peor:.2e})")
        if peor > 1e-9:
            print("   Si la diferencia es exactamente el doble del valor, el signo")
            print("   esta invertido: build_gradient_operator sigue la seccion 10 del")
            print("   pseudocodigo en lugar de la 4.2 (la verificada).")
    else:
        print("\n5. (omitido; usa --con-motor para la comprobacion de signo)")

    print("\n" + "=" * 60)
    print("  Todo pasa." if _fallos == 0 else f"  {_fallos} comprobacion(es) fallan.")
    return 0 if _fallos == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
