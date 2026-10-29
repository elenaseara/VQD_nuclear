#!/usr/bin/env python3
"""Degeneracion y conmutacion entre los operadores activos del pool.

Responde a dos preguntas que surgieron al comparar la ruta completa con la
proyectada en ⁶Be, donde los operadores 2 y 4 se intercambian entre ejecuciones:

  1. ¿Estan DEGENERADOS? Si su gradiente coincide hasta el ruido de coma
     flotante, el ganador de max() lo decide el ultimo digito y la trayectoria
     ADAPT depende del camino aritmetico. Eso es inofensivo para la energia pero
     hace inutil comparar la lista ordenada de operadores entre ejecuciones.

  2. ¿CONMUTAN? Si [T_a, T_b] = 0 el orden en que se apliquen es irrelevante y
     los dos ansatze son literalmente el mismo estado, no solo la misma energia.
     Si no conmutan, el orden importa aunque la energia final coincida.

El conmutador se calcula con algebra de Pauli, no con matrices densas, para que
tambien funcione en sd (2^12 = 4096 haria 134 MB por matriz).

Uso:
    python check_degeneracy.py --shell p --protons 2 --neutrons 0 --J 0
"""
from __future__ import annotations

import argparse
import itertools
import sys

import numpy as np

# Mismo algebra que pauli_algebra.py. Se repite aqui para que el diagnostico
# funcione aunque las fases A y B no esten integradas todavia en el paquete.
_PRODUCTO_1Q = {
    ("X", "X"): (1 + 0j, "I"), ("Y", "Y"): (1 + 0j, "I"), ("Z", "Z"): (1 + 0j, "I"),
    ("X", "Y"): (1j, "Z"), ("Y", "X"): (-1j, "Z"),
    ("Y", "Z"): (1j, "X"), ("Z", "Y"): (-1j, "X"),
    ("Z", "X"): (1j, "Y"), ("X", "Z"): (-1j, "Y"),
}


def pauli_commute(p: tuple, q: tuple) -> bool:
    mapa_q = dict(q)
    n = 0
    for qubit, letra_p in p:
        letra_q = mapa_q.get(qubit)
        if letra_q is not None and letra_q != letra_p:
            n += 1
    return n % 2 == 0


def multiply_pauli_strings(p: tuple, q: tuple) -> tuple[complex, tuple]:
    mapa_p, mapa_q = dict(p), dict(q)
    fase: complex = 1 + 0j
    resultado: dict[int, str] = {}
    for qubit in set(mapa_p) | set(mapa_q):
        a, b = mapa_p.get(qubit), mapa_q.get(qubit)
        if a is None:
            resultado[qubit] = b
        elif b is None:
            resultado[qubit] = a
        else:
            fase_local, letra = _PRODUCTO_1Q[(a, b)]
            fase *= fase_local
            if letra != "I":
                resultado[qubit] = letra
    return fase, tuple(sorted(resultado.items()))


def commutator_norm(op_a, op_b, tolerance: float = 1e-12) -> float:
    """Suma de |coeficientes| de [A,B]. Cero exacto => conmutan."""
    acumulado: dict[tuple, complex] = {}
    for cadena_a, coef_a in op_a.terms.items():
        for cadena_b, coef_b in op_b.terms.items():
            if pauli_commute(cadena_a, cadena_b):
                continue                      # [P,Q] = 0
            fase, cadena_r = multiply_pauli_strings(cadena_a, cadena_b)
            # [P,Q] = 2 P Q cuando anticonmutan
            contribucion = 2.0 * complex(coef_a) * complex(coef_b) * fase
            acumulado[cadena_r] = acumulado.get(cadena_r, 0j) + contribucion
    return float(sum(abs(c) for c in acumulado.values() if abs(c) > tolerance))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shell", choices=["p", "sd"], default="p")
    ap.add_argument("--protons", type=int, default=2)
    ap.add_argument("--neutrons", type=int, default=0)
    ap.add_argument("--J", type=int, default=0)
    ap.add_argument("--rtol", type=float, default=1e-6,
                    help="tolerancia relativa para considerar dos gradientes degenerados")
    args = ap.parse_args()

    from vqd_nuclear.config import ProblemConfig, RunConfig, NumericalConfig
    from vqd_nuclear.vqd import build_context
    from vqd_nuclear import measurements as M

    problem = ProblemConfig(shell=args.shell, n_valence_protons=args.protons,
                            n_valence_neutrons=args.neutrons, total_j=args.J)
    ctx = build_context(problem, RunConfig(n_states=2, n_workers=1),
                        NumericalConfig(n_jobs=1))
    eng = ctx.measurements
    n = ctx.hamiltonian.n_qubits
    pool = ctx.hamiltonian.qubit_generator_pool
    activos = [int(c) for c in ctx.active_operator_indices]

    print(f"\n{args.shell}-shell  Z_val={args.protons} N_val={args.neutrons} J={args.J}")
    print(f"n_qubits={n}   operadores activos: {activos}")

    # =================================================================
    print("\n1. GRADIENTES SOBRE EL ESTADO DE REFERENCIA")
    # =================================================================
    psi = eng.ansatz_builder.state([], [])
    h_psi = M.apply_qubit_operator_to_state(ctx.hamiltonian.qubit_hamiltonian, psi, n)

    gradientes = {}
    for mu in activos:
        t_psi = M.apply_qubit_operator_to_state(pool[mu], psi, n)
        gradientes[mu] = M.commutator_energy_gradient(h_psi, t_psi)

    for mu in activos:
        print(f"   op {mu:>3}:  g = {gradientes[mu]:+.15f}   "
              f"terminos del generador: {len(pool[mu].terms)}")

    # --- grupos degenerados -------------------------------------------
    grupos: list[list[int]] = []
    for mu in activos:
        for grupo in grupos:
            referencia = abs(gradientes[grupo[0]])
            if abs(abs(gradientes[mu]) - referencia) <= args.rtol * max(referencia, 1e-30):
                grupo.append(mu)
                break
        else:
            grupos.append([mu])

    degenerados = [g for g in grupos if len(g) > 1]
    print(f"\n   grupos degenerados (|g| igual a {args.rtol:.0e} relativo): "
          f"{degenerados if degenerados else 'ninguno'}")
    for grupo in degenerados:
        valores = [abs(gradientes[mu]) for mu in grupo]
        dispersion = max(valores) - min(valores)
        print(f"     {grupo}: |g| = {valores[0]:.12f}, dispersion = {dispersion:.3e}")
        if dispersion <= 0.0:
            print("        -> gradientes IDENTICOS bit a bit. max() devuelve el "
                  "PRIMER maximo, asi que elige el orden de la lista de candidatos.")
        else:
            digito = int(-np.log10(dispersion / valores[0])) + 1
            print(f"        -> max() elige segun el digito {digito} aprox. "
                  "La eleccion NO es robusta.")

    # =================================================================
    print("\n2. CONMUTACION ENTRE OPERADORES ACTIVOS")
    # =================================================================
    print(f"   {'par':>10} {'||[A,B]||_1':>14}  conmutan")
    conmutan_todos = True
    pares_degenerados_conmutan = True
    for a, b in itertools.combinations(activos, 2):
        norma = commutator_norm(pool[a], pool[b])
        conmuta = norma < 1e-10
        conmutan_todos &= conmuta
        marca = "SI" if conmuta else "no"
        en_grupo = any(a in g and b in g for g in degenerados)
        if en_grupo and not conmuta:
            pares_degenerados_conmutan = False
        print(f"   {f'({a},{b})':>10} {norma:>14.3e}  {marca}"
              f"{'   <- par degenerado' if en_grupo else ''}")

    # =================================================================
    print("\n3. LECTURA")
    # =================================================================
    if not degenerados:
        print("   No hay degeneracion: el orden de seleccion ADAPT es estable y")
        print("   se puede comparar la lista ordenada de operadores entre ejecuciones.")
    else:
        print("   HAY degeneracion. Consecuencias:")
        print("   - La lista ORDENADA de operadores no es comparable entre caminos")
        print("     de calculo equivalentes. El test de regresion debe comparar el")
        print("     CONJUNTO de operadores, las energias y el numero de capas.")
        if pares_degenerados_conmutan:
            print("   - Los pares degenerados CONMUTAN, asi que el orden entre ellos es")
            print("     irrelevante: los dos ansatze son el MISMO estado, no solo la")
            print("     misma energia. La diferencia observada es puramente cosmetica.")
        else:
            print("   - Algun par degenerado NO conmuta: el orden produce estados")
            print("     distintos aunque la energia final coincida. Merece mirarse si")
            print("     afecta a los solapamientos de VQD.")
        print("   - Conviene un desempate determinista por indice en")
        print("     select_next_operator para que las ejecuciones sean reproducibles.")

    if conmutan_todos:
        print("\n   Ademas TODOS los operadores activos conmutan entre si, asi que el")
        print("   orden del ansatz completo es irrelevante en este nucleo.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
