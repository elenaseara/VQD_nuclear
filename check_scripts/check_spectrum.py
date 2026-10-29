#!/usr/bin/env python3
"""Degeneracion y huecos del espectro, ANTES de lanzar un calculo VQD.

Responde a tres preguntas que conviene tener contestadas antes de confiar en
los resultados de un nucleo nuevo:

  1. ¿Hay niveles DEGENERADOS entre los que se van a calcular?
     Si los hay, VQD devuelve UNO de los estados del multiplete, el que le
     salga, y luego se ortogonaliza contra ese. El siguiente estado puede ser su
     companero degenerado: la tabla final tendria dos niveles con la misma
     energia y no se sabria si eso es el espectro o un fallo de convergencia.
     Ademas, dos ordenes de operadores podrian converger a estados fisicamente
     distintos con la misma energia (en ⁶Be se comprobo que NO pasa, pero esa
     conclusion se apoyaba en que su espectro no esta degenerado).

  2. ¿Hay niveles CASI degenerados? No rompen nada formalmente, pero son el caso
     practico dificil: la penalizacion tiene que separarlos y el optimizador
     sufre.

  3. ¿Que beta minima hace falta? beta_i > E_k - E_i para todo k que se quiera
     separar. El codigo usa 2*max_gap cuando se pide 'auto'.

Lo que NO depende de esto: la proyeccion sobre el espacio del modelo
(model_space.py) es valida en cualquier nucleo. Su hipotesis es que el ansatz no
sale del espacio del modelo, no que el espectro sea no degenerado, y se mide en
cada evaluacion. Lo peor que puede pasar ahi es que no acelere.

Uso:
    python check_spectrum.py --shell p --protons 1 --neutrons 1 --J 0   # ⁶Li
    python check_spectrum.py --shell p --protons 1 --neutrons 2 --J 0   # ⁷Li
    python check_spectrum.py --shell p --protons 2 --neutrons 0 --J 0 --n-states 3
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shell", choices=["p", "sd"], default="p")
    ap.add_argument("--protons", type=int, default=1)
    ap.add_argument("--neutrons", type=int, default=1)
    ap.add_argument("--J", type=int, default=0)
    ap.add_argument("--n-states", type=int, default=3,
                    help="cuantos niveles se piensan calcular con VQD")
    ap.add_argument("--tol", type=float, default=1e-9,
                    help="por debajo de esto, dos niveles son DEGENERADOS (MeV)")
    ap.add_argument("--near-tol", type=float, default=1e-2,
                    help="por debajo de esto, CASI degenerados (MeV)")
    args = ap.parse_args()

    from vqd_nuclear.config import ProblemConfig, RunConfig, NumericalConfig
    from vqd_nuclear.vqd import build_context

    problem = ProblemConfig(shell=args.shell, n_valence_protons=args.protons,
                            n_valence_neutrons=args.neutrons, total_j=args.J)
    ctx = build_context(problem, RunConfig(n_states=args.n_states, n_workers=1),
                        NumericalConfig(n_jobs=1))

    matriz = ctx.hamiltonian.matrix
    densa = matriz.toarray() if hasattr(matriz, "toarray") else np.asarray(matriz)
    energias = np.sort(np.linalg.eigvalsh(np.asarray(densa, dtype=complex)).real)
    d = len(energias)

    print(f"\n{args.shell}-shell  Z_val={args.protons}  N_val={args.neutrons}  "
          f"J={args.J}")
    print(f"n_qubits = {ctx.hamiltonian.n_qubits}   dimension del modelo = {d}   "
          f"operadores activos = {len(ctx.active_operator_indices)}")

    # Techo de capas de ADAPT con la politica actual de consumir operadores.
    print(f"capas maximas de ADAPT = min(max_adapt_iter, {len(ctx.active_operator_indices)})"
          "   <- los operadores se consumen")
    if args.n_states > d:
        print(f"\n  AVISO: se piden {args.n_states} estados pero el espacio solo tiene "
              f"{d}. Los sobrantes no existen.")

    # =================================================================
    print("\n1. ESPECTRO COMPLETO")
    # =================================================================
    print(f"   {'i':>3} {'E_i (MeV)':>18} {'hueco con E_{i-1}':>20}")
    for i, e in enumerate(energias):
        hueco = "" if i == 0 else f"{energias[i] - energias[i-1]:>20.12f}"
        marca = "  <- solicitado" if i < args.n_states else ""
        print(f"   {i:>3} {e:>18.12f} {hueco}{marca}")

    # =================================================================
    print("\n2. DEGENERACION")
    # =================================================================
    huecos = np.diff(energias)
    grupos: list[list[int]] = [[0]]
    for i, hueco in enumerate(huecos, start=1):
        if hueco <= args.tol:
            grupos[-1].append(i)
        else:
            grupos.append([i])

    multipletes = [g for g in grupos if len(g) > 1]
    if not multipletes:
        print(f"   No hay niveles degenerados (tolerancia {args.tol:.0e} MeV).")
        print(f"   Hueco minimo del espectro: {float(np.min(huecos)):.12f} MeV")
    else:
        print(f"   HAY degeneracion (tolerancia {args.tol:.0e} MeV):")
        for g in multipletes:
            print(f"     niveles {g}: E = {energias[g[0]]:.12f} MeV "
                  f"(multiplicidad {len(g)})")

    afectados = [g for g in multipletes if any(i < args.n_states for i in g)]

    # --- casi degenerados ---------------------------------------------
    cercanos = [(i, i + 1, float(h)) for i, h in enumerate(huecos)
                if args.tol < h <= args.near_tol]
    if cercanos:
        print(f"\n   Casi degenerados (hueco <= {args.near_tol:g} MeV):")
        for i, j, h in cercanos:
            dentro = "  <- entre los solicitados" if j < args.n_states else ""
            print(f"     {i} y {j}: hueco = {h:.12f} MeV{dentro}")

    # =================================================================
    print("\n3. BETA")
    # =================================================================
    n = min(args.n_states, d)
    gaps = np.abs(energias[:n] - energias[0])
    minimo = float(np.max(gaps))
    print(f"   minimo teorico : {minimo:.9f} MeV   (beta_i > E_k - E_i)")
    print(f"   'auto' del codigo (2*max_gap) : {2.0 * minimo:.9f} MeV")

    # =================================================================
    print("\n4. VEREDICTO")
    # =================================================================
    if afectados:
        print("   CUIDADO. Hay degeneracion ENTRE LOS NIVELES SOLICITADOS:")
        for g in afectados:
            print(f"     niveles {g} comparten E = {energias[g[0]]:.12f} MeV")
        print()
        print("   Consecuencias, por orden de importancia:")
        print("   a) VQD devolvera UN estado del multiplete, el que salga, y se")
        print("      ortogonalizara contra ese. El siguiente nivel que encuentre")
        print("      puede ser su companero, con la misma energia. Dos filas con")
        print("      la misma energia en la tabla final no distinguen entre")
        print("      'asi es el espectro' y 'no ha convergido'.")
        print("   b) La conclusion de ⁶Be sobre que el orden de operadores es")
        print("      cosmetico NO se traslada: aqui dos ordenes pueden converger")
        print("      a estados distintos con la misma energia. Usa compare_runs.py")
        print("      para detectarlo (solapamiento lejos de 1, energias iguales).")
        print("   c) Conviene mirar el contenido en J de cada nivel antes de")
        print("      comparar con valores publicados.")
    elif multipletes:
        print("   Hay degeneracion, pero FUERA de los niveles solicitados.")
        print(f"   Sin problema mientras n_states <= {min(i for g in multipletes for i in g)}.")
    else:
        print("   Espectro no degenerado en el rango solicitado.")
        print("   Un autovalor determina su autovector salvo fase, asi que dos")
        print("   ejecuciones con la misma energia dan el mismo estado. Se puede")
        print("   comparar por energia con confianza.")

    if cercanos and any(j < args.n_states for _, j, _ in cercanos):
        print("\n   Ademas hay niveles CASI degenerados entre los solicitados. No")
        print("   rompen nada formalmente, pero son el caso practico dificil:")
        print("   la penalizacion tiene que separarlos y el optimizador sufre.")
        print("   Vigila optimizer_nfev y el solapamiento final en esos estados.")

    print("\n   (La proyeccion sobre el espacio del modelo NO depende de nada de")
    print("    esto: su hipotesis es que el ansatz no sale del subespacio, y se")
    print("    mide en cada evaluacion.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
