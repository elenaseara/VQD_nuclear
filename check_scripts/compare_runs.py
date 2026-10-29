#!/usr/bin/env python3
"""Compara dos ejecuciones por el ESTADO FINAL, no por la lista de operadores.

Con operadores degenerados en el pool, dos caminos de calculo equivalentes
eligen ordenes distintos y la lista de `ops` deja de ser comparable. Lo que si
es comparable es el estado al que se converge.

Si el autovalor no esta degenerado, su autovector es unico salvo fase, asi que
dos ejecuciones que den la misma energia DEBEN dar el mismo estado. Este script
lo comprueba: |<psi_A|psi_B>|^2 debe salir 1.

La configuracion del problema se lee del propio JSON, asi que sirve para
cualquier par de ejecuciones del mismo nucleo.

Uso:
    python compare_runs.py sin_proyeccion.json con_proyeccion.json
"""
from __future__ import annotations

import argparse
import json
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("json_a")
    ap.add_argument("json_b")
    ap.add_argument("--route", default="statevector_results",
                    choices=["statevector_results", "full_quantum_circuit_results"])
    ap.add_argument("--tol", type=float, default=1e-9)
    args = ap.parse_args()

    with open(args.json_a, encoding="utf-8") as f:
        a = json.load(f)
    with open(args.json_b, encoding="utf-8") as f:
        b = json.load(f)

    if a["problem"] != b["problem"]:
        print("Las dos ejecuciones NO son del mismo problema:")
        print(f"  A: {a['problem']}")
        print(f"  B: {b['problem']}")
        return 1

    from vqd_nuclear.config import ProblemConfig, RunConfig, NumericalConfig
    from vqd_nuclear.vqd import build_context

    problem = ProblemConfig(**a["problem"])
    resultados_a = a[args.route]
    resultados_b = b[args.route]
    if not resultados_a or not resultados_b:
        print(f"Alguno de los JSON no tiene resultados en '{args.route}'.")
        return 1

    n_states = max(len(resultados_a), len(resultados_b))
    ctx = build_context(problem, RunConfig(n_states=n_states, n_workers=1),
                        NumericalConfig(n_jobs=1))
    eng = ctx.measurements

    print(f"\n{problem.shell}-shell  Z_val={problem.n_valence_protons} "
          f"N_val={problem.n_valence_neutrons} J={problem.total_j}")
    print(f"A: {args.json_a}")
    print(f"B: {args.json_b}\n")

    print(f"{'estado':>7} {'|<A|B>|^2':>14} {'dif energia':>14} "
          f"{'ops A':>16} {'ops B':>16}  {'mismo conjunto':>14}")

    fallos = 0
    for ra, rb in zip(resultados_a, resultados_b):
        solape = eng.exact_overlap(ra["thetas"], ra["ops"], rb["thetas"], rb["ops"])
        dif_e = abs(float(ra["energy"]) - float(rb["energy"]))
        mismo_conjunto = set(ra["ops"]) == set(rb["ops"])
        mismo_orden = list(ra["ops"]) == list(rb["ops"])

        ok = abs(solape - 1.0) < args.tol and dif_e < 1e-10
        fallos += 0 if ok else 1
        print(f"{ra['state_index']:>7} {solape:>14.12f} {dif_e:>14.2e} "
              f"{str(ra['ops']):>16} {str(rb['ops']):>16}  "
              f"{('SI' if mismo_conjunto else 'NO'):>14}"
              f"{'' if mismo_orden else '   (orden distinto)'}")

    print()
    if fallos == 0:
        print("  [OK] Mismo estado final en todos los estados.")
        print("  Si el orden de los operadores difiere, es cosmetico: las dos")
        print("  trayectorias ADAPT llegan al mismo punto por caminos distintos.")
    else:
        print(f"  [FALLO] {fallos} estado(s) no coinciden.")
        print("  Si el solapamiento es ~1 pero la energia difiere, mira la")
        print("  precision del optimizador. Si el solapamiento NO es ~1 con la")
        print("  misma energia, hay degeneracion REAL en el espectro y los dos")
        print("  estados son fisicamente distintos: eso afecta a VQD, porque los")
        print("  estados siguientes se ortogonalizan contra uno u otro.")
    return 0 if fallos == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
