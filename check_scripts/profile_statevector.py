#!/usr/bin/env python3
"""Perfilado REAL de la ruta statevector: donde se va el tiempo.

No proyecta nada ni estima: ejecuta el calculo de verdad bajo cProfile y
reparte el tiempo medido. Sirve para decidir que optimizar antes de tocar nada.

Separa tres bloques:
    1. build_context        (una vez por ejecucion)
    2. barrido de gradientes ADAPT   (una vez por capa)
    3. evaluaciones de COBYLA        (cientos por capa)

Uso:
    python profile_statevector.py --shell p  --protons 2 --neutrons 0 --n-states 3
    python profile_statevector.py --shell sd --protons 2 --neutrons 0 --n-states 2
"""
from __future__ import annotations

import argparse
import cProfile
import io as _io
import pstats
import time


def fmt(segundos: float) -> str:
    if segundos < 1e-3:
        return f"{segundos * 1e6:.0f} us"
    if segundos < 1.0:
        return f"{segundos * 1e3:.1f} ms"
    if segundos < 60:
        return f"{segundos:.2f} s"
    return f"{segundos / 60:.2f} min"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shell", choices=["p", "sd"], default="p")
    ap.add_argument("--protons", type=int, default=2)
    ap.add_argument("--neutrons", type=int, default=0)
    ap.add_argument("--J", type=int, default=0)
    ap.add_argument("--n-states", type=int, default=2)
    ap.add_argument("--max-adapt-iter", type=int, default=20)
    ap.add_argument("--optimizer-maxiter", type=int, default=2000)
    ap.add_argument("--cobyla-rhoend", type=float, default=1e-8)
    ap.add_argument("--cobyla-rhobeg", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--top", type=int, default=25, help="funciones a listar")
    ap.add_argument("--beta", default="auto",
                    help="valor absoluto, o 'auto' para 2*max_gap")
    args = ap.parse_args()

    from vqd_nuclear.config import (ProblemConfig, RunConfig, NumericalConfig,
                                    dataclass_to_dict)
    from vqd_nuclear.vqd import build_context, run_vqd_family
    from vqd_nuclear import measurements as M

    problem = ProblemConfig(shell=args.shell, n_valence_protons=args.protons,
                            n_valence_neutrons=args.neutrons, total_j=args.J)
    run = RunConfig(n_states=args.n_states, n_workers=1, seed=args.seed,
                    run_statevector=True, run_full_quantum_circuit=False)
    numerical = NumericalConfig(max_adapt_iter=args.max_adapt_iter,
                                optimizer_maxiter=args.optimizer_maxiter,
                                n_jobs=1)
    for nombre, valor in (("cobyla_rhoend", args.cobyla_rhoend),
                          ("cobyla_rhobeg", args.cobyla_rhobeg)):
        if hasattr(numerical, nombre):
            setattr(numerical, nombre, valor)

    print("=" * 74)
    print(f"  PERFILADO statevector   {args.shell}-shell  Z_val={args.protons} "
          f"N_val={args.neutrons}  n_states={args.n_states}")
    print("=" * 74)

    # ---------------------------------------------------------------- 1
    t0 = time.perf_counter()
    ctx = build_context(problem, run, numerical)

    from vqd_nuclear.validation import resolve_beta
    beta_check = resolve_beta(run.beta, ctx.hamiltonian.exact_energies, run.n_states)
    run.beta = float(beta_check["beta"])     # ctx.run es el mismo objeto
    print(f"  beta                   : {run.beta:.6f}  ({beta_check.get('message', '')})")

    t_contexto = time.perf_counter() - t0

    n_qubits = ctx.hamiltonian.n_qubits
    n_h = len(ctx.hamiltonian.qubit_hamiltonian.terms)
    n_cand = len(ctx.active_operator_indices)
    dim_modelo = int(ctx.hamiltonian.matrix.shape[0])

    print(f"\n  build_context          : {fmt(t_contexto)}")
    print(f"  n_qubits               : {n_qubits}   (2^n = {2 ** n_qubits})")
    print(f"  dimension del modelo   : {dim_modelo}")
    print(f"  cadenas de Pauli en H  : {n_h}")
    print(f"  candidatos activos     : {n_cand}")
    print(f"  RATIO 2^n / dim_modelo : x{2 ** n_qubits / max(dim_modelo, 1):.0f}"
          "   <- cota superior de lo que daria proyectar sobre el espacio fisico")

    # ---------------------------------------------------------------- contadores
    # Se envuelven las dos primitivas para contar llamadas sin tocar el codigo.
    contadores = {"energia": 0, "aplicar_operador": 0, "barridos": 0}

    energia_original = type(ctx.measurements).statevector_energy
    aplicar_original = M.apply_qubit_operator_to_state
    barrido_original = getattr(type(ctx.measurements), "scan_candidate_gradients", None)

    def energia_contada(self, *a, **k):
        contadores["energia"] += 1
        return energia_original(self, *a, **k)

    def aplicar_contado(*a, **k):
        contadores["aplicar_operador"] += 1
        return aplicar_original(*a, **k)

    type(ctx.measurements).statevector_energy = energia_contada
    M.apply_qubit_operator_to_state = aplicar_contado

    if barrido_original is not None:
        def barrido_contado(self, *a, **k):
            contadores["barridos"] += 1
            return barrido_original(self, *a, **k)
        type(ctx.measurements).scan_candidate_gradients = barrido_contado

    # ---------------------------------------------------------------- 2 y 3
    opciones = dataclass_to_dict(numerical)
    opciones.pop("measurement_strategy", None)
    opciones.pop("use_model_space", None)

    perfil = cProfile.Profile()
    t0 = time.perf_counter()
    perfil.enable()
    resultados = run_vqd_family(ctx, cost_mode="statevector_exact",
                                options=dict(opciones))
    perfil.disable()
    t_calculo = time.perf_counter() - t0

    # restaurar
    type(ctx.measurements).statevector_energy = energia_original
    M.apply_qubit_operator_to_state = aplicar_original
    if barrido_original is not None:
        type(ctx.measurements).scan_candidate_gradients = barrido_original

    print(f"\n  calculo completo       : {fmt(t_calculo)}")
    print(f"  TOTAL                  : {fmt(t_contexto + t_calculo)}")
    print(f"    de los cuales contexto: {100 * t_contexto / (t_contexto + t_calculo):.0f}%")

    print(f"\n[ LLAMADAS ]")
    print(f"  statevector_energy          : {contadores['energia']:>8}")
    print(f"  apply_qubit_operator_to_state: {contadores['aplicar_operador']:>8}")
    if barrido_original is not None:
        print(f"  scan_candidate_gradients    : {contadores['barridos']:>8}")
    capas = sum(int(r.get("n_layers", 0)) for r in resultados)
    print(f"  capas ADAPT totales         : {capas:>8}")
    if capas:
        print(f"  evaluaciones de coste por capa: "
              f"{contadores['energia'] / capas:.0f}  <- esto es COBYLA")

    # ---------------------------------------------------------------- reparto
    flujo = _io.StringIO()
    stats = pstats.Stats(perfil, stream=flujo)

    print(f"\n[ TIEMPO ACUMULADO (cumtime): quien ORQUESTA el trabajo ]")
    stats.sort_stats("cumulative").print_stats(args.top)
    print("\n".join(flujo.getvalue().splitlines()[4:]))

    flujo = _io.StringIO()
    stats = pstats.Stats(perfil, stream=flujo)
    print(f"\n[ TIEMPO PROPIO (tottime): donde se QUEMA el tiempo de verdad ]")
    stats.sort_stats("time").print_stats(args.top)
    print("\n".join(flujo.getvalue().splitlines()[4:]))

    print("\n[ COMO LEERLO ]")
    print("  Mira tottime, no cumtime: cumtime se lo lleva siempre el bucle de")
    print("  arriba. Si apply_pauli_string_to_state o apply_qubit_operator_to_state")
    print("  dominan el tottime, la palanca es reducir el coste por evaluacion de")
    print("  energia (proyectar sobre el espacio del modelo), no repartir workers:")
    print("  COBYLA es serial y no hay nada que paralelizar entre sus iteraciones.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
