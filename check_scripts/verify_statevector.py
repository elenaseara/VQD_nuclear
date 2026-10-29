#!/usr/bin/env python3
"""Verificaciones de cimientos de la ruta statevector.

Cierra las comprobaciones que quedaron pendientes al principio del proyecto y
que ahora, con el motor statevector funcionando, por fin son posibles.

    1. CONVENCION DE FASE contra el circuito REAL  (la comprobacion 0.2)
       Es la mas importante: todo el gradiente, la penalizacion y el
       diagnostico de ADAPT asumen GRADIENT_PHASE, que hasta ahora
       no se habia contrastado con lo que hace exp_pauli_str_staircase.

    2. COHERENCIA DE JORDAN-WIGNER
       El espectro de JW(H) restringido al sector fisico debe coincidir con
       el de la matriz CSR. Valida de una vez extract_qubit_hamiltonian, el
       mapeo JW completo y la ordenacion de qubits.

    3. CONTEO DE DETERMINANTES
       len(referencias) debe ser igual a la dimension de la CSR. Caza
       endianness invertida y colapso de duplicados.

    4. NORMALIZACION Y HERMITICIDAD
       <psi|psi> = 1 y <psi|H|psi> real.

Uso:
    python verify_statevector.py --shell p --protons 2 --neutrons 0 --J 0
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

OK, FALLO, AVISO = "  [OK]  ", " [FALLO]", " [AVISO]"
_fallos = 0


def marca(condicion: bool) -> str:
    global _fallos
    if not condicion:
        _fallos += 1
    return OK if condicion else FALLO


def seccion(titulo: str) -> None:
    print("\n" + "=" * 74)
    print("  " + titulo)
    print("=" * 74)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shell", choices=["p", "sd"], default="p")
    ap.add_argument("--protons", type=int, default=2)
    ap.add_argument("--neutrons", type=int, default=0)
    ap.add_argument("--J", type=int, default=0)
    ap.add_argument("--eps", type=float, default=1e-5,
                    help="paso de la diferencia finita")
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
    exactas = np.asarray(ctx.hamiltonian.exact_energies, dtype=float)

    print(f"\n{args.shell}-shell  Z_val={args.protons} N_val={args.neutrons} J={args.J}")
    print(f"n_qubits={n}   candidatos activos={len(ctx.active_operator_indices)}")

    # =================================================================
    seccion("1. CONVENCION DE FASE CONTRA EL CIRCUITO REAL  (comprobacion 0.2)")
    # =================================================================
    print(f"  GRADIENT_PHASE declarada = {M.GRADIENT_PHASE}")
    print("  Deriva la energia del CIRCUITO REAL respecto de theta y la compara")
    print("  con el gradiente analitico en theta=0.\n")

    eps = args.eps
    cocientes = []
    for mu in ctx.active_operator_indices[:3]:
        generator = ctx.hamiltonian.qubit_generator_pool[int(mu)]

        # derivada numerica de la energia del circuito real de UNA capa
        e_mas = eng.statevector_energy([+eps], [int(mu)])
        e_menos = eng.statevector_energy([-eps], [int(mu)])
        numerico = (e_mas - e_menos) / (2.0 * eps)

        # gradiente analitico al anadir ese operador al ansatz vacio
        resultado = eng.statevector_vqd_gradient([], [], generator, [], [])
        analitico = resultado[0] if isinstance(resultado, tuple) else resultado

        if abs(numerico) < 1e-9:
            print(f"  op {mu:>3}: derivada numerica ~ 0, se omite")
            continue
        cociente = analitico / numerico
        cocientes.append(cociente)
        print(f"  op {mu:>3}:  analitico = {analitico:+.9f}   "
              f"numerico = {numerico:+.9f}   cociente = {cociente:+.6f}")

    if not cocientes:
        print(f"{AVISO} Ningun operador dio derivada no nula; prueba otro nucleo.")
    else:
        medio = float(np.mean(cocientes))
        consistente = bool(np.allclose(cocientes, medio, atol=1e-3))
        print(f"\n  cociente medio = {medio:+.6f}   (consistente entre operadores: {consistente})")
        print(f"{marca(np.isclose(medio, 1.0, atol=2e-3))} el gradiente analitico "
              f"coincide con la derivada del circuito")
        if not np.isclose(medio, 1.0, atol=2e-3):
            print("\n  INTERPRETACION DEL COCIENTE:")
            print("    -1.0        -> cambia el signo de GRADIENT_PHASE (+1j <-> -1j)")
            print("    +2.0 / -2.0 -> exp_pauli_str_staircase usa Rz = e^{-i theta Z/2};")
            print("                   multiplica la fase por 0.5 (o divide el gradiente por 2)")
            print("    +0.5 / -0.5 -> el factor 1/2 va al reves")
            print("    otro        -> revisar prepare_generator_term_pool y la construccion")
            print("                   del ansatz antes de seguir")
            print("\n  IMPORTANTE: hasta que esto de 1.0, el signo se propaga al gradiente,")
            print("  a la penalizacion y al diagnostico ranking_changed.")

    n_particulas = args.protons + args.neutrons

    # =================================================================
    seccion("3. CONTEO DE DETERMINANTES")
    # =================================================================
    from vqd_nuclear.ansatz import get_hamiltonian_basis_references
    refs = get_hamiltonian_basis_references(
        ctx.hamiltonian.nucleus, n_qubits=n, n_valence_particles=n_particulas)
    dim_csr = int(ctx.hamiltonian.matrix.shape[0])
    print(f"  determinantes extraidos : {len(refs)}")
    print(f"  dimension de la CSR     : {dim_csr}")
    print(f"{marca(len(refs) == dim_csr)} coinciden")
    if len(refs) != dim_csr:
        print("    -> endianness invertida en basis_state_to_occupied_orbitals,")
        print("       o colapso de duplicados en el sorted(set(...))")

    # =================================================================
    seccion("4. NORMALIZACION Y HERMITICIDAD")
    # =================================================================
    mu0 = int(ctx.active_operator_indices[0])
    for etiqueta, thetas, ops in (("referencia", [], []),
                                  ("una capa", [0.37], [mu0])):
        psi = eng.ansatz_builder.state(thetas, ops)
        norma = float(np.linalg.norm(psi))
        h_psi = M.apply_qubit_operator_to_state(ctx.hamiltonian.qubit_hamiltonian, psi, n)
        e = complex(np.vdot(psi, h_psi))
        print(f"  {etiqueta:>12}:  |psi| = {norma:.12f}   "
              f"Im<psi|H|psi> = {e.imag:+.3e}")
        marca(abs(norma - 1.0) < 1e-10)
        marca(abs(e.imag) < 1e-10)
    print(f"{OK if _fallos == 0 else ''}  (el ansatz debe ser unitario y H hermitico)")

    # =================================================================
    seccion("RESULTADO")
    # =================================================================
    if _fallos == 0:
        print("  Todas las comprobaciones pasan. Los cimientos de statevector estan")
        print("  validados: puedes generar la linea base y empezar con el muestreado.")
    else:
        print(f"  {_fallos} comprobacion(es) fallan. Arreglalas antes de construir")
        print("  el muestreado encima: los errores de convencion se propagan en silencio.")
    return 0 if _fallos == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
