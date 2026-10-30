#!/usr/bin/env python3
"""Ruta statevector calculada en el ESPACIO DEL MODELO.

QUE ES
------
El vector de estado tiene 2^N componentes, pero solo d de ellas corresponden a
estados fisicos: los determinantes con el numero correcto de particulas y el M
correcto. En ⁶Be, d = 5 de 64. Las otras 59 son cero SIEMPRE, porque ni H ni los
generadores sacan amplitud del espacio del modelo (verificado: fuga 5.6e-17).

Aqui se trabaja con el vector de d componentes. Se proyectan las DOS mitades del
coste, que el perfilado mostro que pesan casi lo mismo:

    PROPAGACION  ansatz_builder.state()  -> ~48% del tiempo
                 Hoy: construir un objeto Circuit de Qibo puerta a puerta y
                 simularlo sobre 2^N amplitudes (455k apply_gate, 1.8M reshape).
                 Aqui: productos de matrices d x d.

    MEDIDA       qubit_operator_expectation_from_state()  -> ~43%
                 Hoy: recorrer las N_H cadenas de Pauli sobre 2^N amplitudes.
                 Aqui: psi_S^dagger H_S psi_S.

POR QUE ES EXACTO, NO UNA APROXIMACION
--------------------------------------
Si T conserva el espacio del modelo, su matriz es diagonal por bloques y la
exponencial de una matriz diagonal por bloques es la matriz por bloques de las
exponenciales:

    exp(theta T) restringido a S  =  exp(theta T_S)

Ademas ya se verifico que todas las cadenas de Pauli dentro de un mismo
generador conmutan entre si, asi que el circuito de una capa es exactamente
exp(theta T) y no una aproximacion de Trotter. Las dos cosas juntas hacen que
esto sea la MISMA cuenta, no una mas barata y peor.

Esto vale solo para la ruta statevector, cuyo proposito es dar el resultado
exacto. La ruta full_quantum_circuit sigue construyendo circuitos de verdad,
que es donde el circuito es el objeto de estudio.

GUARDAS
-------
1. Al proyectar cada operador se mide cuanta amplitud saca fuera del espacio. Si
   alguno no lo conserva, queda marcado y los ansatze que lo usen caen al camino
   completo.
2. En cada evaluacion se mide 1 - ||psi_S||^2. Cuesta O(d), es decir nada.

En los dos casos el fallback es el camino completo: una hipotesis incorrecta
cuesta velocidad, nunca correctitud.

Uso como script (valida contra el camino actual):
    python model_space.py --shell p --protons 2 --neutrons 0 --J 0
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import OrderedDict
from dataclasses import dataclass
from vqd_nuclear.model_space import ModelSpaceEngine, model_space_indices

import numpy as np

LOGGER = logging.getLogger(__name__)

# Debe ser la MISMA constante que measurements.py
ANSATZ_DERIVATIVE_PHASE = -1j

LEAK_TOL = 1e-10

# =====================================================================
# 5. Validacion contra el camino actual
# =====================================================================

def main() -> int:
    import time

    ap = argparse.ArgumentParser()
    ap.add_argument("--shell", choices=["p", "sd"], default="p")
    ap.add_argument("--protons", type=int, default=2)
    ap.add_argument("--neutrons", type=int, default=0)
    ap.add_argument("--J", type=int, default=0)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20261021)
    args = ap.parse_args()

    from vqd_nuclear.config import ProblemConfig, RunConfig, NumericalConfig
    from vqd_nuclear.vqd import build_context
    from vqd_nuclear.ansatz import get_hamiltonian_basis_references
    from vqd_nuclear import measurements as M

    problem = ProblemConfig(shell=args.shell, n_valence_protons=args.protons,
                            n_valence_neutrons=args.neutrons, total_j=args.J)
    ctx = build_context(problem, RunConfig(n_states=2, n_workers=1),
                        NumericalConfig(n_jobs=1))
    eng = ctx.measurements
    n = ctx.hamiltonian.n_qubits
    n_part = args.protons + args.neutrons

    if ANSATZ_DERIVATIVE_PHASE != getattr(M, "ANSATZ_DERIVATIVE_PHASE", ANSATZ_DERIVATIVE_PHASE):
        print(f"AVISO: la constante de este fichero ({ANSATZ_DERIVATIVE_PHASE}) no "
              f"es la de measurements.py ({M.ANSATZ_DERIVATIVE_PHASE}).")

    referencias = get_hamiltonian_basis_references(
        ctx.hamiltonian.nucleus, n_qubits=n, n_valence_particles=n_part)
    indices = model_space_indices(referencias, n)
    d = len(indices)

    print(f"\n{args.shell}-shell  n_qubits={n}  2^n={2 ** n}  dim modelo={d}"
          f"  ratio x{2 ** n / d:.0f}")

    t0 = time.perf_counter()
    motor = ModelSpaceEngine(
        qubit_hamiltonian=ctx.hamiltonian.qubit_hamiltonian,
        qubit_generator_pool=ctx.hamiltonian.qubit_generator_pool,
        indices=indices,
        reference_occupied_orbitals=eng.ansatz_builder.reference_occupied_orbitals,
        n_qubits=n,
        apply_operator=M.apply_qubit_operator_to_state,
    )
    candidatos = [int(c) for c in ctx.active_operator_indices]
    for mu in candidatos:
        motor.generator(mu)
    t_construccion = time.perf_counter() - t0
    print(f"motor construido en {t_construccion:.3f} s  (H_S + {len(candidatos)} "
          f"generadores, coste unico)")
    print(f"conserva el espacio: H={motor.available}, generadores="
          f"{all(motor.generator(m).preserves_space for m in candidatos)}")

    vals = np.linalg.eigvalsh(motor.hamiltonian_s)
    exactas = np.asarray(ctx.hamiltonian.exact_energies, dtype=float)
    k = min(len(vals), len(exactas))
    print(f"espectro de H_S vs exact_energies: "
          f"{float(np.max(np.abs(vals[:k] - exactas[:k]))):.2e}")

    rng = np.random.default_rng(args.seed)
    peor_estado = peor_energia = peor_overlap = 0.0
    t_completo = t_proyectado = 0.0

    # Un segundo ansatz fijo, para los solapamientos.
    ops_ref = candidatos[:2]
    thetas_ref = [0.23, -0.41][:len(ops_ref)]

    for _ in range(args.trials):
        n_capas = int(rng.integers(1, min(4, len(candidatos)) + 1))
        ops = [int(c) for c in rng.choice(candidatos, size=n_capas, replace=False)]
        thetas = [float(x) for x in rng.normal(scale=0.8, size=n_capas)]

        # --- camino actual -------------------------------------------
        t0 = time.perf_counter()
        psi = eng.ansatz_builder.state(thetas, ops)
        h_psi = M.apply_qubit_operator_to_state(ctx.hamiltonian.qubit_hamiltonian, psi, n)
        e_completo = float(np.vdot(psi, h_psi).real)
        psi_ref = eng.ansatz_builder.state(thetas_ref, ops_ref)
        ov_completo = float(abs(np.vdot(psi_ref, psi)) ** 2)
        t_completo += time.perf_counter() - t0

        # --- camino proyectado ---------------------------------------
        motor._cache.clear()        # medir el coste real, no el de la cache
        t0 = time.perf_counter()
        psi_s = motor.state(thetas, ops)
        e_proyectado = motor.energy(thetas, ops)
        ov_proyectado = motor.overlap(thetas_ref, ops_ref, thetas, ops)
        t_proyectado += time.perf_counter() - t0

        # El estado completo restringido al subespacio debe ser el proyectado,
        # salvo fase global: el circuito de Qibo puede introducir una.
        a, b = psi[indices], psi_s
        solape = np.vdot(a, b)
        fase = solape / abs(solape) if abs(solape) > 1e-12 else 1.0
        peor_estado = max(peor_estado, float(np.max(np.abs(a * fase - b))))
        peor_energia = max(peor_energia, abs(e_completo - e_proyectado))
        peor_overlap = max(peor_overlap, abs(ov_completo - ov_proyectado))

    print(f"\n{args.trials} ansatze aleatorios de hasta 4 capas:")
    print(f"  estado (salvo fase global) : {peor_estado:.3e}")
    print(f"  energia                    : {peor_energia:.3e} MeV")
    print(f"  solapamiento               : {peor_overlap:.3e}")
    print(f"\n  camino actual     : {1e6 * t_completo / args.trials:9.1f} us/eval")
    print(f"  camino proyectado : {1e6 * t_proyectado / args.trials:9.1f} us/eval")
    if t_proyectado > 0:
        print(f"  FACTOR            : x{t_completo / t_proyectado:.0f}")

    ok = peor_energia < 1e-10 and peor_overlap < 1e-10 and peor_estado < 1e-9
    print("\n  " + ("[OK] Equivalente.."
                    if ok else
                    "[FALLO] NO equivalente."))
    if peor_estado >= 1e-9 and peor_energia < 1e-10:
        print("  El estado difiere pero la energia no: probablemente una fase")
        print("  relativa, no global. Revisa el signo de ANSATZ_DERIVATIVE_PHASE")
        print("  en ProjectedGenerator.apply antes de fiarte de los gradientes.")
    return 0 if ok else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
