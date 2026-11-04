#!/usr/bin/env python3
"""Tests de la fase D (sampling.py + metodos muestreados de MeasurementEngine).

Criterio: un estimador con shots no tiene que "parecerse" al valor exacto,
tiene que alejarse de el lo que el ruido de shots permite. La varianza exacta
de cada circuito se conoce (MeasurementPlan.exact_statistics), asi que se
comprueba:
  - sesgo nulo y varianza correcta: z = (estimado - exacto)/sigma tiene
    <z^2> ~ 1 al repetir muchas veces;
  - cada estimacion concreta cae dentro de 5 sigma.

Bloques 1-5 autocontenidos (Qibo + operadores sinteticos; el 5 usa un motor
sintetico con MeasurementEngine real). El 6 (--con-motor) usa ⁶Be real.

Uso:
    python test_sampling.py
    python test_sampling.py --con-motor
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import qibo
from qibo import gates
from qibo.models import Circuit

from vqd_nuclear.measurement_plan import (build_energy_measurement_plan,
                                          build_gradient_measurement_plan, build_qwc_plan,
                                          circuit_unitary)
from vqd_nuclear.pauli_algebra import build_gradient_operator
from vqd_nuclear import sampling

import test_measurement_plan as T     # constructores JW sinteticos de la fase C

_fallos = 0
Z_MAX = 5.0


def marca(ok, texto):
    global _fallos
    _fallos += 0 if ok else 1
    print(f"  {'[OK]  ' if ok else '[FALLO]'} {texto}")


def semilla(s):
    np.random.seed(int(s))
    qibo.get_backend().set_seed(int(s))


def hamiltoniano_sintetico(n, rng):
    import itertools
    piezas, pesos = [], []
    for i in range(n):
        piezas.append(T.producto(T.jw(i, True), T.jw(i, False))); pesos.append(rng.normal())
    for i, j in itertools.combinations(range(n), 2):
        piezas.append(T.hopping(i, j)); pesos.append(rng.normal())
    for idx in [(0, 1, 2, 3), (0, 2, 4, 5), (1, 3, 4, 5), (0, 1, 0, 1), (0, 5, 1, 4)]:
        piezas.append(T.dos_cuerpos(*idx)); pesos.append(rng.normal())
    return T.ps_add(*piezas, pesos=pesos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--con-motor", action="store_true")
    ap.add_argument("--shell", default="p")
    ap.add_argument("--protons", type=int, default=2)
    ap.add_argument("--neutrons", type=int, default=0)
    ap.add_argument("--J", type=int, default=0)
    ap.add_argument("--nshots", type=int, default=20000)
    args = ap.parse_args()
    qibo.set_backend("numpy")
    semilla(20261103)
    rng = np.random.default_rng(20261103)

    # =================================================================
    print("\n1. Traduccion de puertas: Qibo == simulador del plan")
    # =================================================================
    lista = [("H", 0), ("SDG", 1), ("CNOT", 0, 2), ("CNOT", 2, 1), ("RX", 1, -np.pi / 2),
             ("RX", 0, 0.37), ("CNOT", 1, 0), ("H", 2)]
    c = sampling.add_plan_gates(Circuit(3), lista)
    d = float(np.max(np.abs(c.unitary() - circuit_unitary(lista, (0, 1, 2)))))
    marca(d < 1e-12, f"CNOT, H, S^dag y RX producen la misma unitaria ({d:.1e})")

    # =================================================================
    print("\n2. Orden de los bits medidos")
    # =================================================================
    # Estado |x> con x conocido; se mide un soporte desordenado respecto a los
    # qubits y se comprueba que la clave de Qibo sigue el orden de gates.M.
    n = 5
    x = 0b10110
    psi = np.zeros(2 ** n, dtype=complex); psi[x] = 1.0
    soporte = (1, 3, 4)
    c = Circuit(n); c.add(gates.M(*soporte))
    clave = next(iter(c(initial_state=psi.copy(), nshots=4).frequencies(binary=True)))
    esperado = "".join(str((x >> (n - 1 - q)) & 1) for q in soporte)
    marca(clave == esperado, f"clave '{clave}' == bits de {soporte} en ese orden ('{esperado}')")

    # =================================================================
    print("\n3. Energia sintetica: sesgo nulo y varianza correcta")
    # =================================================================
    n = 6
    H = hamiltoniano_sintetico(n, rng)
    psi = T.estado_aleatorio(n, rng)
    base = Circuit(n)
    for nombre, plan in (("paper", build_energy_measurement_plan(H)), ("qwc", build_qwc_plan(H))):
        exacto, var = plan.exact_statistics(psi, n)
        marca(abs(exacto - T.esperado(H, psi, n)) < 1e-10,
              f"[{nombre}] exact_statistics da <psi|H|psi>")
        z2 = []
        for nshots in (200, 2000):
            for _ in range(25):
                est = sampling.sampled_expectation(plan, base, nshots, initial_state=psi)
                z2.append(((est - exacto) / np.sqrt(sum(var) / nshots)) ** 2)
        m = float(np.mean(z2))
        # 50 muestras de chi^2_1: <z^2> = 1 con desviacion ~0.2
        marca(0.5 < m < 1.6, f"[{nombre}] <z^2> = {m:.2f} sobre 50 repeticiones (debe ser ~1)")
        nshots = 100000
        sigma = np.sqrt(sum(var) / nshots)
        est = sampling.sampled_expectation(plan, base, nshots, initial_state=psi)
        print(f"   [{nombre}] {plan.summary()['circuits']} circuitos, sigma por shot "
              f"{np.sqrt(sum(var)):.3f}; con {nshots} shots: {est:+.5f} vs {exacto:+.5f} "
              f"({(est - exacto) / sigma:+.2f} sigma)")
        marca(abs(est - exacto) < Z_MAX * sigma, f"[{nombre}] dentro de {Z_MAX:.0f} sigma")

    # =================================================================
    print("\n4. Gradientes sinteticos: <psi|G_mu|psi> con shots")
    # =================================================================
    nshots = 50000
    for idx in [(0, 1, 2, 3), (0, 2, 4, 5), (1, 3, 4, 5)]:
        G = build_gradient_operator(H, T.generador(*idx))
        plan = build_gradient_measurement_plan(G)
        exacto, var = plan.exact_statistics(psi, n)
        sigma = np.sqrt(sum(var) / nshots)
        est = sampling.sampled_expectation(plan, base, nshots, initial_state=psi)
        marca(abs(est - exacto) < Z_MAX * sigma,
              f"G_{idx}: {est:+.5f} vs {exacto:+.5f} ({(est - exacto) / sigma:+.2f} sigma, "
              f"{plan.summary()['circuits']} circuitos)")

    # =================================================================
    print("\n5. MeasurementEngine con un motor sintetico")
    # =================================================================
    from vqd_nuclear.measurements import MeasurementEngine

    class AnsatzSintetico:
        """Imita la interfaz de AnsatzBuilder: build, state, cache_key, clear_cache.

        Cada operador k es una capa RY(theta) sobre el qubit k % n + escalera de
        CNOTs. No es un ansatz fisico: solo sirve para tener circuitos de Qibo
        con parametros y estados no triviales.
        """

        def __init__(self, n):
            self.n = n

        def build(self, thetas, ops):
            c = Circuit(self.n)
            c.add(gates.X(0)); c.add(gates.X(3))
            for th, op in zip(thetas, ops):
                c.add(gates.RY(int(op) % self.n, theta=float(th)))
                for q in range(self.n - 1):
                    c.add(gates.CNOT(q, q + 1))
            return c

        def state(self, thetas, ops):
            return np.asarray(self.build(thetas, ops)().state(), dtype=complex)

        def cache_key(self, thetas, ops):
            return tuple(np.round(thetas, 12)), tuple(ops)

        def clear_cache(self):
            pass

    pool = [T.generador(*idx) for idx in [(0, 1, 2, 3), (0, 2, 4, 5), (1, 3, 4, 5), (0, 3, 1, 2)]]
    eng = MeasurementEngine(ansatz_builder=AnsatzSintetico(n), qubit_hamiltonian=H, n_qubits=n,
                            available_workers=1, qubit_generator_pool=pool)
    thetas, ops = [0.4, -0.7, 1.1], [1, 2, 4]
    psi = eng.ansatz_builder.state(thetas, ops)

    nshots = 50000
    exacto = eng.statevector_energy(thetas, ops)
    sigma = eng.energy_plan.exact_standard_error(psi, n, nshots)
    est = eng.sampled_energy(thetas, ops, nshots=nshots)
    marca(abs(est - exacto) < Z_MAX * sigma,
          f"sampled_energy {est:+.5f} vs statevector {exacto:+.5f} ({(est - exacto) / sigma:+.2f} sigma)")
    marca(eng.measurement_summary["circuits"] == eng.energy_plan.summary()["circuits"],
          f"measurement_summary disponible: {eng.measurement_summary}")

    exactos = {}
    for mu in range(len(pool)):
        exactos[mu] = eng.statevector_energy_gradient(thetas, ops, pool[mu], psi=psi)
        sigma = eng.gradient_plan(mu).exact_standard_error(psi, n, nshots)
        est = eng.sampled_energy_gradient(thetas, ops, mu, nshots=nshots)
        marca(abs(est - exactos[mu]) < Z_MAX * sigma,
              f"op {mu}: g^E muestreado {est:+.5f} vs statevector {exactos[mu]:+.5f} "
              f"({(est - exactos[mu]) / sigma:+.2f} sigma)")

    scan = eng.sampled_candidate_scan(thetas, ops, list(range(len(pool))), nshots=nshots)
    ganador_sv = max(exactos, key=lambda m: abs(exactos[m]))
    ganador_mu = max(scan.g_total, key=lambda m: abs(scan.g_total[m]))
    orden = sorted((abs(v) for v in exactos.values()), reverse=True)
    print(f"   |g| exactos ordenados: {[round(v, 4) for v in orden]}")
    marca(ganador_mu == ganador_sv, f"el barrido muestreado elige el mismo operador ({ganador_mu})")
    marca(all(np.isnan(v) for v in scan.g_penalty.values()),
          "g_penalty marcado como NaN (no medible), no como 0")

    t2, o2 = [0.2, 0.5], [0, 3]
    psi2 = eng.ansatz_builder.state(t2, o2)
    F = eng.exact_overlap(thetas, ops, t2, o2)
    est = eng.destructive_swap_overlap(thetas, ops, t2, o2, nshots=nshots)
    sigma = np.sqrt(max(1 - F * F, 1e-12) / nshots)
    marca(abs(est - F) < Z_MAX * sigma,
          f"SWAP destructivo {est:.5f} vs exacto {F:.5f} ({(est - F) / sigma:+.2f} sigma)")

    # =================================================================
    if args.con_motor:
        print(f"\n6. ⁶Be real ({args.nshots} shots por circuito)")
        # =============================================================
        from vqd_nuclear.config import ProblemConfig, RunConfig, NumericalConfig
        from vqd_nuclear.vqd import build_context

        problem = ProblemConfig(shell=args.shell, n_valence_protons=args.protons,
                                n_valence_neutrons=args.neutrons, total_j=args.J)
        ctx = build_context(problem, RunConfig(n_states=2, n_workers=1), NumericalConfig(n_jobs=1))
        eng = ctx.measurements
        nq = ctx.hamiltonian.n_qubits
        semilla(20261103)
        nshots = int(args.nshots)

        ops = [int(i) for i in ctx.active_operator_indices[:2]]
        thetas = [0.31, -0.17]
        psi = eng.ansatz_builder.state(thetas, ops)

        exacto = eng.statevector_energy(thetas, ops)
        sigma = eng.energy_plan.exact_standard_error(psi, nq, nshots)
        est = eng.sampled_energy(thetas, ops, nshots=nshots)
        print(f"   plan de H: {eng.measurement_summary}")
        marca(abs(est - exacto) < Z_MAX * sigma,
              f"energia {est:+.5f} vs {exacto:+.5f} (sigma {sigma:.1e}, {(est - exacto) / sigma:+.2f} sigma)")

        cand = [int(m) for m in ctx.active_operator_indices]
        exactos = {}
        print(f"   {'op':>4} {'muestreado':>12} {'statevector':>12} {'sigma':>9} {'z':>6}")
        ok = True
        for mu in cand:
            gen = ctx.hamiltonian.qubit_generator_pool[mu]
            exactos[mu] = eng.statevector_energy_gradient(thetas, ops, gen, psi=psi)
            sigma = eng.gradient_plan(mu).exact_standard_error(psi, nq, nshots)
            est = eng.sampled_energy_gradient(thetas, ops, mu, nshots=nshots)
            z = (est - exactos[mu]) / sigma if sigma > 0 else 0.0
            ok &= abs(z) < Z_MAX
            print(f"   {mu:>4} {est:>12.5f} {exactos[mu]:>12.5f} {sigma:>9.1e} {z:>+6.2f}")
        marca(ok, f"todos los gradientes dentro de {Z_MAX:.0f} sigma")

        scan = eng.sampled_candidate_scan(thetas, ops, cand, nshots=nshots)
        g_sv = max(exactos, key=lambda m: abs(exactos[m]))
        g_mu = max(scan.g_total, key=lambda m: abs(scan.g_total[m]))
        marca(g_mu == g_sv, f"el barrido muestreado elige el mismo operador que statevector ({g_mu})")

        t2, o2 = [0.5], [ops[1]]
        F = eng.exact_overlap(thetas, ops, t2, o2)
        est = eng.destructive_swap_overlap(thetas, ops, t2, o2, nshots=nshots)
        sigma = np.sqrt(max(1 - F * F, 1e-12) / nshots)
        marca(abs(est - F) < Z_MAX * sigma,
              f"SWAP destructivo {est:.5f} vs exacto {F:.5f} ({(est - F) / sigma:+.2f} sigma)")
    else:
        print("\n6. (omitido; usa --con-motor para 6Be real)")

    print("\n" + "=" * 60)
    print("  Todo pasa." if _fallos == 0 else f"  {_fallos} comprobacion(es) fallan.")
    return 0 if _fallos == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
