#!/usr/bin/env python3
"""Tests de la fase C (measurement_plan.py).

Bloques 1-8 autocontenidos: operadores fermionicos sinteticos con Jordan-Wigner
construido aqui, comparados contra matrices densas. No necesitan qUBshell.
El bloque 9 (--con-motor) usa el H real de 6Be y los G_mu de todo el pool.

Uso:
    python test_measurement_plan.py
    python test_measurement_plan.py --con-motor
"""
from __future__ import annotations

import argparse
import itertools
import sys

import numpy as np

from vqd_nuclear.pauli_algebra import PauliSum, build_gradient_operator, multiply_pauli_strings
from vqd_nuclear.measurement_plan import (
    QWCGroup, apply_circuit, build_energy_measurement_plan,
    build_gradient_measurement_plan, circuit_unitary, compression_basis_change,
    diagonalize_bucket, paper_basis_change, specialized_8_index_basis_change,
)

_P = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}
_fallos = 0


def marca(ok, texto):
    global _fallos
    _fallos += 0 if ok else 1
    print(f"  {'[OK]  ' if ok else '[FALLO]'} {texto}")


# ---------------------------------------------------------------------
# utilidades: matrices densas y Jordan-Wigner minimo
# ---------------------------------------------------------------------

def densa_cadena(cadena, qubits):
    """Matriz de una cadena sobre la lista ordenada `qubits` (qubits[0] = MSB)."""
    mapa = dict(cadena)
    m = np.eye(1, dtype=complex)
    for q in qubits:
        m = np.kron(m, _P[mapa.get(q, "I")])
    return m


def densa(op, qubits):
    dim = 2 ** len(qubits)
    out = np.zeros((dim, dim), dtype=complex)
    for cadena, c in op.terms.items():
        out += complex(c) * densa_cadena(cadena, qubits)
    return out


def ps_mul(a, b):
    out = {}
    for p, ca in a.terms.items():
        for q, cb in b.terms.items():
            fase, r = multiply_pauli_strings(p, q)
            out[r] = out.get(r, 0) + complex(ca) * complex(cb) * fase
    return PauliSum({k: v for k, v in out.items() if abs(v) > 1e-14})


def ps_add(*ops, pesos=None):
    out = {}
    pesos = pesos or [1.0] * len(ops)
    for w, op in zip(pesos, ops):
        for k, v in op.terms.items():
            out[k] = out.get(k, 0) + w * complex(v)
    return PauliSum({k: v for k, v in out.items() if abs(v) > 1e-14})


def ps_dag(a):
    return PauliSum({k: np.conj(v) for k, v in a.terms.items()})


def jw(p, crear):
    """a_p^dagger (crear=True) o a_p con Jordan-Wigner: Z_0..Z_{p-1} (X -+ iY)/2."""
    z = tuple((q, "Z") for q in range(p))
    s = -0.5j if crear else 0.5j
    return PauliSum({tuple(sorted(z + ((p, "X"),))): 0.5, tuple(sorted(z + ((p, "Y"),))): s})


def producto(*factores):
    out = PauliSum({(): 1.0})
    for f in factores:
        out = ps_mul(out, f)
    return out


def hopping(i, j):          # a_i^dag a_j + h.c.
    t = producto(jw(i, True), jw(j, False))
    return ps_add(t, ps_dag(t))


def dos_cuerpos(i, j, k, l):  # a_i^dag a_j^dag a_k a_l + h.c.
    t = producto(jw(i, True), jw(j, True), jw(k, False), jw(l, False))
    return ps_add(t, ps_dag(t))


def generador(i, j, k, l):    # A = i (tau - tau^dag), hermitico
    t = producto(jw(i, True), jw(j, True), jw(k, False), jw(l, False))
    return ps_add(t, ps_dag(t), pesos=[1j, -1j])


def estado_aleatorio(n, rng):
    v = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
    return v / np.linalg.norm(v)


def esperado(op, psi, n):
    return float(np.real(np.vdot(psi, densa(op, list(range(n))) @ psi)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--con-motor", action="store_true")
    ap.add_argument("--shell", default="p")
    ap.add_argument("--protons", type=int, default=2)
    ap.add_argument("--neutrons", type=int, default=0)
    ap.add_argument("--J", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(20261103)

    # =================================================================
    print("\n1. Simulador de puertas contra matrices densas")
    # =================================================================
    n = 4
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    SDG = np.diag([1, -1j])

    def gate_dense(g):
        if g[0] in ("H", "SDG"):
            ops = [np.eye(2)] * n
            ops[g[1]] = H if g[0] == "H" else SDG
            m = np.eye(1)
            for o in ops:
                m = np.kron(m, o)
            return m
        c, t = g[1], g[2]
        m = np.zeros((2 ** n, 2 ** n))
        for x in range(2 ** n):
            bits = [(x >> (n - 1 - q)) & 1 for q in range(n)]
            if bits[c]:
                bits[t] ^= 1
            y = int("".join(map(str, bits)), 2)
            m[y, x] = 1
        return m

    peor = 0.0
    for g in [("H", 0), ("H", 3), ("SDG", 1), ("CNOT", 0, 2), ("CNOT", 3, 1), ("CNOT", 2, 0)]:
        psi = estado_aleatorio(n, rng)
        peor = max(peor, np.max(np.abs(apply_circuit(psi, [g], n) - gate_dense(g) @ psi)))
    marca(peor < 1e-12, f"H, S^dag y CNOT (en ambos sentidos) correctos ({peor:.1e})")

    # =================================================================
    print("\n2. Por que D = V O V^dag y no V^dag O V")
    # =================================================================
    act = (0, 1, 2, 3)
    for nombre, gates in [("M_ijkl (palindromo)", paper_basis_change("double_hopping", act)),
                          ("compresion eje Y", compression_basis_change(act, "Y"))]:
        V = circuit_unitary(gates, act)
        print(f"   {nombre:22s} ||V - V^dag|| = {np.linalg.norm(V - V.conj().T):.2e}")
    # Un caso donde la convencion antigua falla: Y sobre un qubit, rotacion S^dag H.
    O = _P["Y"]
    V = circuit_unitary([("SDG", 0), ("H", 0)], (0,))
    nuevo, viejo = V @ O @ V.conj().T, V.conj().T @ O @ V
    psi = estado_aleatorio(1, rng)
    probs = np.abs(V @ psi) ** 2
    verdad = float(np.real(np.vdot(psi, O @ psi)))
    marca(abs(probs @ np.diag(nuevo).real - verdad) < 1e-12,
          f"V O V^dag reproduce <Y> = {verdad:+.4f}")
    off_viejo = np.linalg.norm(viejo - np.diag(np.diag(viejo)))
    print(f"   con V^dag O V: diag = {np.diag(viejo).real}, fuera de diagonal {off_viejo:.2f}"
          "  -> la convencion antigua ni siquiera diagonaliza")

    # =================================================================
    print("\n3. Factorizacion por espectadores == D densa del pseudocodigo")
    # =================================================================
    peor_norma = peor_diag = 0.0
    for _ in range(40):
        activos = tuple(sorted(rng.choice(6, size=4, replace=False).tolist()))
        espect = [q for q in range(6) if q not in activos]
        terms = {}
        for _ in range(5):
            cad = [(q, rng.choice(["X", "Y"])) for q in activos]
            cad += [(q, "Z") for q in espect if rng.random() < 0.5]
            terms[tuple(sorted(cad))] = complex(rng.normal())
        soporte = tuple(sorted({q for t in terms for q, _ in t}))
        gates = paper_basis_change("double_hopping", activos)
        bloques, norma = diagonalize_bucket(terms, activos, soporte, gates)
        V = circuit_unitary(gates, soporte)
        Od = sum(c * densa_cadena(t, soporte) for t, c in terms.items())
        D = V @ Od @ V.conj().T
        norma_densa = np.linalg.norm(D - np.diag(np.diag(D)))
        peor_norma = max(peor_norma, abs(norma - norma_densa))
        from vqd_nuclear.measurement_plan import MeasurementGroup
        g = MeasurementGroup("x", activos, soporte, gates, terms, bloques, norma)
        peor_diag = max(peor_diag, np.max(np.abs(g.diagonal_values() - np.diag(D).real)))
    marca(peor_norma < 1e-10, f"||offdiag D|| identica a la densa ({peor_norma:.1e})")
    marca(peor_diag < 1e-10, f"diag(D) identica a la densa ({peor_diag:.1e})")

    # =================================================================
    print("\n4. Circuitos del articulo sobre operadores fermionicos reales")
    # =================================================================
    # Single hopping con cadenas JW intermedias: incluye el ejemplo de la
    # seccion 7, Z2 Z3 (X1 X4 + Y1 Y4), que es a1^dag a4 + h.c. con JW.
    h14 = hopping(1, 4)
    print(f"   a1^dag a4 + h.c. -> {sorted(h14.terms)}")
    plan = build_energy_measurement_plan(h14)
    marca(plan.summary().get("single_hopping") == 1 and not plan.qwc_groups,
          "el ejemplo de la seccion 7 va a un solo grupo single_hopping")
    # Los tres patrones de double hopping (articulo, ec. 22-24).
    for idx in [(0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2), (1, 4, 2, 5)]:
        op = dos_cuerpos(*idx)
        plan = build_energy_measurement_plan(op)
        s = plan.summary()
        marca(s.get("double_hopping") == 1 and not plan.qwc_groups,
              f"a^dag_{idx[0]} a^dag_{idx[1]} a_{idx[2]} a_{idx[3]} + h.c. -> 1 circuito double_hopping")

    # =================================================================
    print("\n5. Puerta de validacion: rechaza lo que no se puede diagonalizar")
    # =================================================================
    # X0 X1 y Y0 X1 tienen los mismos activos pero ANTICONMUTAN: ningun
    # circuito los diagonaliza a la vez. Deben ir a QWC.
    malo = PauliSum({((0, "X"), (1, "X")): 0.7, ((0, "Y"), (1, "X")): 0.4})
    plan = build_energy_measurement_plan(malo)
    marca(not plan.groups and plan.rejected, f"rechazado (||offdiag|| = {plan.rejected[(0, 1)]:.3f})")
    psi = estado_aleatorio(2, rng)
    marca(abs(plan.exact_expectation(psi, 2) - esperado(malo, psi, 2)) < 1e-12,
          "y medido por QWC sigue dando el valor correcto")

    # =================================================================
    print("\n6. QWC: las rotaciones dejan +Z (seccion 13)")
    # =================================================================
    peor = 0.0
    for letras in itertools.product("XYZ", repeat=3):
        cad = tuple((q, l) for q, l in enumerate(letras))
        g = QWCGroup(basis=dict(cad), terms={cad: 1.0})
        V = circuit_unitary(g.gates, (0, 1, 2))
        peor = max(peor, np.max(np.abs(V @ densa_cadena(cad, [0, 1, 2]) @ V.conj().T
                                       - densa_cadena(tuple((q, "Z") for q in range(3)), [0, 1, 2]))))
    marca(peor < 1e-12, f"V P V^dag = +Z en las 27 cadenas de 3 qubits ({peor:.1e})")

    # =================================================================
    print("\n7. Plan completo: H sintetico y G_mu, valor exacto vs <psi|O|psi>")
    # =================================================================
    n = 6
    piezas, pesos = [], []
    for i in range(n):
        piezas.append(producto(jw(i, True), jw(i, False))); pesos.append(rng.normal())
    for i, j in itertools.combinations(range(n), 2):
        piezas.append(hopping(i, j)); pesos.append(rng.normal())
    for i, j, k, l in [(0, 1, 2, 3), (0, 2, 4, 5), (1, 3, 4, 5), (0, 1, 0, 1), (2, 4, 2, 4),
                       (0, 5, 1, 4), (1, 2, 3, 0)]:
        piezas.append(dos_cuerpos(i, j, k, l)); pesos.append(rng.normal())
    Hs = ps_add(*piezas, pesos=pesos)
    plan_E = build_energy_measurement_plan(Hs)
    psi = estado_aleatorio(n, rng)
    dif = abs(plan_E.exact_expectation(psi, n) - esperado(Hs, psi, n))
    print(f"   H: {len(Hs)} cadenas -> {plan_E.summary()}")
    marca(dif < 1e-10, f"energia por el plan == <psi|H|psi> ({dif:.1e})")

    for idx in [(0, 1, 2, 3), (0, 2, 4, 5), (1, 3, 4, 5)]:
        A = generador(*idx)
        G = build_gradient_operator(Hs, A)
        p1 = build_gradient_measurement_plan(G)
        p2 = build_gradient_measurement_plan(G, compression_fallback=True)
        v = esperado(G, psi, n)
        d1 = abs(p1.exact_expectation(psi, n) - v)
        d2 = abs(p2.exact_expectation(psi, n) - v)
        s1, s2 = p1.summary(), p2.summary()
        print(f"   G_{idx}: {len(G)} cadenas | pseudocodigo: {s1['circuits']} circuitos "
              f"({s1['terms_qwc']} cadenas en QWC) | con compresion: {s2['circuits']}")
        marca(d1 < 1e-10 and d2 < 1e-10, f"   gradiente por el plan == <psi|G|psi> ({max(d1, d2):.1e})")

    # =================================================================
    print("\n7b. 8 orbitales con H de dos cuerpos completo: aparecen terminos de 3 cuerpos")
    # =================================================================
    # Con 6 orbitales (6Be) G_mu solo tiene 0, 2 y 4 activos. A partir de 7
    # aparecen cadenas de 6 activos (conmutador de dos cuerpos con dos cuerpos
    # compartiendo un orbital), que el pseudocodigo manda a QWC.
    n = 8
    piezas, pesos = [], []
    for i, j in itertools.combinations(range(n), 2):
        piezas.append(hopping(i, j)); pesos.append(rng.normal())
    for i, j, k, l in itertools.permutations(range(n), 4):
        if i < j and k < l:
            piezas.append(dos_cuerpos(i, j, k, l)); pesos.append(rng.normal())
    H8 = ps_add(*piezas, pesos=pesos)
    psi = estado_aleatorio(n, rng)
    G = build_gradient_operator(H8, generador(0, 1, 2, 3))
    cuenta = {}
    for c in G.terms:
        a = sum(p in "XY" for _, p in c)
        cuenta[a] = cuenta.get(a, 0) + 1
    v = esperado(G, psi, n)
    p1 = build_gradient_measurement_plan(G)
    p2 = build_gradient_measurement_plan(G, compression_fallback=True)
    print(f"   G_(0,1,2,3): {len(G)} cadenas, por n. de activos {dict(sorted(cuenta.items()))}")
    print(f"   pseudocodigo:   {p1.summary()['circuits']:>4} circuitos ({p1.summary()['qwc']} QWC)")
    print(f"   con compresion: {p2.summary()['circuits']:>4} circuitos ({p2.summary()['qwc']} QWC)")
    d = max(abs(p1.exact_expectation(psi, n) - v), abs(p2.exact_expectation(psi, n) - v))
    marca(d < 1e-10, f"ambos planes dan el valor exacto ({d:.1e})")

    # =================================================================
    print("\n8. Ocho indices (seccion 12)")
    # =================================================================
    # Orbitales intercalados para que haya cadenas JW entre ellos.
    n = 10
    h = dos_cuerpos(0, 3, 5, 8)
    T = producto(jw(1, True), jw(4, True), jw(6, False), jw(9, False))
    T = ps_add(T, ps_dag(T), pesos=[1.0, -1.0])          # tau - tau^dag
    G = build_gradient_operator(h, ps_mul(PauliSum({(): 1j}), T))
    activos_G = {len([1 for _, p in c if p in "XY"]) for c in G.terms}
    marca(not G.terms, f"[h_ijkl, T_pq^rs] = 0 con 8 orbitales distintos "
                       f"({len(G)} cadenas; activos presentes: {sorted(activos_G) or 'ninguno'})")

    prod = ps_mul(ps_mul(PauliSum({(): 1j}), h), T)       # i h T, hermitico
    act8 = tuple(sorted({q for c in prod.terms for q, p in c if p in "XY"}))
    print(f"   producto i h T: {len(prod)} cadenas, {len(act8)} activos {act8}")
    plan8 = build_gradient_measurement_plan(prod)
    s = plan8.summary()
    print(f"   plan: {s}")
    psi = estado_aleatorio(n, rng)
    d = abs(plan8.exact_expectation(psi, n) - esperado(prod, psi, n))
    marca(s.get("gradient_8", 0) >= 1, "algun bucket de 8 activos aceptado con la compresion")
    marca(d < 1e-10, f"valor exacto correcto ({d:.1e})")
    for g in plan8.groups:
        if g.kind == "gradient_8":
            diag = g.diagonal_values()
            print(f"   grupo gradient_8: {len(g.terms)} cadenas, eje "
                  f"{'Y' if ('SDG', g.active[0]) in g.gates else 'X'}, "
                  f"{np.count_nonzero(np.abs(diag) > 1e-12)} de {diag.size} entradas de D no nulas")

    # =================================================================
    print("\n8b. Fig. 8 del articulo: formula explicita del gradiente de 8 indices")
    # =================================================================
    # Circuito reconstruido a partir de los puntos de control de la Fig. 8
    # (la extraccion de texto del PDF pierde los targets). Comprime h_ijkl en
    # el qubit i y T_pq^rs en el qubit p, y CX_ip Rx_i CX_ip diagonaliza X_i Y_p.
    # Indices contiguos (i..s = 0..7), como en la formula del articulo (P = 1).
    i, j, k, l, p, q, r, s = range(8)
    n = 8
    h = dos_cuerpos(i, j, k, l)
    Tpool = generador(p, q, r, s)                 # T_pq^rs = i(a+a+aa - h.c.)
    hT = ps_mul(h, Tpool)

    # (a) El producto JW es exactamente la ec. (25) del suplemento.
    def ket(cadena):
        v = np.zeros(2 ** n, dtype=complex)
        v[int(cadena, 2)] = 1.0
        return v
    A, B, C, D = (ket(c) for c in ("00110011", "11001100", "11000011", "00111100"))
    ec25 = (1j * np.outer(B, A) - 1j * np.outer(A, B)
            + 1j * np.outer(D, C) - 1j * np.outer(C, D))
    d = float(np.max(np.abs(densa(hT, list(range(n))) - ec25)))
    marca(d < 1e-12, f"h_0123 T_4567 (JW) == ec. (25) del articulo ({d:.1e})")

    def fig8(angulo):
        return [("CNOT", i, j), ("CNOT", k, l), ("CNOT", i, k),
                ("CNOT", p, q), ("CNOT", r, s), ("CNOT", p, r),
                ("CNOT", i, p), ("RX", i, angulo), ("CNOT", i, p)]

    formula = {"00100010": -1.0, "00101010": +1.0, "10100010": -1.0, "10101010": +1.0}
    activos = tuple(range(n))
    from vqd_nuclear.measurement_plan import MeasurementGroup
    for angulo, esperado_signo in ((-np.pi / 2, +1), (+np.pi / 2, -1)):
        bloques, norma = diagonalize_bucket(dict(hT.terms), activos, activos, fig8(angulo))
        g = MeasurementGroup("paper_8", activos, activos, fig8(angulo),
                             dict(hT.terms), bloques, norma)
        diag = g.diagonal_values()
        no_nulos = {format(x, "08b"): round(float(diag[x]), 9)
                    for x in range(2 ** n) if abs(diag[x]) > 1e-12}
        objetivo = {c: esperado_signo * v for c, v in formula.items()}
        nombre = "Rx(-pi/2)" if angulo < 0 else "Rx(+pi/2)"
        marca(norma < 1e-10, f"{nombre}: diagonaliza h T (||offdiag D|| = {norma:.1e})")
        if esperado_signo > 0:
            marca(no_nulos == objetivo,
                  f"{nombre}: diag(D) == -p00100010 + p00101010 - p10100010 + p10101010")
        else:
            marca(no_nulos == objetivo,
                  f"{nombre}: los cuatro signos invertidos (la diferencia V vs V^dag)")

    # (b) La formula, escrita a mano con probabilidades, da <psi|h T|psi>.
    psi = estado_aleatorio(n, rng)
    probs = np.abs(apply_circuit(psi, fig8(-np.pi / 2), n)) ** 2
    por_formula = sum(signo * probs[int(c, 2)] for c, signo in formula.items())
    v = esperado(hT, psi, n)
    marca(abs(por_formula - v) < 1e-12,
          f"formula del articulo sobre un estado aleatorio == <psi|hT|psi> "
          f"({por_formula:+.6f} vs {v:+.6f})")

    # =================================================================
    if args.con_motor:
        print("\n9. 6Be real: plan de la energia y de cada G_mu del pool")
        # =============================================================
        from vqd_nuclear.config import ProblemConfig, RunConfig, NumericalConfig
        from vqd_nuclear.vqd import build_context
        from vqd_nuclear import measurements as M

        problem = ProblemConfig(shell=args.shell, n_valence_protons=args.protons,
                                n_valence_neutrons=args.neutrons, total_j=args.J)
        ctx = build_context(problem, RunConfig(n_states=2, n_workers=1), NumericalConfig(n_jobs=1))
        eng = ctx.measurements
        nq = ctx.hamiltonian.n_qubits
        Hq = ctx.hamiltonian.qubit_hamiltonian
        ops = [int(i) for i in ctx.active_operator_indices[:2]]
        psi = eng.ansatz_builder.state([0.31, -0.17], ops)

        plan = build_energy_measurement_plan(Hq)
        ref = float(np.real(np.vdot(psi, M.apply_qubit_operator_to_state(Hq, psi, nq))))
        d = abs(plan.exact_expectation(psi, nq) - ref)
        print(f"   H: {len(Hq.terms)} cadenas -> {plan.summary()}")
        marca(d < 1e-10, f"energia por el plan == statevector ({d:.1e})")

        peor = 0.0
        print(f"   {'op':>4} {'cadenas':>8} {'circ.':>6} {'QWC':>4} {'rechazados':>11}")
        for mu in ctx.active_operator_indices:
            G = build_gradient_operator(Hq, ctx.hamiltonian.qubit_generator_pool[int(mu)])
            if not G:
                print(f"   {mu:>4}  G_mu vacio")
                continue
            p = build_gradient_measurement_plan(G)
            ref = float(np.real(np.vdot(psi, M.apply_qubit_operator_to_state(G, psi, nq))))
            peor = max(peor, abs(p.exact_expectation(psi, nq) - ref))
            s = p.summary()
            print(f"   {mu:>4} {len(G):>8} {s['circuits']:>6} {s['qwc']:>4} {len(p.rejected):>11}")
        marca(peor < 1e-10, f"todos los gradientes por el plan == statevector ({peor:.1e})")
    else:
        print("\n9. (omitido; usa --con-motor para 6Be real)")

    print("\n" + "=" * 60)
    print("  Todo pasa." if _fallos == 0 else f"  {_fallos} comprobacion(es) fallan.")
    return 0 if _fallos == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
