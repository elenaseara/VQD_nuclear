#!/usr/bin/env python3
"""Comprobacion 2, CORREGIDA: coherencia de Jordan-Wigner.

La version anterior comparaba el espectro del sector de numero de particulas
(dimension C(n, A)) contra exact_energies, indice a indice. Mal: ese sector es
la UNION de los bloques de M, su espectro intercala autovalores de bloques
distintos, y su dimension no es la del espacio del modelo.

Aqui se proyecta JW(H) sobre el span de los determinantes que usa el codigo
(get_hamiltonian_basis_references) y se compara contra el espectro COMPLETO de
la CSR. Mismo espacio, misma dimension, sin ambiguedad. De paso valida el
indexado de los determinantes: si basis_state_to_occupied_orbitals tuviera el
endianness invertido, el subespacio proyectado seria otro y el espectro no
cuadraria.

Uso:
    python check_jw_block.py --shell p --protons 2 --neutrons 0 --J 0
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shell", choices=["p", "sd"], default="p")
    ap.add_argument("--protons", type=int, default=2)
    ap.add_argument("--neutrons", type=int, default=0)
    ap.add_argument("--J", type=int, default=0)
    args = ap.parse_args()

    from vqd_nuclear.config import ProblemConfig, RunConfig, NumericalConfig
    from vqd_nuclear.vqd import build_context
    from vqd_nuclear.ansatz import get_hamiltonian_basis_references
    from vqd_nuclear import measurements as M

    problem = ProblemConfig(shell=args.shell, n_valence_protons=args.protons,
                            n_valence_neutrons=args.neutrons, total_j=args.J)
    ctx = build_context(problem, RunConfig(n_states=2, n_workers=1),
                        NumericalConfig(n_jobs=1))

    n = ctx.hamiltonian.n_qubits
    n_part = args.protons + args.neutrons
    dim_csr = int(ctx.hamiltonian.matrix.shape[0])

    print(f"\n{args.shell}-shell  Z_val={args.protons} N_val={args.neutrons} J={args.J}")
    print(f"n_qubits = {n}   dim(CSR) = {dim_csr}")

    # --- 1. los determinantes del espacio del modelo --------------------
    refs = get_hamiltonian_basis_references(
        ctx.hamiltonian.nucleus, n_qubits=n, n_valence_particles=n_part)
    print(f"determinantes extraidos = {len(refs)}")
    if len(refs) != dim_csr:
        print(" [FALLO] no coinciden con la dimension de la CSR; para aqui.")
        return 1

    # Indice entero de cada determinante en la base computacional.
    # Mismo convenio de bits que apply_pauli_string_to_state: el orbital q
    # pesa 1 << (n - 1 - q).
    indices = []
    for ref in refs:
        occupied = getattr(ref, "occupied_orbitals", ref)
        idx = 0
        for q in occupied:
            idx |= 1 << (n - 1 - int(q))
        indices.append(idx)

    if len(set(indices)) != len(indices):
        print(" [FALLO] dos determinantes colapsan en el mismo indice:")
        print("         endianness invertida en la conversion a orbitales.")
        return 1
    if any(bin(i).count("1") != n_part for i in indices):
        print(f" [FALLO] algun determinante no tiene {n_part} orbitales ocupados.")
        return 1

    # --- 2. proyeccion de JW(H) sobre ese span --------------------------
    # H_proj[a,b] = <det_a| JW(H) |det_b>.  Una aplicacion por determinante,
    # asi que el coste va con dim_csr, no con 2^n: esto tambien corre en sd.
    d = len(indices)
    H_proj = np.zeros((d, d), dtype=complex)
    vec = np.zeros(2 ** n, dtype=complex)
    for b, idx_b in enumerate(indices):
        vec[:] = 0.0
        vec[idx_b] = 1.0
        h_vec = M.apply_qubit_operator_to_state(
            ctx.hamiltonian.qubit_hamiltonian, vec, n)
        for a, idx_a in enumerate(indices):
            H_proj[a, b] = h_vec[idx_a]

    herm = float(np.max(np.abs(H_proj - H_proj.conj().T)))
    print(f"\nhermiticidad del bloque proyectado : {herm:.2e}")

    # Fuga fuera del subespacio: si JW(H) saca amplitud del span de los
    # determinantes, el espacio del modelo no es invariante bajo H y el
    # ansatz podria salirse de el.
    fuga_max = 0.0
    mascara = np.ones(2 ** n, dtype=bool)
    mascara[indices] = False
    for idx_b in indices:
        vec[:] = 0.0
        vec[idx_b] = 1.0
        h_vec = M.apply_qubit_operator_to_state(
            ctx.hamiltonian.qubit_hamiltonian, vec, n)
        fuga_max = max(fuga_max, float(np.max(np.abs(h_vec[mascara]))))
    print(f"fuga fuera del subespacio          : {fuga_max:.2e}")
    if fuga_max > 1e-10:
        print("   AVISO: H conecta el espacio del modelo con estados de fuera.")
        print("          Revisa si el sector esta bien definido (M, Tz).")

    # --- 3. comparacion de espectros ------------------------------------
    vals_jw = np.linalg.eigvalsh(H_proj)
    csr = ctx.hamiltonian.matrix
    dense = csr.toarray() if hasattr(csr, "toarray") else np.asarray(csr)
    vals_csr = np.linalg.eigvalsh(np.asarray(dense, dtype=complex))

    print(f"\n{'':4} {'JW proyectado':>18} {'CSR completa':>18} {'diferencia':>14}")
    ok = True
    for i in range(d):
        dif = abs(vals_jw[i] - vals_csr[i])
        ok &= dif < 1e-8
        print(f"  E{i}  {vals_jw[i]:>18.9f} {vals_csr[i]:>18.9f} {dif:>14.2e}")

    print()
    if ok:
        print("  [OK] JW(H) restringido al espacio del modelo reproduce la CSR.")
        print("       El mapeo, la ordenacion de qubits y el indexado de los")
        print("       determinantes son coherentes.")
    else:
        print("  [FALLO] los espectros no coinciden en el MISMO espacio.")
        print("          Ahora si es un problema real del mapeo o del indexado.")

    # --- 4. que es exact_energies ---------------------------------------
    # Util saberlo explicitamente: si exact_energies no es un prefijo del
    # espectro de la CSR, viene de otra diagonalizacion (p.ej. proyectada en J)
    # y el objetivo de VQD no es lo que se compara al final.
    exactas = np.asarray(ctx.hamiltonian.exact_energies, dtype=float)
    k = min(len(exactas), d)
    print(f"\nexact_energies ({len(exactas)} valores) : {np.round(exactas[:k], 6)}")
    print(f"espectro CSR (primeros {k})        : {np.round(vals_csr[:k].real, 6)}")
    prefijo = bool(np.allclose(exactas[:k], vals_csr[:k].real, atol=1e-8))
    print(f"exact_energies es el prefijo del espectro de la CSR: {prefijo}")
    if not prefijo:
        print("   OJO: entonces exact_energies NO sale de diagonalizar la CSR.")
        print("        VQD minimiza sobre el espacio de la CSR, asi que puede")
        print("        converger a un estado que no esta en la lista de")
        print("        referencia. Averigua de donde sale exact_energies antes")
        print("        de dar por buenos los errores de energia.")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
