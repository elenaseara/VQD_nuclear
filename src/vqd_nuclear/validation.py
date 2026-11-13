"""Validation helpers that do not require Qibo or pynucshell imports."""
from __future__ import annotations

from typing import Any
import numpy as np


def beta_from_largest_requested_gap(exact_energies: np.ndarray, n_states: int) -> float:
    """Return a VQD penalty large enough for all requested states.

    The deflation penalty must be larger than the energy separation between
    the ground state and any lower state that could otherwise be selected again.
    For multi-state VQD runs this uses

        beta = 2 * max_i |E_i - E_0|,

    where the maximum is taken over the exact energies of the requested states.
    This avoids the failure mode in which a higher excited-state calculation
    collapses back to the ground state because beta = 2*|E1 - E0| is too small.
    """
    exact_energies = np.asarray(exact_energies, dtype=float)
    n_requested = int(n_states)

    if n_requested < 2:
        return 0.0

    if len(exact_energies) < n_requested:
        raise ValueError(
            f"At least {n_requested} exact energies are required to set beta "
            "from the largest requested excitation gap."
        )

    ground_energy = float(exact_energies[0])
    requested_energies = exact_energies[:n_requested]
    gaps = np.abs(requested_energies - ground_energy)
    max_gap = float(np.max(gaps))

    if not np.isfinite(max_gap):
        raise ValueError("The requested excitation gaps are not finite, so beta cannot be determined.")

    return float(2.0 * max_gap)


def beta_from_first_excitation_gap(exact_energies: np.ndarray, n_states: int) -> float:
    """Return the legacy two-state penalty beta = 2*|E1 - E0|.

    This public helper is kept for backward compatibility and for tests that
    explicitly validate the original two-state rule. The workflow itself uses
    :func:`beta_from_largest_requested_gap` through :func:`resolve_beta`.
    """
    exact_energies = np.asarray(exact_energies, dtype=float)

    if int(n_states) < 2:
        return 0.0

    if len(exact_energies) < 2:
        raise ValueError("At least two exact energies are required to set beta = 2*max_i abs(E_i - E_0).")

    gap = abs(float(exact_energies[1]) - float(exact_energies[0]))

    if not np.isfinite(gap):
        raise ValueError("The exact first excitation gap is not finite, so beta cannot be determined.")

    return float(2.0 * gap)


def resolve_beta(requested_beta, exact_energies, n_states, basis_beta: dict | None = None) -> dict[str, Any]:
    """Resolve the beta used by the workflow.

    ``requested_beta`` may be ``None`` or ``"auto"`` to derive beta from the
    exact spectrum, or an explicit non-negative number, which is respected.
    An explicit value below the largest requested gap is flagged as unsafe but
    is NOT overridden.
    """
    exact_energies = np.asarray(exact_energies, dtype=float)
    n_requested = int(n_states)

    is_auto = requested_beta is None or (
        isinstance(requested_beta, str) and 
            (requested_beta.strip().lower() == "auto" or requested_beta.strip().lower() == "basis")
    )
    requested = None if is_auto else float(requested_beta)

    if requested is not None and (requested < 0.0 or not np.isfinite(requested)):
        raise ValueError(f"beta must be a finite non-negative number, got {requested_beta!r}.")

    if n_requested < 2:
        return {
            "beta": 0.0,
            "requested_beta": "auto" if is_auto else requested,
            "status": "not_needed",
            "gap_01": None,
            "max_requested_gap": None,
            "auto_beta": None,
            "warning": None,
            "formula": "beta is not used for a single-state VQE/VQD run",
            "message": "beta is not used because only one state was requested.",
        }

    if isinstance(requested_beta, str) and requested_beta.strip().lower() == "basis":
        if basis_beta is None:
            raise ValueError("beta='basis' requiere basis_beta (resultado de get_beta).")
        # Solo para  informar: el hueco exacto no interviene en el calculo de beta.
        max_gap_exacto = float(np.max(np.abs(exact_energies[:n_requested] - exact_energies[0])))
        beta = float(basis_beta["beta"])
        return {
            "beta": beta,
            "requested_beta": "basis",
            "status": "basis_diagonal_gap_plus_offdiag",
            "i": int(basis_beta["i"]), "k": int(basis_beta["k"]),
            "E_i": basis_beta["E_i"], "E_k": basis_beta["E_k"],
            "diagonal_gap": basis_beta["diagonal_gap"],
            "B": basis_beta["B"], "B_row": int(basis_beta["B_row"]),
            "gap_01": abs(float(exact_energies[1]) - float(exact_energies[0])),
            "max_requested_gap": max_gap_exacto,
            "warning": None,
            "formula": basis_beta["formula"],
            "message": (f"beta = ({basis_beta['E_i']:.6g} - {basis_beta['E_k']:.6g}) "
                        f"+ 2*{basis_beta['B']:.6g} = {beta:.12g}"),
        }

    auto_beta = beta_from_largest_requested_gap(exact_energies, n_requested)
    gap_01 = abs(float(exact_energies[1]) - float(exact_energies[0]))
    max_requested_gap = float(
        np.max(np.abs(exact_energies[:n_requested] - float(exact_energies[0])))
    )

    if is_auto:
        return {
            "beta": auto_beta,
            "requested_beta": "auto",
            "status": "auto_from_largest_requested_gap",
            "gap_01": gap_01,
            "max_requested_gap": max_requested_gap,
            "auto_beta": auto_beta,
            "warning": None,
            "formula": "beta = 2*max_i(abs(E_i - E_0)) for requested states",
            "message": (
                f"beta was set automatically to 2*max_i(|Ei - E0|) = {auto_beta:.12g} "
                f"using the largest requested gap {max_requested_gap:.12g}."
            ),
        }

    warning = None
    if requested < max_requested_gap:
        warning = (
            f"beta = {requested:.12g} is below the largest requested gap "
            f"{max_requested_gap:.12g}. VQD requires beta_i > E_k - E_i, so an excited "
            "state may collapse onto a lower one. Check the final overlaps."
        )

    return {
        "beta": requested,
        "requested_beta": requested,
        "status": "user_supplied",
        "gap_01": gap_01,
        "max_requested_gap": max_requested_gap,
        "auto_beta": auto_beta,
        "warning": warning,
        "formula": "beta supplied explicitly by the user",
        "message": (
            f"beta = {requested:.12g} was supplied explicitly "
            f"(the automatic value would have been {auto_beta:.12g})."
            + (f" WARNING: {warning}" if warning else "")
        ),
    }


def validate_beta(beta: float | None, exact_energies: np.ndarray, n_states: int) -> dict[str, Any]:
    """Backward-compatible alias for beta resolution."""
    return resolve_beta(beta, exact_energies, n_states)


def determinant_basis_matrix(references, qubit_hamiltonian, n_qubits: int) -> np.ndarray:
    """Matriz H_jl = <b_j|H|b_l> en la base de determinantes `references`.
 
    Para cada cadena P y cada determinante b_k: P|b_k> = fase * |b_k XOR m_P>,
    con m_P los qubits con X o Y, y
        fase = i^(numero de Y) * (-1)^(paridad de b_k en los qubits con Y o Z).
    Si b_k XOR m_P no es un determinante de la base, la contribucion se
    descarta: H conserva N, M y T_z, asi que esas contribuciones se cancelan
    entre cadenas y no forman parte de la matriz del espacio del modelo.
 
    Coste: (cadenas) x (determinantes), vectorizado sobre los determinantes.
    """
    n = int(n_qubits)
    d = len(references)
    estados = np.array([sum(1 << (n - 1 - int(q)) for q in occ) for occ in references],
                       dtype=np.int64)
    orden = np.argsort(estados)
    ordenados = estados[orden]
 
    H = np.zeros((d, d), dtype=complex)
    columnas = np.arange(d)
    for cadena, coef in qubit_hamiltonian.terms.items():
        coef = complex(coef)
        if coef == 0:
            continue
        m_flip = m_signo = 0
        n_y = 0
        for q, p in cadena:
            bit = 1 << (n - 1 - int(q))
            p = str(p)
            if p == "X":
                m_flip |= bit
            elif p == "Y":
                m_flip |= bit
                m_signo |= bit
                n_y += 1
            elif p == "Z":
                m_signo |= bit
            elif p != "I":
                raise ValueError(f"Letra de Pauli desconocida: {p}")
 
        destino = estados ^ m_flip
        pos = np.searchsorted(ordenados, destino)
        pos = np.minimum(pos, d - 1)
        dentro = ordenados[pos] == destino
        if not np.any(dentro):
            continue
        filas = orden[pos[dentro]]
        paridad = np.array([bin(int(x)).count("1") & 1
                            for x in (estados[dentro] & m_signo)], dtype=np.int64)
        fase = (1j ** n_y) * (1 - 2 * paridad)
        np.add.at(H, (filas, columnas[dentro]), coef * fase)
 
    herm = float(np.max(np.abs(H - H.conj().T))) if d else 0.0
    if herm > 1e-9:
        raise ValueError(f"La matriz en la base de determinantes no es hermitica ({herm:.2e}).")
    return H
 
def diagonal_gap_term(H_basis: np.ndarray, n_states: int) -> dict:
    """Primer termino de beta: E_i - E_k con los elementos diagonales ordenados.
 
    i = estado excitado mas alto simulado (n_states - 1, contando desde 0),
    k = 0.
    """
    n_states = int(n_states)
    diagonal = np.sort(np.real(np.diag(H_basis)))
    d = len(diagonal)
    if n_states < 2:
        raise ValueError("La beta solo tiene sentido con n_states >= 2.")
    if n_states > d:
        raise ValueError(f"Se piden {n_states} estados y el espacio del modelo tiene {d} determinantes.")
    i = n_states - 1
    k = 0
    return {
        "diagonal_sorted": diagonal,
        "i": i,
        "k": k,
        "E_i": float(diagonal[i]),
        "E_k": float(diagonal[k]),
        "diagonal_gap": float(diagonal[i] - diagonal[k]),
    }
 
def off_diagonal_bound(H_basis: np.ndarray) -> dict:
    """
    Segundo termino: B = max_j sum_{k != j} |H_jk|.
    """
    off = H_basis - np.diag(np.diag(H_basis))
    filas = np.abs(off).sum(axis=1)
    j = int(np.argmax(filas))
    return {"row_sums": filas, "B_row": j, "B": float(filas[j])}
 
 
def get_beta(references, qubit_hamiltonian, n_qubits: int, n_states: int) -> dict:
    """Beta a partir de la matriz en la base de determinantes.
 
    El cálculo es: Beta = E_i - E_0 + 2B; siendo B el mayor valor de la suma por filas
    de todos los elementos no diagonales de H_jl.

    """
    H = determinant_basis_matrix(references, qubit_hamiltonian, n_qubits)
    diagonal = np.diag(np.diag(H))
    off_diagonal = H - diagonal
    termino_1 = diagonal_gap_term(H, n_states)
    termino_2 = off_diagonal_bound(H)
    beta = termino_1["diagonal_gap"] + 2.0 * termino_2["B"]

    return {
        "H_basis": H,
        "H_diagonal": diagonal,
        "H_off_diagonal": off_diagonal,
        **termino_1,
        **termino_2,
        "beta": beta,
        "formula": "beta = (E_i - E_0) + 2B;"
                    "i = n_states-1; B = max_j sum_{k!=j} |H_jk|"
    }
 
