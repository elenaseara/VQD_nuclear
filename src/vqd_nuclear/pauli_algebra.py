"""Ruta muestreada: algebra de Pauli y observable del gradiente.

Modulo del paquete: src/vqd_nuclear/pauli_algebra.py.
Nombres tomados del pseudocodigo:
    PAULI_COMMUTE            -> seccion 10.1
    BUILD_GRADIENT_OPERATOR  -> seccion 10.2

CORRECCION DE SIGNO RESPECTO AL PSEUDOCODIGO
--------------------------------------------
La seccion 10 define G_mu = i[H,A] y la 10.2 lo implementa con
coefficient = 2*i*h*a*phase. Eso contradice la seccion 4.2 (|d psi> = -i A|psi>),
que es la convencion VERIFICADA contra el circuito real en verify_statevector.py.

    seccion 4.2  =>  dE/dtheta = 2 Re(-i z) = +2 Im z        (con z = <H psi|A psi>)
    seccion 10   =>  <psi|i[H,A]|psi> = i(z - z*) = -2 Im z

Aqui se usa G_mu = ANSATZ_DERIVATIVE_PHASE * [H, A], que reproduce la seccion 10
si la constante vale +1j y la 4.2 (la correcta) si vale -1j. Una sola constante
gobierna commutator_energy_gradient, penalty_gradient y build_gradient_operator,
de modo que no pueden quedar en convenciones distintas.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Fuente unica de la convencion del ansatz: la misma que usan
# commutator_energy_gradient, penalty_gradient y la propagacion proyectada.
from .conventions import ANSATZ_DERIVATIVE_PHASE


# =====================================================================
# Representacion
# =====================================================================
#
# Una cadena de Pauli es una tupla ORDENADA de pares (qubit, letra), con las
# letras en {"X","Y","Z"} y sin entradas "I". La identidad es la tupla vacia ().
# Es la misma convencion que usa qubit_hamiltonian.terms.


@dataclass(slots=True)
class PauliSum:
    """Suma de cadenas de Pauli con coeficientes.

    Expone .terms para ser intercambiable con qubit_hamiltonian en todo lo que
    ya lee ese atributo (apply_qubit_operator_to_state, los planes de medida).
    """

    terms: dict[tuple, complex] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.terms)

    def __bool__(self) -> bool:
        return bool(self.terms)


# =====================================================================
# A.1  Producto de dos cadenas de Pauli
# =====================================================================

# (fase, letra_resultante) para cada par de letras de un mismo qubit.
_PRODUCTO_1Q: dict[tuple[str, str], tuple[complex, str]] = {
    ("X", "X"): (1.0 + 0j, "I"), ("Y", "Y"): (1.0 + 0j, "I"), ("Z", "Z"): (1.0 + 0j, "I"),
    ("X", "Y"): (1j, "Z"),  ("Y", "X"): (-1j, "Z"),
    ("Y", "Z"): (1j, "X"),  ("Z", "Y"): (-1j, "X"),
    ("Z", "X"): (1j, "Y"),  ("X", "Z"): (-1j, "Y"),
}


def multiply_pauli_strings(p: tuple, q: tuple) -> tuple[complex, tuple]:
    """Devuelve (fase, R) tal que P Q = fase * R.

    R es una cadena de Pauli normalizada (ordenada, sin identidades) y la fase
    es una de {+1, -1, +i, -i}.
    """
    mapa_p = dict(p)
    mapa_q = dict(q)
    fase: complex = 1.0 + 0j
    resultado: dict[int, str] = {}

    for qubit in set(mapa_p) | set(mapa_q):
        letra_p = mapa_p.get(qubit)
        letra_q = mapa_q.get(qubit)
        if letra_p is None:
            resultado[qubit] = letra_q
            continue
        if letra_q is None:
            resultado[qubit] = letra_p
            continue
        fase_local, letra = _PRODUCTO_1Q[(letra_p, letra_q)]
        fase *= fase_local
        if letra != "I":
            resultado[qubit] = letra

    return fase, tuple(sorted(resultado.items()))


# =====================================================================
# A.2  Conmutacion de Pauli  (pseudocodigo 10.1)
# =====================================================================

def pauli_commute(p: tuple, q: tuple) -> bool:
    """True si [P,Q] = 0.

    Dos cadenas de Pauli conmutan si el numero de qubits en los que ambas son
    no-identidad Y DISTINTAS es par; anticonmutan si es impar. No hay otro caso:
    dos cadenas de Pauli o conmutan o anticonmutan.

    NO confundir con qwc_commutes. Son preguntas distintas:
        pauli_commute  -> "la pareja (P,Q), contribuye al gradiente?"
        qwc_commutes   -> "P y Q se pueden medir con el mismo circuito?"
    Difieren en aproximadamente el 26% de los pares a 3 qubits.
    """
    mapa_q = dict(q)
    n_anticonmutan = 0
    for qubit, letra_p in p:
        letra_q = mapa_q.get(qubit)
        if letra_q is not None and letra_q != letra_p:
            n_anticonmutan += 1
    return n_anticonmutan % 2 == 0


# =====================================================================
# B  Observable del gradiente  (pseudocodigo 10.2, con el signo corregido)
# =====================================================================

def build_gradient_operator(
    qubit_hamiltonian,
    generator,
    *,
    tolerance: float = 1e-12,
    check_hermitian: bool = True,
) -> PauliSum:
    """G_mu = ANSATZ_DERIVATIVE_PHASE * [H_q, A_mu,q], de modo que

        g^E_mu = <psi| G_mu |psi>.

    G_mu es hermitico, asi que su valor esperado es medible con shots igual que
    una energia. Es lo que hace posible ADAPT en hardware.

    Solo contribuyen las parejas (P,Q) que ANTICONMUTAN: si conmutan,
    [P,Q] = 0. Si anticonmutan, [P,Q] = 2 P Q.

    Un G_mu vacio significa [H, A_mu] = 0 exactamente: el candidato no cambia la
    energia y se puede descartar sin gastar un solo shot. OJO: eso no implica
    que su g^penalty sea cero, de modo que en VQD statevector ese candidato
    sigue siendo legitimo.
    """
    acumulado: dict[tuple, complex] = {}

    for cadena_h, coeficiente_h in qubit_hamiltonian.terms.items():
        for cadena_a, coeficiente_a in generator.terms.items():
            if pauli_commute(cadena_h, cadena_a):
                continue
            fase, cadena_r = multiply_pauli_strings(cadena_h, cadena_a)
            # [P,Q] = 2 P Q = 2 * fase * R  (solo si anticonmutan)
            contribucion = (2.0 * ANSATZ_DERIVATIVE_PHASE
                            * complex(coeficiente_h) * complex(coeficiente_a) * fase)
            acumulado[cadena_r] = acumulado.get(cadena_r, 0.0 + 0j) + contribucion

    # Cancelaciones entre parejas distintas que dan la misma cadena.
    limpio = {cadena: c for cadena, c in acumulado.items() if abs(c) > tolerance}

    if check_hermitian:
        # Las cadenas de Pauli son hermiticas e independientes, asi que G es
        # hermitico si y solo si todos sus coeficientes son reales.
        peor = max((abs(c.imag) for c in limpio.values()), default=0.0)
        if peor > 1e-9:
            raise ValueError(
                f"G_mu no es hermitico: parte imaginaria maxima {peor:.3e}. "
                "Revisa multiply_pauli_strings o la convencion de fase."
            )
        limpio = {cadena: complex(c.real, 0.0) for cadena, c in limpio.items()}

    return PauliSum(terms=limpio)
