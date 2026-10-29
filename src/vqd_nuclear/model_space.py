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

import logging
from operator import index as integer_index
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

LOGGER = logging.getLogger(__name__)

# Debe ser la MISMA constante que measurements.py
ANSATZ_DERIVATIVE_PHASE = -1j

LEAK_TOL = 1e-10


# =====================================================================
# 1. Indices del espacio del modelo
# =====================================================================

def model_space_indices(references, n_qubits: int) -> np.ndarray:
    """Indice en la base computacional de cada determinante del modelo.

    Convenio de bits identico al de apply_pauli_string_to_state: el orbital q
    pesa 1 << (n_qubits - 1 - q).

    El ORDEN de esta lista define la base del subespacio y se usa en todo lo
    demas. No se toma de la CSR de la libreria a proposito: su orden de filas lo
    decide la libreria y no tiene por que coincidir. Que los espectros
    coincidieran en check_jw_block no demuestra lo contrario, porque los
    autovalores son invariantes bajo permutacion.
    """
    indices = []
    for referencia in references:
        ocupados = getattr(referencia, "occupied_orbitals", referencia)
        valor = 0
        for q in ocupados:
            valor |= 1 << (int(n_qubits) - 1 - int(q))
        indices.append(valor)
    if len(set(indices)) != len(indices):
        raise ValueError("Dos determinantes colapsan en el mismo indice: "
                         "revisa el convenio de bits.")
    return np.asarray(indices, dtype=np.int64)


def project_operator(qubit_operator, indices: np.ndarray, n_qubits: int,
                     apply_operator) -> tuple[np.ndarray, float]:
    """Devuelve (O_S, fuga).

    O_S[a,b] = <det_a| O |det_b>, en el orden de `indices`.
    fuga = maxima amplitud que O saca FUERA del espacio al actuar sobre un
    determinante. Si no es ~0, el operador no conserva el subespacio y no se
    puede proyectar.

    Aplica las cadenas directamente a los indices computacionales: no reserva
    vectores de longitud 2**n_qubits ni llama a apply_operator. Este argumento
    se conserva para mantener la interfaz de los llamantes existentes.
    """
    try:
        n_qubits = integer_index(n_qubits)
    except TypeError as exc:
        raise ValueError("n_qubits must be a nonnegative integer.") from exc
    if n_qubits < 0:
        raise ValueError("n_qubits must be a nonnegative integer.")
    raw_indices = np.asarray(indices)
    if raw_indices.ndim != 1:
        raise ValueError("Model-space indices must be one-dimensional.")
    try:
        basis_indices = [integer_index(value) for value in raw_indices]
    except TypeError as exc:
        raise ValueError("Model-space indices must be integers.") from exc
    if any(value < 0 or value >= (1 << n_qubits) for value in basis_indices):
        raise ValueError("Model-space index is outside the qubit register.")
    if len(set(basis_indices)) != len(basis_indices):
        raise ValueError("Model-space indices must not contain duplicates.")
    positions = {value: row for row, value in enumerate(basis_indices)}

    # Compile masks once. Y contributes i, a sign from the original bit, and
    # a flip; Z contributes only that sign. Qibo's qubit 0 is the high bit.
    terms = []
    for pauli_string, coefficient in qubit_operator.terms.items():
        coefficient = complex(coefficient)
        if not np.isfinite(coefficient):
            raise ValueError("Pauli coefficients must be finite.")
        flip_mask, sign_mask, n_y = 0, 0, 0
        for qubit, pauli in pauli_string:
            try:
                qubit = integer_index(qubit)
            except TypeError as exc:
                raise ValueError("Pauli qubit indices must be integers.") from exc
            if not 0 <= qubit < n_qubits:
                raise ValueError("Pauli qubit index is outside the qubit register.")
            if pauli not in {"I", "X", "Y", "Z"}:
                raise ValueError(f"Unsupported Pauli operator: {pauli}")
            mask = 1 << (n_qubits - 1 - qubit)
            if pauli in {"X", "Y"}:
                flip_mask ^= mask
            if pauli in {"Y", "Z"}:
                sign_mask ^= mask
            n_y += pauli == "Y"
        terms.append((flip_mask, sign_mask, coefficient * (1j ** n_y)))

    d = len(basis_indices)
    o_s = np.zeros((d, d), dtype=complex)
    fuga = 0.0
    for column, source in enumerate(basis_indices):
        outside = {}
        for flip_mask, sign_mask, coefficient in terms:
            target = source ^ flip_mask
            amplitude = -coefficient if (source & sign_mask).bit_count() % 2 else coefficient
            row = positions.get(target)
            if row is None:
                outside[target] = outside.get(target, 0.0j) + amplitude
            else:
                o_s[row, column] += amplitude
        # Contributions must cancel before measuring leakage, not afterwards.
        if outside:
            fuga = max(fuga, float(max(abs(value) for value in outside.values())))
    return o_s, fuga


# =====================================================================
# 2. Generador proyectado y diagonalizado
# =====================================================================

@dataclass(slots=True)
class ProjectedGenerator:
    """T_S diagonalizado, para evaluar exp(PHASE * theta * T_S) en O(d^2).

    T_S es hermitica (los coeficientes del pool son reales), asi que basta una
    eigh, que se hace UNA vez por operador. Despues, para cualquier theta, la
    exponencial sale de los autovalores sin ninguna exponencial de matriz.
    """

    eigenvalues: np.ndarray        # (d,) reales
    eigenvectors: np.ndarray       # (d,d) unitaria
    preserves_space: bool

    @classmethod
    def from_operator(cls, qubit_operator, indices, n_qubits, apply_operator,
                      leak_tol: float = LEAK_TOL) -> "ProjectedGenerator":
        t_s, fuga = project_operator(qubit_operator, indices, n_qubits, apply_operator)
        asimetria = float(np.max(np.abs(t_s - t_s.conj().T))) if len(indices) else 0.0
        if asimetria > 1e-9:
            raise ValueError(
                f"T_S no es hermitica (asimetria {asimetria:.2e}). El generador "
                "deberia tener coeficientes reales sobre cadenas de Pauli."
            )
        valores, vectores = np.linalg.eigh(t_s)
        return cls(eigenvalues=valores, eigenvectors=vectores,
                   preserves_space=bool(fuga <= leak_tol))

    def apply(self, theta: float, psi_s: np.ndarray) -> np.ndarray:
        """exp(PHASE * theta * T_S) |psi_S>, en dos productos matriz-vector."""
        fases = np.exp(ANSATZ_DERIVATIVE_PHASE * float(theta) * self.eigenvalues)
        return self.eigenvectors @ (fases * (self.eigenvectors.conj().T @ psi_s))

    def action(self, psi_s: np.ndarray) -> np.ndarray:
        """Apply the Hermitian generator A_S, without the evolution phase."""
        return self.eigenvectors @ (self.eigenvalues * (self.eigenvectors.conj().T @ psi_s))

    def effective_generator_action(self, psi_s: np.ndarray) -> np.ndarray:
        """Derivative at zero of exp(-i theta A_S), with the phase once."""
        return ANSATZ_DERIVATIVE_PHASE * self.action(psi_s)


# =====================================================================
# 3. El motor proyectado
# =====================================================================

class ModelSpaceEngine:
    """Propagacion y medida dentro del espacio del modelo.

    `available` es False si H no conserva el subespacio o si falta informacion;
    en ese caso el llamante debe usar el camino completo.
    """

    def __init__(self, *, qubit_hamiltonian, qubit_generator_pool, indices,
                 reference_occupied_orbitals, n_qubits, apply_operator,
                 max_cache_items: int = 64) -> None:
        self.n_qubits = int(n_qubits)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.qubit_generator_pool = qubit_generator_pool
        self._apply = apply_operator
        self._max_cache_items = int(max_cache_items)
        self._cache: OrderedDict = OrderedDict()
        self._generators: dict[int, ProjectedGenerator] = {}

        # --- Hamiltoniano proyectado -------------------------------------
        self.hamiltonian_s, fuga_h = project_operator(
            qubit_hamiltonian, self.indices, self.n_qubits, apply_operator)
        self.available = bool(fuga_h <= LEAK_TOL)
        if not self.available:
            LOGGER.warning(
                "H saca amplitud fuera del espacio del modelo (%.3e). No se "
                "proyecta nada; se usa el camino completo.", fuga_h)

        # --- vector de referencia ----------------------------------------
        valor = 0
        for q in reference_occupied_orbitals:
            valor |= 1 << (self.n_qubits - 1 - int(q))
        posiciones = np.flatnonzero(self.indices == valor)
        if posiciones.size != 1:
            LOGGER.warning(
                "El determinante de referencia no esta en el espacio del modelo. "
                "No se proyecta nada.")
            self.available = False
            self.reference_s = None
        else:
            self.reference_s = np.zeros(len(self.indices), dtype=complex)
            self.reference_s[int(posiciones[0])] = 1.0

    # ---------------------------------------------------------------- utilidades

    def generator(self, operator_index: int) -> ProjectedGenerator:
        """T_S diagonalizado del operador, construido la primera vez que se pide."""
        mu = int(operator_index)
        if mu not in self._generators:
            self._generators[mu] = ProjectedGenerator.from_operator(
                self.qubit_generator_pool[mu], self.indices, self.n_qubits, self._apply)
            if not self._generators[mu].preserves_space:
                LOGGER.warning(
                    "El generador %d NO conserva el espacio del modelo. Los ansatze "
                    "que lo usen iran por el camino completo. Esto merece "
                    "investigarse: es un hallazgo de fisica, no solo de rendimiento.",
                    mu)
        return self._generators[mu]

    def can_handle(self, operator_indices) -> bool:
        """True si se puede propagar este ansatz dentro del subespacio."""
        if not self.available:
            return False
        return all(self.generator(mu).preserves_space for mu in operator_indices)

    @staticmethod
    def _key(thetas, operator_indices, ndigits: int = 12):
        return (tuple(round(float(t), ndigits) for t in thetas),
                tuple(int(i) for i in operator_indices))

    def _cache_set(self, key, value) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self._max_cache_items:
            self._cache.popitem(last=False)

    # ---------------------------------------------------------------- API

    def state(self, thetas, operator_indices) -> np.ndarray:
        """psi_S = exp(PHASE th_L T_L,S) ... exp(PHASE th_1 T_1,S) |ref_S>.

        Sustituye a construir y simular el circuito de Qibo.
        """
        if len(thetas) != len(operator_indices):
            raise ValueError("Each model-space operator requires exactly one parameter.")
        if any(not 0 <= int(mu) < len(self.qubit_generator_pool) for mu in operator_indices):
            raise ValueError("Model-space operator index is outside the pool.")
        key = self._key(thetas, operator_indices)
        cacheado = self._cache.get(key)
        if cacheado is not None:
            self._cache.move_to_end(key)
            return cacheado

        psi_s = self.reference_s.copy()
        for theta, mu in zip(thetas, operator_indices):
            psi_s = self.generator(int(mu)).apply(float(theta), psi_s)

        self._cache_set(key, psi_s)
        return psi_s

    def energy(self, thetas, operator_indices) -> float:
        psi_s = self.state(thetas, operator_indices)
        return float(np.vdot(psi_s, self.hamiltonian_s @ psi_s).real)

    def overlap(self, thetas_i, ops_i, thetas_k, ops_k) -> float:
        """|<psi_i|psi_k>|^2, la misma convencion que exact_overlap."""
        a = self.state(thetas_i, ops_i)
        b = self.state(thetas_k, ops_k)
        return float(abs(np.vdot(a, b)) ** 2)
