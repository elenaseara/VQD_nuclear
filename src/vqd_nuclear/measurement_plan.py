"""Ruta muestreada: planes de medida para la energia y el gradiente.

Modulo del paquete: src/vqd_nuclear/measurement_plan.py
Pseudocodigo: secciones 5 (plan de energia), 6-8 (cambios de base y medida de un
grupo), 11-12 (plan del gradiente) y 13 (QWC residual).

QUE HACE
--------
Recibe un operador escrito como suma de cadenas de Pauli -- el Hamiltoniano H_q
o el observable del gradiente G_mu de la fase B -- y lo reparte en CIRCUITOS DE
MEDIDA. Cada circuito es un cambio de base que se anade detras del ansatz y que
deja diagonal una parte del operador, de modo que su valor esperado se obtiene
de las probabilidades de los bitstrings medidos:

    <O_grupo> = sum_x D_xx P(x)

No ejecuta nada en Qibo. Produce una descripcion de los circuitos (listas de
puertas) y de como interpretar cada bitstring.

EL PROCESO (seccion 5.1 / 11)
-----------------------------
1. Clasificar cada cadena por sus qubits ACTIVOS (los que llevan X o Y): son
   los que el operador cambia de estado. Los Z restantes son "espectadores":
   cadenas de paridad de Jordan-Wigner o factores de numero n_i.
2. Agrupar las cadenas que comparten conjunto activo (un "bucket").
3. Proponer un cambio de base segun cuantos activos tenga el bucket:
       0 activos -> ninguno (ya es diagonal)
       2 activos -> M_jk   del articulo
       4 activos -> M_ijkl del articulo
       8 activos -> compresion por CNOTs (solo en el plan del gradiente)
4. PUERTA DE VALIDACION: calcular D y aceptar el grupo SOLO si
   ||D - diag(diag D)|| < tol. Tener 2 o 4 activos no garantiza que el cambio
   de base valga; esto lo comprueba numericamente en vez de suponerlo.
5. Lo que no pasa la puerta, o no tiene circuito especializado, va a grupos
   QWC (seccion 13): rotaciones de un solo qubit.

DOS DESVIACIONES DELIBERADAS RESPECTO AL CODIGO ANTIGUO
-------------------------------------------------------
(a) D = V O V^dagger, no V^dagger O V.
    V es la unitaria del circuito de medida en el orden en que se aplica. Tras
    el ansatz, P(x) = |<x|V psi>|^2, asi que sum_x D_xx P(x) = <psi|V^dag D V|psi>,
    y para que valga <psi|O|psi> hace falta D = V O V^dag. El codigo antiguo
    calculaba V^dag O V. Coinciden SOLO si V = V^dag, que es el caso de M_jk y
    M_ijkl (palindromos de puertas autoinversas) pero no de la compresion de 8
    indices ni de cualquier circuito con S^dag o Rx. Con la convencion antigua,
    esos daria valores esperados incorrectos sin ningun aviso.

(b) D se calcula sin construir la matriz densa sobre todo el soporte.
    El codigo antiguo construia O_local de 2^|soporte| x 2^|soporte|. Con cadenas
    de paridad largas (soporte de 12 qubits en sd) son 268 MB por bucket. Aqui
    se usa que V solo actua sobre los qubits activos y que en los espectadores
    solo hay Z:
        V P V^dag = (V_A Q V_A^dag) (x) Z_S
    Basta con matrices de 2^|activos| (16x16 para 4 activos), una por cada
    patron de espectadores S. La norma ||D - diag(diag D)|| que se compara es
    EXACTAMENTE la del pseudocodigo, no una aproximacion: los Z_S son
    ortogonales en traza, asi que
        ||offdiag D||_F^2 = 2^m * sum_S ||offdiag B_S||_F^2,  m = |espectadores|.
    El test lo contrasta con la D densa en casos pequenos.

SOBRE EL CASO DE 8 INDICES (seccion 12)
--------------------------------------
El articulo mide productos h_ijkl T_pq^rs con 8 orbitales distintos. Nosotros
medimos el CONMUTADOR G_mu = PHASE [H, A_mu]. Dos operadores fermionicos pares
sobre orbitales disjuntos conmutan exactamente (16 anticonmutaciones, signo +),
y Jordan-Wigner conserva el algebra, asi que [h_ijkl, T_pq^rs] = 0 identicamente.
Las cadenas de 8 activos se cancelan al construir G_mu y NO aparecen.

El circuito de 8 indices se implementa igualmente (specialized_8_index_basis_change)
siguiendo la tesis (X.39-X.54): una red de CNOTs comprime la transicion a un
unico qubit y una rotacion lo diagonaliza. El test comprueba que diagonaliza el
producto h T de la tesis con la estructura de 4 probabilidades del articulo,
y que el conmutador correspondiente es nulo.

CONVENIOS
---------
- Cadena de Pauli: tupla ordenada de (qubit, letra), letras en {X,Y,Z}, sin I.
  Identidad = (). Igual que qubit_hamiltonian.terms.
- Base computacional: el qubit q pesa 1 << (n_qubits - 1 - q). Igual que
  apply_pauli_string_to_state.
- Indice LOCAL de un grupo: el qubit support[0] es el bit mas significativo.
  La fase D debe medir los qubits en el orden de `support` y usar
  `group.local_index_from_bits` o `group.diagonal_value`.
- Puertas: ("CNOT", control, target), ("H", q), ("SDG", q) con SDG = S^dagger,
  ("RX", q, theta) con RX(theta) = exp(-i theta X / 2) (misma convencion que
  qibo.gates.RX). Se listan en el orden de aplicacion.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_TOL = 1e-8
MAX_TABLE_QUBITS = 20     # por encima no se materializa la tabla completa de D

PauliTerm = tuple
Gate = tuple

_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}
_HADAMARD = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2.0)


# =====================================================================
# 1. Simulacion de las puertas de medida (sin Qibo)
# =====================================================================
# Sirve para dos cosas: construir la unitaria V de cada cambio de base y
# calcular el valor esperado EXACTO (limite de shots infinitos) que la fase D
# tendra que reproducir con muestreo.

def _apply_gate(psi: np.ndarray, gate: Gate, n: int) -> np.ndarray:
    """Aplica una puerta a un estado con forma (2,)*n. El eje q es el qubit q."""
    name = gate[0]
    if name == "H":
        q = int(gate[1])
        out = np.tensordot(_HADAMARD, psi, axes=([1], [q]))
        return np.moveaxis(out, 0, q)
    if name == "SDG":
        q = int(gate[1])
        out = psi.copy()
        idx = [slice(None)] * n
        idx[q] = 1
        out[tuple(idx)] *= -1j
        return out
    if name == "RX":
        q, theta = int(gate[1]), float(gate[2])
        rx = np.array([[np.cos(theta / 2), -1j * np.sin(theta / 2)],
                       [-1j * np.sin(theta / 2), np.cos(theta / 2)]], dtype=complex)
        out = np.tensordot(rx, psi, axes=([1], [q]))
        return np.moveaxis(out, 0, q)
    if name == "CNOT":
        c, t = int(gate[1]), int(gate[2])
        if c == t:
            raise ValueError("CNOT con control y target iguales.")
        out = psi.copy()
        idx = [slice(None)] * n
        idx[c] = 1
        eje_t = t if t < c else t - 1          # al fijar el control se pierde un eje
        out[tuple(idx)] = np.flip(out[tuple(idx)], axis=eje_t).copy()
        return out
    raise ValueError(f"Puerta no soportada: {gate}")


def apply_circuit(state: np.ndarray, gates: list[Gate], n_qubits: int) -> np.ndarray:
    """Aplica la lista de puertas, en orden, a un vector de 2^n componentes."""
    psi = np.asarray(state, dtype=complex).reshape((2,) * int(n_qubits))
    for gate in gates:
        psi = _apply_gate(psi, gate, int(n_qubits))
    return psi.reshape(-1)


def circuit_unitary(gates: list[Gate], qubits: tuple[int, ...]) -> np.ndarray:
    """Unitaria V del circuito restringida a `qubits` (ordenados; qubits[0] = MSB).

    Las puertas deben actuar solo sobre esos qubits.
    """
    posicion = {q: i for i, q in enumerate(qubits)}
    locales = []
    for gate in gates:
        try:
            if gate[0] == "RX":          # ("RX", qubit, angulo): solo se reindexa el qubit
                locales.append(("RX", posicion[int(gate[1])], gate[2]))
            else:
                locales.append((gate[0],) + tuple(posicion[int(q)] for q in gate[1:]))
        except KeyError as exc:
            raise ValueError(f"La puerta {gate} actua fuera de {qubits}.") from exc
    m = len(qubits)
    dim = 2 ** m
    unitaria = np.zeros((dim, dim), dtype=complex)
    for columna in range(dim):
        base = np.zeros(dim, dtype=complex)
        base[columna] = 1.0
        unitaria[:, columna] = apply_circuit(base, locales, m) if m else base
    return unitaria


# =====================================================================
# 2. Cambios de base
# =====================================================================

def paper_basis_change(kind: str, active: tuple[int, ...]) -> list[Gate]:
    """M_jk y M_ijkl del articulo (seccion 6), con los indices activos ordenados.

    Mismas secuencias que el codigo antiguo (add_paper_basis_change), que ya se
    verificaron para los tres patrones de double hopping.
    """
    if kind == "diagonal":
        return []
    if kind == "single_hopping":
        j, k = active
        return [("CNOT", k, j), ("H", k), ("CNOT", k, j)]
    if kind == "double_hopping":
        i, j, k, l = active
        return [("CNOT", i, j), ("CNOT", k, i), ("CNOT", l, k), ("H", l),
                ("CNOT", l, k), ("CNOT", k, i), ("CNOT", i, j)]
    raise ValueError(f"Tipo de grupo sin circuito del articulo: {kind}")


def compression_basis_change(active: tuple[int, ...], axis: str) -> list[Gate]:
    """Red de CNOTs que concentra la transicion en un qubit, mas una rotacion.

    Todas las cadenas de un bucket tienen X o Y en exactamente los mismos qubits
    activos, asi que todas cambian el estado x por x XOR M, con M = activos. Un
    abanico de CNOTs desde el pivote (activos[0]) hacia el resto convierte esa
    transicion en un cambio solo del pivote. Lo que queda sobre el pivote es una
    X o una Y (por la fase) multiplicada por signos diagonales del resto, y se
    diagonaliza rotando el pivote:
        axis = "X" -> H          (V X V^dag = Z)
        axis = "Y" -> S^dag, H   (V Y V^dag = Z)
    Es el mecanismo que describen el articulo y la tesis para M_ijkl ("una
    serie de CNOTs para que X actue solo en el ultimo qubit, luego Hadamard"),
    generalizado a cualquier numero de activos. No incluye la red inversa: no
    hace falta, porque los CNOTs permutan la base computacional y D sigue siendo
    diagonal.
    """
    if len(active) < 1:
        raise ValueError("La compresion necesita al menos un qubit activo.")
    pivote = active[0]
    puertas: list[Gate] = [("CNOT", pivote, t) for t in active[1:]]
    if axis == "X":
        puertas.append(("H", pivote))
    elif axis == "Y":
        puertas.extend([("SDG", pivote), ("H", pivote)])
    else:
        raise ValueError("axis debe ser 'X' o 'Y'.")
    return puertas


def specialized_8_index_basis_change(active: tuple[int, ...]) -> list[list[Gate]]:
    """Candidatos para la estructura de 8 indices (seccion 12).

    Siguiendo la tesis (X.39-X.54), la transicion se comprime con CNOTs y el
    resto se diagonaliza con una rotacion. Se devuelven las dos orientaciones
    posibles del pivote; la puerta de validacion decide cual vale.
    """
    if len(active) != 8:
        raise ValueError("La estructura de 8 indices necesita exactamente 8 qubits activos.")
    return [compression_basis_change(active, "X"), compression_basis_change(active, "Y")]


# =====================================================================
# 3. Grupos
# =====================================================================

def _active(term: PauliTerm) -> tuple[int, ...]:
    return tuple(sorted(int(q) for q, p in term if p in ("X", "Y")))


def _support(term: PauliTerm) -> tuple[int, ...]:
    return tuple(sorted(int(q) for q, p in term if p != "I"))


def _bits_matrix(indices: np.ndarray, width: int) -> np.ndarray:
    """Matriz (len(indices), width) con los bits; la columna 0 es el MSB."""
    shifts = width - 1 - np.arange(width)
    return (indices[:, None] >> shifts) & 1


def _register_bits(support: tuple[int, ...], n_qubits: int) -> np.ndarray:
    """Bits de `support` (en ese orden) para los 2^n estados del registro."""
    n = int(n_qubits)
    X = np.arange(2 ** n)
    if not support:
        return np.zeros((2 ** n, 0), dtype=np.int64)
    return np.stack([(X >> (n - 1 - int(q))) & 1 for q in support], axis=1)


@dataclass(slots=True)
class MeasurementGroup:
    """Un circuito de medida especializado y la forma de leer sus bitstrings.

    `blocks` guarda D factorizada: una lista de (espectadores, tabla), con los
    espectadores como posiciones dentro de `support` y la tabla indexada por los
    bits de los qubits activos. Para un bitstring x sobre `support`:
        D_xx = sum_bloques tabla[x_activos] * (-1)^(suma de x en espectadores)
    """

    kind: str
    active: tuple[int, ...]
    support: tuple[int, ...]
    gates: list[Gate]
    terms: dict[PauliTerm, complex]
    blocks: list[tuple[tuple[int, ...], np.ndarray]]
    offdiag_norm: float

    # -- lectura de un bitstring ------------------------------------------

    def local_index_from_bits(self, bits) -> int:
        """Bits medidos en el orden de `support` -> indice local (support[0] = MSB)."""
        valor = 0
        for b in bits:
            valor = (valor << 1) | int(b)
        return valor

    def diagonal_value(self, bits) -> float:
        """D_xx para un bitstring medido sobre `support` (en ese orden)."""
        bits = [int(b) for b in bits]
        pos_activos = [self.support.index(q) for q in self.active]
        idx_activos = 0
        for p in pos_activos:
            idx_activos = (idx_activos << 1) | bits[p]
        total = 0.0
        for espectadores, tabla in self.blocks:
            signo = -1.0 if sum(bits[p] for p in espectadores) % 2 else 1.0
            total += float(tabla[idx_activos]) * signo
        return total

    def diagonal_values(self) -> np.ndarray:
        """Tabla completa diag(D) sobre `support`, como la guardaba el legacy.

        Indexada por local_index_from_bits. Solo para soportes pequenos.
        """
        L = len(self.support)
        if L > MAX_TABLE_QUBITS:
            raise ValueError(f"Soporte de {L} qubits: tabla de 2^{L} entradas. "
                             "Usa diagonal_value(bits) en su lugar.")
        bits = _bits_matrix(np.arange(2 ** L), L)
        return self._diag_from_support_bits(bits)

    def _diag_from_support_bits(self, bits: np.ndarray) -> np.ndarray:
        pos_activos = [self.support.index(q) for q in self.active]
        m = len(pos_activos)
        idx_activos = np.zeros(bits.shape[0], dtype=np.int64)
        for j, p in enumerate(pos_activos):
            idx_activos |= bits[:, p] << (m - 1 - j)
        valores = np.zeros(bits.shape[0], dtype=float)
        for espectadores, tabla in self.blocks:
            if espectadores:
                signo = 1 - 2 * (bits[:, list(espectadores)].sum(axis=1) % 2)
            else:
                signo = 1
            valores += tabla[idx_activos] * signo
        return valores

    # -- valor esperado exacto -------------------------------------------

    def register_values(self, n_qubits: int) -> np.ndarray:
        """D_xx para cada uno de los 2^n estados del registro completo."""
        return self._diag_from_support_bits(_register_bits(self.support, n_qubits))

    def expectation_from_probabilities(self, probs: np.ndarray, n_qubits: int) -> float:
        """sum_x D_xx P(x), con P sobre todo el registro (2^n)."""
        return float(np.dot(probs, self.register_values(n_qubits)))


@dataclass(slots=True)
class QWCGroup:
    """Grupo de cadenas qubit-wise commuting (seccion 13).

    Tras las rotaciones X -> H y Y -> S^dag, H, cada cadena del grupo se convierte
    en +Z sobre su soporte (signo +1 en ambos casos; el test lo comprueba), asi
    que <P> = sum_x P(x) (-1)^(paridad de x en el soporte de P).
    """

    basis: dict[int, str]
    terms: dict[PauliTerm, complex] = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return "qwc"

    @property
    def support(self) -> tuple[int, ...]:
        return tuple(sorted(self.basis))

    @property
    def gates(self) -> list[Gate]:
        puertas: list[Gate] = []
        for q in sorted(self.basis):
            letra = self.basis[q]
            if letra == "X":
                puertas.append(("H", q))
            elif letra == "Y":
                puertas.extend([("SDG", q), ("H", q)])
        return puertas

    # Misma interfaz de lectura que MeasurementGroup: el valor de un bitstring
    # medido sobre `support` es sum_P c_P (-1)^(paridad de x en el soporte de P).
    # Asi el muestreador no necesita distinguir entre tipos de grupo.

    def diagonal_value(self, bits) -> float:
        """Valor del grupo para un bitstring medido sobre `support` (en ese orden)."""
        posicion = {q: i for i, q in enumerate(self.support)}
        bits = [int(b) for b in bits]
        total = 0.0
        for term, coef in self.terms.items():
            paridad = sum(bits[posicion[int(q)]] for q, _ in term) % 2
            total += complex(coef).real * (-1.0 if paridad else 1.0)
        return total

    def register_values(self, n_qubits: int) -> np.ndarray:
        """Valor del grupo para cada uno de los 2^n estados del registro."""
        n = int(n_qubits)
        X = np.arange(2 ** n)
        valores = np.zeros(2 ** n, dtype=float)
        for term, coef in self.terms.items():
            paridad = np.zeros(2 ** n, dtype=np.int64)
            for q, _ in term:
                paridad ^= (X >> (n - 1 - int(q))) & 1
            valores += complex(coef).real * (1 - 2 * paridad)
        return valores

    def expectation_from_probabilities(self, probs: np.ndarray, n_qubits: int) -> float:
        return float(np.dot(probs, self.register_values(n_qubits)))


# =====================================================================
# 4. La puerta de validacion
# =====================================================================

def diagonalize_bucket(terms: dict[PauliTerm, complex], active: tuple[int, ...],
                       support: tuple[int, ...], gates: list[Gate]):
    """Calcula D = V O V^dag por bloques y devuelve (bloques, ||offdiag D||_F).

    Ver la cabecera del modulo para la factorizacion y por que la norma es la
    del pseudocodigo.
    """
    m = len(active)
    pos_activo = {q: i for i, q in enumerate(active)}
    pos_soporte = {q: i for i, q in enumerate(support)}
    V = circuit_unitary(gates, active) if m else np.eye(1, dtype=complex)

    crudos: dict[tuple[int, ...], np.ndarray] = {}
    for term, coef in terms.items():
        letras = ["I"] * m
        espectadores = []
        for q, p in term:
            q = int(q)
            if q in pos_activo:
                letras[pos_activo[q]] = p
            elif p == "Z":
                espectadores.append(pos_soporte[q])
            else:
                raise ValueError(f"{term}: un qubit no activo con letra {p}.")
        Q = np.eye(1, dtype=complex)
        for letra in letras:
            Q = np.kron(Q, _PAULI[letra])
        clave = tuple(sorted(espectadores))
        crudos[clave] = crudos.get(clave, 0) + complex(coef) * Q

    n_espectadores = len(support) - m
    off2 = 0.0
    bloques = []
    peor_imag = 0.0
    for clave, B in crudos.items():
        B = V @ B @ V.conj().T
        diagonal = np.diag(B)
        off2 += float(np.sum(np.abs(B - np.diag(diagonal)) ** 2))
        peor_imag = max(peor_imag, float(np.max(np.abs(diagonal.imag))))
        bloques.append((clave, diagonal.real.copy()))
    norma = float(np.sqrt(off2 * 2.0 ** n_espectadores))
    if peor_imag > 1e-9:
        # Un operador hermitico rotado por una unitaria tiene diagonal real.
        norma = max(norma, peor_imag)
    return bloques, norma


# =====================================================================
# 5. QWC (seccion 13)
# =====================================================================

def qwc_compatible(p: PauliTerm, q: PauliTerm) -> bool:
    letras_q = dict(q)
    return all(letras_q.get(qb, letra) == letra for qb, letra in p)


def group_qwc(terms: dict[PauliTerm, complex]) -> list[QWCGroup]:
    """Agrupacion voraz (seccion 13.1). Orden determinista."""
    grupos: list[QWCGroup] = []
    for term in sorted(terms):
        for grupo in grupos:
            if all(grupo.basis.get(int(q), p) == p for q, p in term):
                grupo.terms[term] = terms[term]
                for q, p in term:
                    grupo.basis[int(q)] = p
                break
        else:
            grupos.append(QWCGroup(basis={int(q): p for q, p in term},
                                   terms={term: terms[term]}))
    return grupos


# =====================================================================
# 6. El plan
# =====================================================================

@dataclass(slots=True)
class MeasurementPlan:
    identity: complex
    groups: list[MeasurementGroup]
    qwc_groups: list[QWCGroup]
    rejected: dict[tuple[int, ...], float]     # buckets que no pasaron la puerta

    def summary(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for g in self.groups:
            out[g.kind] = out.get(g.kind, 0) + 1
        out["qwc"] = len(self.qwc_groups)
        out["circuits"] = len(self.groups) + len(self.qwc_groups)
        out["terms_specialized"] = sum(len(g.terms) for g in self.groups)
        out["terms_qwc"] = sum(len(g.terms) for g in self.qwc_groups)
        return out

    def all_groups(self):
        return [*self.groups, *self.qwc_groups]

    def exact_expectation(self, state: np.ndarray, n_qubits: int) -> float:
        """Limite de shots infinitos: cada circuito aplicado al estado, P(x)
        exactas y suma de D_xx P(x). Es el numero que la fase D debe reproducir.
        """
        media, _ = self.exact_statistics(state, n_qubits)
        return media

    def exact_statistics(self, state: np.ndarray, n_qubits: int) -> tuple[float, list[float]]:
        """(media exacta, varianza POR SHOT de cada circuito).

        Para un circuito, cada shot devuelve un bitstring x con probabilidad P(x)
        y aporta el valor D_xx, asi que su varianza es
            Var = sum_x P(x) D_xx^2 - (sum_x P(x) D_xx)^2.
        Con nshots por circuito y circuitos independientes, el error tipico del
        estimador es sqrt(sum_g Var_g / nshots). Es la referencia contra la que
        se juzga la ruta muestreada (fase D): no "se parece al exacto", sino
        "se aleja de el lo que el ruido de shots permite".
        """
        media = complex(self.identity).real
        varianzas: list[float] = []
        for grupo in self.all_groups():
            psi = apply_circuit(state, grupo.gates, n_qubits)
            probs = np.abs(psi) ** 2
            valores = grupo.register_values(n_qubits)
            m = float(np.dot(probs, valores))
            media += m
            varianzas.append(max(float(np.dot(probs, valores ** 2)) - m * m, 0.0))
        return float(media), varianzas

    def exact_standard_error(self, state: np.ndarray, n_qubits: int, nshots: int) -> float:
        """Error tipico del estimador muestreado con `nshots` por circuito."""
        _, varianzas = self.exact_statistics(state, n_qubits)
        return float(np.sqrt(sum(varianzas) / int(nshots)))


def _attempts(n_activos: int, gradient: bool, compression_fallback: bool,
              active: tuple[int, ...]) -> list[tuple[str, list[Gate]]]:
    intentos: list[tuple[str, list[Gate]]] = []
    if n_activos == 0:
        intentos.append(("diagonal", []))
    elif n_activos == 2:
        intentos.append(("single_hopping", paper_basis_change("single_hopping", active)))
    elif n_activos == 4:
        intentos.append(("double_hopping", paper_basis_change("double_hopping", active)))
    elif n_activos == 8 and gradient:
        intentos.extend(("gradient_8", g) for g in specialized_8_index_basis_change(active))
    if compression_fallback and n_activos >= 1 and not (n_activos == 8 and gradient):
        intentos.extend(("compressed", compression_basis_change(active, eje)) for eje in ("X", "Y"))
    return intentos


def _build_plan(operator, *, gradient: bool, compression_fallback: bool,
                tol: float) -> MeasurementPlan:
    identidad = complex(operator.terms.get((), 0.0))
    buckets: dict[tuple[int, ...], dict[PauliTerm, complex]] = {}
    for term, coef in operator.terms.items():
        if term == () or abs(complex(coef)) == 0.0:
            continue
        buckets.setdefault(_active(term), {})[term] = complex(coef)

    grupos: list[MeasurementGroup] = []
    residual: dict[PauliTerm, complex] = {}
    rechazados: dict[tuple[int, ...], float] = {}
    for active, terms in sorted(buckets.items()):
        support = tuple(sorted(set().union(*(set(_support(t)) for t in terms))))
        aceptado = None
        mejor = np.inf
        for kind, gates in _attempts(len(active), gradient, compression_fallback, active):
            bloques, norma = diagonalize_bucket(terms, active, support, gates)
            mejor = min(mejor, norma)
            if norma <= tol:
                aceptado = MeasurementGroup(kind=kind, active=active, support=support,
                                            gates=gates, terms=dict(terms),
                                            blocks=bloques, offdiag_norm=norma)
                break
        if aceptado is None:
            residual.update(terms)
            if np.isfinite(mejor):
                rechazados[active] = float(mejor)
        else:
            grupos.append(aceptado)

    return MeasurementPlan(identity=identidad, groups=grupos,
                           qwc_groups=group_qwc(residual), rejected=rechazados)


def build_energy_measurement_plan(qubit_hamiltonian, *, tol: float = DEFAULT_TOL,
                                  compression_fallback: bool = False) -> MeasurementPlan:
    """Plan de la energia (seccion 5.1): diagonal, M_jk, M_ijkl y QWC residual."""
    return _build_plan(qubit_hamiltonian, gradient=False,
                       compression_fallback=compression_fallback, tol=tol)


def build_gradient_measurement_plan(gradient_operator, *, tol: float = DEFAULT_TOL,
                                    compression_fallback: bool = False) -> MeasurementPlan:
    """Plan del gradiente (seccion 11) para G_mu de build_gradient_operator.

    Igual que el de la energia mas el intento de 8 indices. Con
    `compression_fallback=True` (NO esta en el pseudocodigo) los buckets que
    fallan con los circuitos del articulo, y los de 5-7 activos, prueban la
    compresion por CNOTs antes de ir a QWC. Siempre detras de la puerta de
    validacion, asi que no puede dar valores incorrectos; solo cambia cuantos
    circuitos hacen falta.
    """
    return _build_plan(gradient_operator, gradient=True,
                       compression_fallback=compression_fallback, tol=tol)


def build_qwc_plan(operator) -> MeasurementPlan:
    """Plan solo QWC (estrategia 'qwc'): sin circuitos especializados.

    Sirve de linea base para comparar con la estrategia 'paper'.
    """
    terms = {t: complex(c) for t, c in operator.terms.items()
             if t != () and abs(complex(c)) > 0.0}
    return MeasurementPlan(identity=complex(operator.terms.get((), 0.0)), groups=[],
                           qwc_groups=group_qwc(terms), rejected={})
