"""Ruta muestreada: ejecutar los planes de medida con shots en Qibo.

Modulo del paquete: src/vqd_nuclear/sampling.py
Pseudocodigo: secciones 8 (medir un grupo), 14 (gradiente muestreado) y 16.2
(SWAP destructivo).

QUE HACE
--------
MeasurementPlan dice qué medir: para cada circuito, las puertas del cambio de base,
los qubits a medir (`support`) y el valor de cada bitstring (`diagonal_value`).
Este modulo lo EJECUTA:

    circuito del ansatz  ->  + puertas del grupo  ->  M(support)  ->  nshots
    ->  frecuencias de bitstrings  ->  media de diagonal_value(bits)

Es la misma funcion para la energia y para cualquier G_mu, y para grupos
especializados y QWC: todos exponen la misma interfaz (support, gates,
diagonal_value). Las cuatro funciones de muestreo del codigo antiguo se reducen
a `sample_group` + `sampled_expectation`.

CONVENIO DE SHOTS
-----------------
`nshots` es por CIRCUITO, como en el codigo antiguo. El coste total de una
estimacion es nshots * plan.summary()["circuits"].

CONVENIO DE BITS
----------------
Qibo devuelve las claves de `frequencies(binary=True)` en el orden de los qubits
pasados a gates.M. Se mide en el orden de `group.support`, que es el que espera
`diagonal_value`. El test lo comprueba explicitamente.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from qibo import gates
from qibo.models import Circuit

from .measurement_plan import apply_circuit


# =====================================================================
# Traduccion de las puertas del plan a Qibo
# =====================================================================

def add_plan_gates(circuit: Circuit, plan_gates) -> Circuit:
    """Anade al circuito las puertas del plan, en orden. Devuelve el circuito.

    Mismas convenciones que measurement_plan.apply_circuit:
        ("CNOT", c, t), ("H", q), ("SDG", q), ("RX", q, theta)
    RX de Qibo es exp(-i theta X / 2), igual que en el simulador del plan.
    """
    for gate in plan_gates:
        name = gate[0]
        if name == "CNOT":
            circuit.add(gates.CNOT(int(gate[1]), int(gate[2])))
        elif name == "H":
            circuit.add(gates.H(int(gate[1])))
        elif name == "SDG":
            circuit.add(gates.SDG(int(gate[1])))
        elif name == "RX":
            circuit.add(gates.RX(int(gate[1]), theta=float(gate[2])))
        else:
            raise ValueError(f"Puerta del plan sin traduccion a Qibo: {gate}")
    return circuit


def bitstring_to_array(bitstring, length: int) -> np.ndarray:
    """Clave de frecuencias de Qibo -> array de bits (sin cambios respecto al legacy)."""
    if isinstance(bitstring, str):
        bits = bitstring.replace(" ", "")
    elif isinstance(bitstring, (tuple, list, np.ndarray)):
        arr = np.array(bitstring, dtype=int).ravel()
        if len(arr) != length:
            raise ValueError(f"Se esperaban {length} bits, llegaron {len(arr)}.")
        return arr
    else:
        bits = format(int(bitstring), f"0{length}b")
    if len(bits) != length:
        raise ValueError(f"Se esperaban {length} bits, llegaron {len(bits)}.")
    return np.array([int(b) for b in bits], dtype=int)


# =====================================================================
# Un grupo (seccion 8)
# =====================================================================

def measurement_circuit(base_circuit: Circuit, group) -> Circuit:
    """Copia del circuito base + cambio de base del grupo + medida de `support`."""
    circuit = base_circuit.copy(deep=True)
    add_plan_gates(circuit, group.gates)
    circuit.add(gates.M(*[int(q) for q in group.support]))
    return circuit


def stats_from_frquencies(group, frequencies) -> tuple[float, float, int]:
    """
    calcula (media, varianza muestral por shot, numero de shots) de diagonal_value.
    La varianza sale de los propios bitstrings medidos.
    """

    L = len(group.support)
    values = []
    counts = []

    for key, count in frequencies.items():
        values.append(group.diagonal_value(bitstring_to_array(key, L)))
        counts.append(int(count))
    n = int(sum(counts))
    if n == 0:
        raise RuntimeError("No shots returned from Qibo")
    v = np.asarray(values, dtype=float)
    c = np.asarray(counts, dtype=float)

    average = float(np.dot(c, v) / n)
    var = float(np.dot(c, (v - average) ** 2) / (n - 1)) if n > 1 else 0.0

    return average, var, n


def value_from_frequencies(group, frequencies) -> tuple[float, int]:
    """(media de diagonal_value sobre los shots, numero de shots)."""
    average, _, n = stats_from_frquencies(group, frequencies)
    return average, n


def sample_group_stats(base_circuit: Circuit, group, nshots: int, *,
                       initial_state: np.ndarray | None = None) -> tuple[float, float]:
    """
    (contribucion estimada del grupo, varianza de esa estimacion).

    La varianza devuelta es la de la meddia: var_por_shot / nshots.
    `initial_state` solo se debería usar en los tests.
    """
    if not group.support:          # nothing to measure (should not happen)
        return 0.0, 0.0
    circuit = measurement_circuit(base_circuit, group)
    kwargs: dict[str, Any] = {"nshots": int(nshots)}
    if initial_state is not None:
        kwargs["initial_state"] = np.array(initial_state, dtype=complex, copy=True)
    result = circuit(**kwargs)
    average, var, n = stats_from_frquencies(group, result.frequencies(binary=True))
    return float(average), float(var / n)


def sample_group(base_circuit: Circuit, group, nshots: int, *,
                 initial_state: np.ndarray | None = None) -> float:
    """Estimacion de la contribucion de un grupo con `nshots` shots."""
    average, _ = sample_group_stats(base_circuit, group, nshots, initial_state=initial_state)
    return average


# =====================================================================
# Un plan completo (secciones 5, 11, 14)
# =====================================================================

def sampled_expectation_with_error(plan, base_circuit: Circuit, nshots: int, *,
                                   initial_state: np.ndarray | None = None) -> tuple[float, float]:
    """
    (<O> estimado, error tipico estimado a partir de los shots).

    Los circuitos son independientes, asi que las varianzas se suman. El error
    tipico converge a plan.exact_standard_error(psi, n, nshots).
    """
    value = complex(plan.identity).real
    var_total = 0.0

    for group in plan.all_groups():
        m, v = sample_group_stats(base_circuit, group, nshots, initial_state=initial_state)
        value += m
        var_total += v

    return float(value), float(np.sqrt(var_total))


def sampled_expectation(plan, base_circuit: Circuit, nshots: int, *,
                        initial_state: np.ndarray | None = None) -> float:
    """<O> estimado con shots: identidad + suma de cada circuito del plan.

    Converge a plan.exact_expectation(psi) cuando nshots -> inf, con error
    tipico plan.exact_standard_error(psi, n, nshots).
    """
    value, _ = sampled_expectation_with_error(plan, base_circuit, nshots,
                                              initial_state=initial_state)
    return value


# =====================================================================
# SWAP destructivo (seccion 16.2) igual al legacy
# =====================================================================

def copy_gate_with_offset(gate: Any, offset: int) -> Any:
    """Copia una puerta de Qibo desplazando sus qubits."""
    shifted = [int(q) + int(offset) for q in gate.qubits]
    kwargs = getattr(gate, "init_kwargs", {}) or {}
    try:
        return gate.__class__(*shifted, **kwargs)
    except TypeError:
        params = getattr(gate, "parameters", ())
        return gate.__class__(*shifted, *params, **kwargs)


def destructive_swap_circuit(circuit_i: Circuit, circuit_k: Circuit, n_qubits: int) -> Circuit:
    """Dos registros con cada ansatz, CNOT + H por pares y medida de todo."""
    n = int(n_qubits)
    circuit = Circuit(2 * n)
    for gate in circuit_i.queue:
        circuit.add(copy_gate_with_offset(gate, 0))
    for gate in circuit_k.queue:
        circuit.add(copy_gate_with_offset(gate, n))
    for q in range(n):
        circuit.add(gates.CNOT(q, q + n))
        circuit.add(gates.H(q))
    circuit.add(gates.M(*range(2 * n)))
    return circuit


def destructive_swap_overlap(circuit_i: Circuit, circuit_k: Circuit, n_qubits: int,
                             nshots: int) -> float:
    """Estimacion de |<psi_i|psi_k>|^2. Cada shot vale (-1)^(a.b) con a, b los
    dos registros; la media es el solapamiento. Varianza por shot: 1 - F^2.
    """
    n = int(n_qubits)
    result = destructive_swap_circuit(circuit_i, circuit_k, n)(nshots=int(nshots))
    return swap_overlap_from_frequencies(result.frequencies(binary=True), n)


def swap_overlap_from_frequencies(frequencies, n_qubits: int) -> float:
    """Media de (-1)^(a.b) sobre los shots: el estimador del SWAP destructivo.

    Las claves tienen 2N bits: los N primeros son el registro a (qubits 0..N-1)
    y los N siguientes el registro b. Lo usan los dos muestreadores.
    """
    n = int(n_qubits)
    total = 0.0
    shots = 0
    for key, counts in frequencies.items():
        bits = bitstring_to_array(key, 2 * n)
        signo = -1.0 if int(np.sum(bits[:n] * bits[n:]) % 2) else 1.0
        total += signo * int(counts)
        shots += int(counts)
    if shots == 0:
        raise RuntimeError("No shots in the destructive SWAP.")
    return float(total / shots)

# =====================================================================
# Compute-Uncompute : alternative to SWAP for overlap comparison 
# =====================================================================

def compute_uncompute_circuit(circuit: Circuit, inverse_previous: Circuit, n_qubits: int,
                              *, measure: bool = True) -> Circuit:
    """
    """
    n = int(n_qubits)

    if has_measurements(circuit) or has_measurements(inverse_previous):
        raise ValueError("U or U_j cannot contain measurement gates")
    for part in (circuit, inverse_previous):
        if part.nqubits != n:
            raise ValueError(f"Circuit with {part.nqubits} qubits, expected N = {n}.")

    out = Circuit(n)
    for gate in circuit.queue:                      # U|0^N>
        out.add(copy_gate_with_offset(gate, 0))
    for gate in inverse_previous.queue:             # U_j^dagger
        out.add(copy_gate_with_offset(gate, 0))
    if measure:
        out.add(gates.M(*range(n)))
    return out


def compute_uncompute_stats_from_counts(n0: int, total: int) -> tuple[float, float, int]:
    """(F, Var(F), n_shots) from n0 = #(x = 0^N) and the total number of shots.

        F      = n0 / n
        Var(F) = F (1 - F) / (n - 1)      (variance of the mean, ddof = 1)

    Used by both samplers.
    """
    total = int(total)
    n0 = int(n0)
    if total == 0:
        raise RuntimeError("No shots in the compute-uncompute circuit.")
    if not 0 <= n0 <= total:
        raise ValueError(f"n0 = {n0} out of range for {total} shots.")
    f = n0 / total
    var = f * (1.0 - f) / (total - 1) if total > 1 else 0.0
    return float(f), float(var), total


def compute_uncompute_stats_from_frequencies(frequencies, n_qubits: int) -> tuple[float, float, int]:
    """(F, Var(F), n_shots) from Qibo frequencies of M(0, ..., N-1)."""
    n = int(n_qubits)
    n0 = 0
    total = 0
    for key, counts in frequencies.items():
        if not bitstring_to_array(key, n).any():        # if x = 0^N
            n0 += int(counts)
        total += int(counts)
    return compute_uncompute_stats_from_counts(n0, total)


def compute_uncompute_overlap(circuit: Circuit, inverse_previous: Circuit, n_qubits: int,
                              nshots: int) -> float:
    """Estimate of |<psi_j|psi>|^2 with compute-uncompute. Only returns the overlap value)."""
    result = compute_uncompute_circuit(circuit, inverse_previous, n_qubits)(nshots=int(nshots))
    f, _, _ = compute_uncompute_stats_from_frequencies(result.frequencies(binary=True), n_qubits)
    return f

# =====================================================================
# Gradiente de la penalizacion
# =====================================================================

_CONTROLLED = {"X": gates.CNOT, "Y": gates.CY, "Z": gates.CZ}


def has_measurements(circuit: Circuit) -> bool:
    """True if the circuit contains a measurement (M) gate"""
    return any(isinstance(gate, gates.M) for gate in circuit.queue)


def inverse_circuit(circuit: Circuit) -> Circuit:
    """
    Obtains U^dagg of a Circuit U. 

    We need to check that there is no M gates in U beffore inversion
    """
    if has_measurements(circuit):
        raise ValueError("Can not use as U_j a circuit with measurement gates.")
    return circuit.invert()


def penalty_gradient_circuit(base_circuit: Circuit, inverse_previous: Circuit,
                             pauli_term, n_qubits: int, *, measure: bool = True) -> Circuit:
    """
    Construction of the elemental N+1 qunit circuit for m = Re<psi|Pi_j(-iP)|psi>

    base_circuit:      U, current ansatz (N qubits, no measurements).
    inverse_previous:  U_j^dag from previous state (N qubits, no measurements).
    pauli_term:        P chain as tuple (qubit, letter), No I.
    measure:           Default True, False only for tests (to know exact value from state).

    Measure: M(ancilla, 0, 1, ..., N-1), so, in frequencie keys, first bit is ancilla and
    next N are system in order.    
    """
    n = n_qubits
    ancilla = n
    if has_measurements(base_circuit) or has_measurements(inverse_previous):
        raise ValueError("U or U_j cannot contain measurement gates")
    if not pauli_term:
        raise ValueError("I chain does not add to the gradient -> not measured")

    circuit = Circuit(n+1)
    for gate in base_circuit.queue:
        circuit.add(copy_gate_with_offset((gate), 0))   # U|0^N>
    
    circuit.add(gates.H(ancilla))                       # H_a

    for qubit, letter in pauli_term:                    # Controlled P (control in ancilla)
        letter = str(letter)
        if letter not in _CONTROLLED:
            raise ValueError(f"Pauli operator not valid: {letter}")
        if not 0 <= int(qubit) < n:
            raise ValueError(f"Qubit {qubit} out of system range (N = {n}).")
        circuit.add(_CONTROLLED[letter](ancilla, int(qubit)))

    circuit.add(gates.SDG(ancilla))                     # S^dag_a y H_a: A = -iP
    circuit.add(gates.H(ancilla))

    for gate in inverse_previous.queue:                 # U_j^dag
        circuit.add(copy_gate_with_offset(gate, 0))

    if measure:                                         # M gates as: ancilla + system
        circuit.add(gates.M(ancilla, *range(n)))

    return circuit


def penalty_stats_from_frequencies(frequencies, n_qubits: int) -> tuple[float, float, int]:
    """
    (m, var(m), n_shots), from frequencies for penalty circuit.

    w = +1 if (a=0, x=0^N), -1 if (a=1, x=0^N), 0 if else.
        m      = (n0 - n1) / n                     (n=n_shots, n0=n_(w=+1), n1=n_(w=-1))
        Var(m) = ((n0 + n1)/n - m^2) / (n - 1)     (variance of the mean)
    """
    n_sys = int(n_qubits)
    n0 = n1 = total = 0
    for key, counts in frequencies.items():
        bits = bitstring_to_array(key, n_sys + 1)
        counts = int(counts)
        total += counts
        if not bits[1:].any():  # if X=0^N:
            if bits[0] == 0:
                n0 += counts
            else:
                n1 += counts
    return penalty_stats_from_counts(n0, n1, total)


def penalty_stats_from_counts(n0: int, n1: int, total: int) -> tuple[float, float, int]:
    """(m, var(m), n_shots) from n0 = #(a=0, x=0^N), n1 = #(a=1, x=0^N) and the
    total number of shots. The estimator only depends on these three numbers.

        m      = (n0 - n1) / n
        Var(m) = ((n0 + n1)/n - m^2) / (n - 1)     (variance of the mean)
    """
    total = int(total)
    if total == 0:
        raise RuntimeError("No shots in the penalty circuit.")
    m = (int(n0) - int(n1)) / total
    var = ((int(n0) + int(n1)) / total - m * m) / (total - 1) if total > 1 else 0.0
    return float(m), float(max(var, 0.0)), int(total)
    

def sample_penalty_term(base_circuit: Circuit, inverse_previous: Circuit, pauli_term,
                        n_qubits: int, nshots: int) -> tuple[float, float]:
    """(estimated m, var(m)), running the circuit with `nshots`."""
    circuit = penalty_gradient_circuit(base_circuit, inverse_previous, pauli_term, n_qubits)
    result = circuit(nshots=int(nshots))
    m, var, _ = penalty_stats_from_frequencies(result.frequencies(binary=True), n_qubits)
    return m, var


def penalty_term_exact(base_circuit: Circuit, inverse_previous: Circuit, pauli_term,
                       n_qubits: int) -> tuple[float, float]:
    """(exact m, exact var per shot), from the state from the circuit (no measures).

    Infinite shots limit, for testing. Index of (a, x=0^N) is 0: (a=0), 1: (a=1).
    """
    circuit = penalty_gradient_circuit(base_circuit, inverse_previous, pauli_term,
                                       n_qubits, measure=False)
    state = np.asarray(circuit().state())
    p0, p1 = float(abs(state[0]) ** 2), float(abs(state[1]) ** 2)
    m = p0 - p1
    return m, max(p0 + p1 - m * m, 0.0)


# =====================================================================
# Muestreador por la regla de Born ("born"), alternativa a ejecutar en Qibo
# =====================================================================
#
# QUE HACE. Para un circuito "prefijo U + sufijo V + medida final" Qibo simula
# U y V sobre 2^N amplitudes, calcula P(x) y extrae nshots muestras de P. Aqui
# se hace lo mismo, pero:
#   - el estado del prefijo, |psi> = U|0>, lo da AnsatzBuilder.state (el MISMO
#     circuito simulado en Qibo) y se reutiliza en todos los circuitos que lo
#     comparten (todos los del plan de la energia, los de cada G_mu, ...);
#   - el sufijo V (3-7 puertas del cambio de base) se aplica con
#     measurement_plan.apply_circuit;
#   - las cuentas de los nshots se extraen de golpe de una Multinomial(n, P).
#
# POR QUE ES LO MISMO. Las cuentas de n extracciones independientes de una
# distribucion P siguen EXACTAMENTE una Multinomial(n, P) (es su definicion), y
# los estimadores solo dependen de las cuentas. El algoritmo recibe el mismo
# diccionario bitstring -> cuenta, con la misma ley de probabilidad, y los
# circuitos y los shots que se contabilizan (usage) no cambian.
#
# REGLA: se reutilizan ESTADOS, nunca cuentas. Cada llamada extrae una muestra
# nueva e independiente, como cada ejecucion en el hardware.
#
# Numeros aleatorios: los del generador global de NumPy (el que fija
# np.random.seed en set_reproducible_seed), salvo que se pase `rng`.

MAX_BORN_QUBITS = 28        # 2^28 amplitudes = 4 GB en complex128


def _generator(rng):
    return np.random if rng is None else rng


def born_probabilities(state: np.ndarray, n_qubits: int, measured) -> np.ndarray:
    """P(x) de los qubits `measured`, en ESE orden, sumando sobre el resto.

    Indice de salida: measured[0] es el bit mas significativo, igual que las
    claves de Qibo para gates.M(*measured). El qubit q del estado pesa
    1 << (N - 1 - q) (convenio de Qibo y de apply_circuit).
    """
    n = int(n_qubits)
    measured = [int(q) for q in measured]
    if not measured:
        raise ValueError("No measured qubits.")
    if len(set(measured)) != len(measured) or not all(0 <= q < n for q in measured):
        raise ValueError(f"Measured qubits {measured} not valid for N = {n}.")
    if n > MAX_BORN_QUBITS:
        raise ValueError(f"{n} qubits: 2^{n} amplitudes do not fit in memory.")
    probs = np.abs(np.asarray(state).ravel()) ** 2
    if probs.size != 2 ** n:
        raise ValueError(f"State with {probs.size} amplitudes, expected 2^{n}.")
    norm = float(probs.sum())
    if not np.isfinite(norm) or abs(norm - 1.0) > 1e-8:
        raise ValueError(f"State not normalized (norm^2 = {norm}).")
    tensor = (probs / norm).reshape((2,) * n)
    unmeasured = tuple(q for q in range(n) if q not in measured)
    marginal = tensor.sum(axis=unmeasured) if unmeasured else tensor
    # Tras sumar, los ejes que quedan estan en orden creciente de qubit.
    in_order = sorted(measured)
    marginal = np.transpose(marginal, [in_order.index(q) for q in measured]).reshape(-1)
    return marginal / marginal.sum()


def born_frequencies(state: np.ndarray, n_qubits: int, measured, nshots: int, *,
                     rng=None) -> dict[str, int]:
    """Cuentas de `nshots` medidas de `measured` sobre `state`: el mismo
    diccionario {bitstring: cuenta} que result.frequencies(binary=True) de Qibo
    para gates.M(*measured), con la misma ley (Multinomial(nshots, P))."""
    probs = born_probabilities(state, n_qubits, measured)
    counts = _generator(rng).multinomial(int(nshots), probs)
    width = len(list(measured))
    return {format(int(i), f"0{width}b"): int(counts[i]) for i in np.flatnonzero(counts)}


def sample_group_stats_from_state(state: np.ndarray, group, nshots: int, n_qubits: int, *,
                                  rng=None) -> tuple[float, float]:
    """Como sample_group_stats, pero partiendo del estado del ansatz.

    Aplica las puertas del grupo (V) a |psi> y mide `support`. Devuelve
    (contribucion estimada, varianza de la media), con el mismo estimador.
    """
    if not group.support:
        return 0.0, 0.0
    phi = apply_circuit(state, group.gates, n_qubits) if group.gates else np.asarray(state).ravel()
    frequencies = born_frequencies(phi, n_qubits, group.support, nshots, rng=rng)
    average, var, n = stats_from_frquencies(group, frequencies)
    return float(average), float(var / n)


def sampled_expectation_with_error_from_state(plan, state: np.ndarray, nshots: int,
                                              n_qubits: int, *, rng=None) -> tuple[float, float]:
    """Como sampled_expectation_with_error, pero con el estado ya calculado:
    identidad + suma de los grupos, y suma de varianzas (circuitos independientes)."""
    value = complex(plan.identity).real
    var_total = 0.0
    for group in plan.all_groups():
        m, v = sample_group_stats_from_state(state, group, nshots, n_qubits, rng=rng)
        value += m
        var_total += v
    return float(value), float(np.sqrt(var_total))


def _parity(values: np.ndarray) -> np.ndarray:
    """Paridad del numero de bits a 1 de cada entero (int64, vectorizado)."""
    v = np.array(values, dtype=np.int64, copy=True)
    for shift in (32, 16, 8, 4, 2, 1):
        v ^= v >> shift
    return v & 1


def apply_pauli_string(state: np.ndarray, pauli_term, n_qubits: int) -> np.ndarray:
    """P|state> para una cadena de Pauli, vectorizado.

    P|b> = i^(n_Y) (-1)^(paridad de b en los qubits con Y o Z) |b XOR m>, con m
    los qubits con X o Y. Mismo convenio que apply_pauli_string_to_state
    (measurements.py): Y|0> = i|1>, Y|1> = -i|0>.
    """
    n = int(n_qubits)
    state = np.asarray(state, dtype=complex).ravel()
    m_flip = m_sign = n_y = 0
    for q, letter in pauli_term:
        bit = 1 << (n - 1 - int(q))
        letter = str(letter)
        if letter == "X":
            m_flip |= bit
        elif letter == "Y":
            m_flip |= bit
            m_sign |= bit
            n_y += 1
        elif letter == "Z":
            m_sign |= bit
        elif letter != "I":
            raise ValueError(f"Pauli operator not valid: {letter}")
    index = np.arange(state.size, dtype=np.int64)
    phase = (1j ** n_y) * (1 - 2 * _parity(index & m_sign))
    out = np.empty_like(state)
    out[index ^ m_flip] = phase * state
    return out


def penalty_probabilities_from_states(psi: np.ndarray, psi_previous: np.ndarray,
                                      pauli_term, n_qubits: int) -> tuple[float, float]:
    """(p0, p1) = P(a=0, x=0^N), P(a=1, x=0^N) del circuito de penalizacion.

    Antes de U_j^dag el estado es 1/2 [|0>_a (I - iP)|psi> + |1>_a (I + iP)|psi>],
    y <0^N|U_j^dag = <psi_j|. Con c = <psi_j|psi> y d = <psi_j|P psi>:
        p0 = 1/4 |c - i d|^2,     p1 = 1/4 |c + i d|^2.
    Una fase global en psi o en psi_j no cambia nada (|.|^2).
    """
    if not pauli_term:
        raise ValueError("I chain does not add to the gradient -> not measured")
    psi = np.asarray(psi, dtype=complex).ravel()
    psi_previous = np.asarray(psi_previous, dtype=complex).ravel()
    c = np.vdot(psi_previous, psi)
    d = np.vdot(psi_previous, apply_pauli_string(psi, pauli_term, n_qubits))
    return float(0.25 * abs(c - 1j * d) ** 2), float(0.25 * abs(c + 1j * d) ** 2)


def sample_penalty_term_from_states(psi: np.ndarray, psi_previous: np.ndarray, pauli_term,
                                    n_qubits: int, nshots: int, *, rng=None) -> tuple[float, float]:
    """(m estimado, var(m)) del circuito de penalizacion, con `nshots` shots.

    El estimador solo usa (n0, n1, n): se extraen de una
    Multinomial(n; p0, p1, 1 - p0 - p1), que es su ley exacta. No se generan los
    bitstrings de los shots con x != 0^N (se cuentan juntos en el tercer grupo).
    """
    p0, p1 = penalty_probabilities_from_states(psi, psi_previous, pauli_term, n_qubits)
    rest = max(1.0 - p0 - p1, 0.0)
    probs = np.array([p0, p1, rest]) / (p0 + p1 + rest)
    n0, n1, _ = _generator(rng).multinomial(int(nshots), probs)
    m, var, _ = penalty_stats_from_counts(n0, n1, int(nshots))
    return m, var


def destructive_swap_overlap_from_states(psi_i: np.ndarray, psi_k: np.ndarray, n_qubits: int,
                                         nshots: int, *, rng=None) -> float:
    """SWAP destructivo a partir de los dos estados: el mismo circuito de 2N qubits.

    Registro a (qubits 0..N-1) = psi_i y registro b (N..2N-1) = psi_k, es decir
    psi_i (x) psi_k. Se aplica la capa final (CNOT(q, q+N), H(q)), se miden los
    2N qubits y se usa el mismo estimador que en Qibo. Ahorra simular los dos
    ansatze sobre 2N qubits; sigue haciendo falta el vector de 2^(2N).
    """
    n = int(n_qubits)
    if 2 * n > MAX_BORN_QUBITS:
        raise ValueError(f"Destructive SWAP with 2N = {2 * n} qubits does not fit in memory.")
    phi = np.kron(np.asarray(psi_i, dtype=complex).ravel(), np.asarray(psi_k, dtype=complex).ravel())
    layer = [gate for q in range(n) for gate in (("CNOT", q, q + n), ("H", q))]
    phi = apply_circuit(phi, layer, 2 * n)
    frequencies = born_frequencies(phi, 2 * n, range(2 * n), nshots, rng=rng)
    return swap_overlap_from_frequencies(frequencies, n)


def compute_uncompute_overlap_from_states(psi: np.ndarray, psi_previous: np.ndarray, n_qubits: int,
                                          nshots: int, *, rng=None) -> float:
    """Compute-uncompute from the two states.

    Before measuring, the state is U_j^dag|psi>, and its amplitude on 0^N is
    <0^N|U_j^dag|psi> = <psi_j|psi>, so P(x = 0^N) = F = |<psi_j|psi>|^2.
    The estimator only uses n0 = #(x = 0^N): (n0, n - n0) ~ Multinomial(n; F, 1 - F),
    i.e. n0 ~ Binomial(n, F), its exact law. The bitstrings of the shots with
    x != 0^N are not generated (they are counted together).
    """
    n = int(n_qubits)
    psi = np.asarray(psi, dtype=complex).ravel()
    psi_previous = np.asarray(psi_previous, dtype=complex).ravel()
    for name, vector in (("psi", psi), ("psi_previous", psi_previous)):
        if vector.size != 2 ** n:
            raise ValueError(f"{name} with {vector.size} amplitudes, expected 2^{n}.")
        norm = float(np.vdot(vector, vector).real)
        if abs(norm - 1.0) > 1e-8:
            raise ValueError(f"{name} not normalized (norm^2 = {norm}).")
    f_exact = min(max(abs(np.vdot(psi_previous, psi)) ** 2, 0.0), 1.0)
    n0 = _generator(rng).binomial(int(nshots), f_exact)
    f, _, _ = compute_uncompute_stats_from_counts(n0, int(nshots))
    return f