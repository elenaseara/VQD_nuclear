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
# SWAP destructivo (seccion 16.2) — portado del legacy sin cambios de logica
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
    total = 0.0
    shots = 0
    for key, counts in result.frequencies(binary=True).items():
        bits = bitstring_to_array(key, 2 * n)
        signo = -1.0 if int(np.sum(bits[:n] * bits[n:]) % 2) else 1.0
        total += signo * int(counts)
        shots += int(counts)
    if shots == 0:
        raise RuntimeError("Qibo no devolvio ningun shot en el SWAP destructivo.")
    return float(total / shots)


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
    if total == 0:
        raise RuntimeError("Qibo did not return any shot in the penalty circuit.")
    m = (n0 - n1) / total
    var = ((n0 + n1) / total - m * m) / (total - 1) if total > 1 else 0.0
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