"""Ruta muestreada: ejecutar los planes de medida con shots en Qibo.

Modulo del paquete: src/vqd_nuclear/sampling.py
Pseudocodigo: secciones 8 (medir un grupo), 14 (gradiente muestreado) y 16.2
(SWAP destructivo).

QUE HACE
--------
La fase C dice QUE medir: para cada circuito, las puertas del cambio de base,
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


def value_from_frequencies(group, frequencies) -> tuple[float, int]:
    """(media de diagonal_value sobre los shots, numero de shots)."""
    L = len(group.support)
    total = 0.0
    n = 0
    for key, counts in frequencies.items():
        bits = bitstring_to_array(key, L)
        total += group.diagonal_value(bits) * int(counts)
        n += int(counts)
    if n == 0:
        raise RuntimeError("Qibo no devolvio ningun shot.")
    return total / n, n


def sample_group(base_circuit: Circuit, group, nshots: int, *,
                 initial_state: np.ndarray | None = None) -> float:
    """Estimacion de la contribucion de un grupo con `nshots` shots.

    `initial_state` solo se usa en los tests (estado arbitrario sin ansatz).
    """
    if not group.support:          # no hay nada que medir (no deberia ocurrir)
        return 0.0
    circuit = measurement_circuit(base_circuit, group)
    kwargs: dict[str, Any] = {"nshots": int(nshots)}
    if initial_state is not None:
        kwargs["initial_state"] = np.array(initial_state, dtype=complex, copy=True)
    result = circuit(**kwargs)
    valor, _ = value_from_frequencies(group, result.frequencies(binary=True))
    return float(valor)


# =====================================================================
# Un plan completo (secciones 5, 11, 14)
# =====================================================================

def sampled_expectation(plan, base_circuit: Circuit, nshots: int, *,
                        initial_state: np.ndarray | None = None) -> float:
    """<O> estimado con shots: identidad + suma de cada circuito del plan.

    Converge a plan.exact_expectation(psi) cuando nshots -> inf, con error
    tipico plan.exact_standard_error(psi, n, nshots).
    """
    valor = complex(plan.identity).real
    for group in plan.all_groups():
        valor += sample_group(base_circuit, group, nshots, initial_state=initial_state)
    return float(valor)


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
