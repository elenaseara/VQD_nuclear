"""Reference-state selection and ADAPT ansatz construction."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import gc
import re
from typing import Any

import numpy as np
from qibo import gates
from qibo.models import Circuit
from pynucshell.operators import QubitOperator

from .config import ProblemConfig


@dataclass(slots=True)
class ReferenceData:
    """Selected Slater-determinant reference state."""

    occupied_orbitals: list[int]
    diagonal_energy: float
    first_active_operator: int
    first_active_diff: float


def clean_pauli_string(pauli_string: tuple[tuple[int, str], ...]) -> tuple[tuple[int, str], ...]:
    """Convert NumPy integer qubit indices into Python ints."""
    return tuple((int(qubit), str(pauli)) for qubit, pauli in pauli_string)


def build_reference_circuit(n_qubits: int, occupied_orbitals: list[int] | tuple[int, ...]) -> Circuit:
    """Prepare a Slater determinant in the JW computational basis."""
    circuit = Circuit(int(n_qubits))
    for qubit in occupied_orbitals:
        circuit.add(gates.X(int(qubit)))
    return circuit


def basis_state_to_occupied_orbitals(state: Any, *, n_qubits: int, n_valence_particles: int) -> tuple[int, ...]:
    """Extract occupied orbitals from one many-body basis state."""
    if isinstance(state, (list, tuple, np.ndarray)):
        values = list(state)
        if len(values) == n_qubits and set(values).issubset({0, 1, False, True}):
            return tuple(i for i, value in enumerate(values) if int(value) == 1)
        if all(isinstance(x, (int, np.integer)) for x in values):
            return tuple(int(x) for x in values)

    if isinstance(state, (int, np.integer)):
        return tuple(i for i in range(n_qubits) if (int(state) >> i) & 1)

    text = str(state)
    numbers = [int(x) for x in re.findall(r"\d+", text)]
    if len(numbers) == n_qubits and set(numbers).issubset({0, 1}):
        return tuple(i for i, value in enumerate(numbers) if value == 1)
    if len(numbers) == n_valence_particles:
        return tuple(numbers)
    raise ValueError(f"Could not extract occupied orbitals from basis state: {state}")


def get_hamiltonian_basis_references(nucleus: Any, *, n_qubits: int, n_valence_particles: int) -> list[tuple[int, ...]]:
    """Extract all Slater determinants from the Hamiltonian many-body basis."""
    mbasis = nucleus.get_basis().mb()
    entries = mbasis.values() if isinstance(mbasis, dict) else mbasis
    references: list[tuple[int, ...]] = []
    for state in entries:
        occupied = tuple(sorted(set(basis_state_to_occupied_orbitals(
            state,
            n_qubits=n_qubits,
            n_valence_particles=n_valence_particles,
        ))))
        if len(occupied) != n_valence_particles:
            raise ValueError(f"Basis state {state} gives {len(occupied)} particles, expected {n_valence_particles}.")
        if not all(0 <= q < n_qubits for q in occupied):
            raise ValueError(f"Basis state {state} contains orbitals outside [0,{n_qubits - 1}].")
        references.append(occupied)
    references = sorted(set(references))
    if not references:
        raise RuntimeError("No Slater determinants found in nucleus.get_basis().mb().")
    return references


def diagonal_energy(occupied_orbitals: tuple[int, ...], qubit_hamiltonian: Any) -> float:
    """Compute <Phi|H|Phi> for a computational-basis Slater determinant."""
    occupied = set(int(q) for q in occupied_orbitals)
    energy = 0.0
    for pauli_string, coefficient in qubit_hamiltonian.terms.items():
        value = complex(coefficient)
        for qubit, pauli in pauli_string:
            qubit = int(qubit)
            pauli = str(pauli)
            if pauli == "Z":
                value *= -1.0 if qubit in occupied else 1.0
            elif pauli in ("X", "Y"):
                value = 0.0
                break
            elif pauli == "I":
                continue
            else:
                raise ValueError(f"Unknown Pauli operator: {pauli}")
        energy += np.real(value)
    return float(energy)


def prepare_generator_term_pool(qubit_generator_pool: list[Any]) -> list[list[QubitOperator]]:
    """Pre-convert each generator term to one QubitOperator for faster circuit builds."""
    prepared_pool: list[list[QubitOperator]] = []
    for generator in qubit_generator_pool:
        prepared_terms = []
        for pauli_string, coefficient in generator.terms.items():
            if pauli_string == ():
                continue
            prepared_terms.append(QubitOperator(clean_pauli_string(pauli_string), float(np.real_if_close(coefficient))))
        prepared_pool.append(prepared_terms)
    return prepared_pool


class AnsatzBuilder:
    """Build and cache ADAPT ansatz circuits and statevectors."""

    def __init__(self, *, n_qubits: int, backend: Any, prepared_generator_term_pool: list[list[QubitOperator]], reference_occupied_orbitals: list[int], max_state_cache_items: int = 32) -> None:
        self.n_qubits = int(n_qubits)
        self.backend = backend
        self.prepared_generator_term_pool = prepared_generator_term_pool
        self.reference_occupied_orbitals = [int(q) for q in reference_occupied_orbitals]
        self.max_state_cache_items = int(max_state_cache_items)
        self._state_cache: OrderedDict[tuple[tuple[float, ...], tuple[int, ...]], np.ndarray] = OrderedDict()

    @staticmethod
    def cache_key(thetas: list[float] | tuple[float, ...], operator_indices: list[int] | tuple[int, ...], ndigits: int = 12) -> tuple[tuple[float, ...], tuple[int, ...]]:
        return (tuple(round(float(theta), int(ndigits)) for theta in thetas), tuple(int(index) for index in operator_indices))

    def clear_cache(self) -> None:
        self._state_cache.clear()
        gc.collect()

    def _cache_set(self, key: tuple[tuple[float, ...], tuple[int, ...]], value: np.ndarray) -> None:
        self._state_cache[key] = value
        self._state_cache.move_to_end(key)
        while len(self._state_cache) > self.max_state_cache_items:
            self._state_cache.popitem(last=False)

    def build(self, thetas: list[float] | tuple[float, ...], operator_indices: list[int] | tuple[int, ...]) -> Circuit:
        """Build the ADAPT ansatz circuit from the selected reference determinant."""
        circuit = Circuit(self.n_qubits)
        for qubit in self.reference_occupied_orbitals:
            circuit.add(gates.X(int(qubit)))
        for theta, index in zip(thetas, operator_indices):
            for term_operator in self.prepared_generator_term_pool[int(index)]:
                self.backend.exp_pauli_str_staircase(circuit, float(theta), term_operator)
        return circuit

    def state(self, thetas: list[float] | tuple[float, ...], operator_indices: list[int] | tuple[int, ...]) -> np.ndarray:
        """Return a dense simulator statevector for the ansatz."""
        key = self.cache_key(thetas, operator_indices)
        if key in self._state_cache:
            self._state_cache.move_to_end(key)
            return self._state_cache[key]
        try:
            state = self.build(thetas, operator_indices)().state()
        except Exception as exc:
            raise RuntimeError("The active Qibo backend could not return a dense statevector. Use qibo.set_backend('numpy') for this route.") from exc
        self._cache_set(key, state)
        return state


def build_probe_circuit(n_qubits: int, occupied_orbitals: tuple[int, ...], operator_index: int | None, theta: float | None, *, qubit_generator_pool: list[Any], backend: Any) -> Circuit:
    """Prepare the reference determinant and optionally apply one ADAPT generator."""
    circuit = build_reference_circuit(n_qubits, occupied_orbitals)
    if theta is not None and operator_index is not None:
        generator = qubit_generator_pool[int(operator_index)]
        for pauli_string, coefficient in generator.terms.items():
            if pauli_string == ():
                continue
            term_operator = QubitOperator(clean_pauli_string(pauli_string), float(np.real_if_close(coefficient)))
            backend.exp_pauli_str_staircase(circuit, float(theta), term_operator)
    return circuit


def find_first_active_operator(occupied_orbitals: tuple[int, ...], *, n_qubits: int, qubit_generator_pool: list[Any], backend: Any, theta_probe: float = 0.1, tol: float = 1e-8) -> tuple[int | None, float]:
    """Return the first pool operator that changes the reference determinant."""
    reference_state = build_probe_circuit(n_qubits, occupied_orbitals, None, None, qubit_generator_pool=qubit_generator_pool, backend=backend)().state()
    for operator_index in range(len(qubit_generator_pool)):
        test_state = build_probe_circuit(n_qubits, occupied_orbitals, operator_index, theta_probe, qubit_generator_pool=qubit_generator_pool, backend=backend)().state()
        diff = np.linalg.norm(test_state - reference_state)
        if diff > tol:
            return int(operator_index), float(diff)
    return None, 0.0


def find_reference_from_hamiltonian_basis(problem: ProblemConfig, nucleus: Any, *, n_qubits: int, qubit_hamiltonian: Any, qubit_generator_pool: list[Any], backend: Any) -> ReferenceData:
    """Choose the lowest-energy Slater determinant activated by the ADAPT pool."""
    n_valence_particles = int(problem.n_valence_protons + problem.n_valence_neutrons)
    references = get_hamiltonian_basis_references(nucleus, n_qubits=n_qubits, n_valence_particles=n_valence_particles)
    ranked = sorted((diagonal_energy(occ, qubit_hamiltonian), occ) for occ in references)
    for energy, occupied_orbitals in ranked:
        operator_index, diff = find_first_active_operator(
            occupied_orbitals,
            n_qubits=n_qubits,
            qubit_generator_pool=qubit_generator_pool,
            backend=backend,
        )
        if operator_index is not None:
            return ReferenceData(list(occupied_orbitals), float(energy), int(operator_index), float(diff))
    raise RuntimeError("No active reference determinant found in the Hamiltonian basis.")
