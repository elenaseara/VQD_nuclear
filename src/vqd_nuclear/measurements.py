"""

measurements.py v2

"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .ansatz import AnsatzBuilder

PauliTerm = tuple[tuple[int, str], ...]

GRADIENT_PHASE = -1j


def apply_pauli_string_to_state(state: np.ndarray, pauli_string: PauliTerm, n_qubits: int) -> np.ndarray:
    """Apply one Pauli string to a dense statevector using Qibo ordering."""
    if pauli_string == ():
        return np.asarray(state, dtype=complex).copy()
    state = np.asarray(state, dtype=complex).ravel()
    transformed = np.zeros_like(state, dtype=complex)
    for basis_index, amplitude in enumerate(state):
        if amplitude == 0:
            continue
        target_index = int(basis_index)
        phase = 1.0 + 0.0j
        for qubit, pauli in pauli_string:
            qubit = int(qubit)
            pauli = str(pauli)
            mask = 1 << (int(n_qubits) - 1 - qubit)
            bit = 1 if (basis_index & mask) else 0
            if pauli == "I":
                continue
            if pauli == "Z":
                phase *= -1.0 if bit else 1.0
            elif pauli == "X":
                target_index ^= mask
            elif pauli == "Y":
                phase *= -1.0j if bit else 1.0j
                target_index ^= mask
            else:
                raise ValueError(f"Unsupported Pauli operator: {pauli}")
        transformed[target_index] += phase * amplitude
    return transformed


def statevector_pauli_expectation(state: np.ndarray, pauli_string: PauliTerm, n_qubits: int) -> complex:
    """Compute <state|P|state> directly using Qibo statevector ordering."""
    state = np.asarray(state, dtype=complex).ravel()
    return np.vdot(state, apply_pauli_string_to_state(state, pauli_string, n_qubits))


def apply_qubit_operator_to_state(operator: Any, state: np.ndarray, n_qubits: int) -> np.ndarray:
    """Apply a QubitOperator-like object to a dense statevector."""
    output = np.zeros_like(np.asarray(state, dtype=complex).ravel(), dtype=complex)
    for pauli_string, coefficient in operator.terms.items():
        output += complex(coefficient) * apply_pauli_string_to_state(state, pauli_string, n_qubits)
    return output


def qubit_operator_expectation_from_state(operator: Any, state: np.ndarray, n_qubits: int) -> float:
    """Compute <state|operator|state> for a pynucshell QubitOperator."""
    expectation = 0.0 + 0.0j
    for pauli_string, coefficient in operator.terms.items():
        expectation += complex(coefficient) * statevector_pauli_expectation(state, pauli_string, n_qubits)
    return float(np.real_if_close(expectation).real)


"""
*
"""
def commutator_energy_gradient(h_psi: np.array, a_psi: np.array) -> float:
    """dE/dtheta_mu a partir de H|psi> y A|psi> ya calculados.

    Identidad:  <psi|[H,A]|psi> = 2i Im<H psi|A psi>
    luego       <psi| i[H,A] |psi> = -2 Im<H psi|A psi>
    """
    return float(2.0 * np.real(GRADIENT_PHASE * np.vdot(h_psi, a_psi)))


class MeasurementEngine:
    """Evaluate energies and overlaps using exact statevectors or sampled circuits."""

    """
    *
    """
    def __init__(self, *, ansatz_builder: "AnsatzBuilder", qubit_hamiltonian: Any,
                 n_qubits: int, available_workers: int, qubit_generator_pool=None, measurement_strategy: str = "paper", model_space=None) -> None:
        self.ansatz_builder = ansatz_builder
        self.qubit_hamiltonian = qubit_hamiltonian
        self.n_qubits = int(n_qubits)
        self.qubit_generator_pool = list(qubit_generator_pool) or []
        self.model_space = model_space

        if measurement_strategy not in {"paper", "qwc"}:
            raise ValueError("measurement_strategy must be 'paper' or 'qwc'.")
        self.measurement_strategy = str(measurement_strategy)

        self._energy_cache: OrderedDict[tuple[tuple[float, ...], tuple[int, ...]], float] = OrderedDict()
        self.max_energy_cache_items = 128

        # planes perezosos: la ruta statevector no construye ninguno
        self._paper_plan_cache = None
        self._qwc_groups_cache = None

        # fase 2: G_mu y su plan, cacheados por indice de candidato
        self._gradient_operators = {}
        self._gradient_plans = {}



    def clear_cache(self) -> None:
        self.ansatz_builder.clear_cache()
        self._energy_cache.clear()


    def _cache_set(self, key: tuple[tuple[float, ...], tuple[int, ...]], value: float) -> None:
        self._energy_cache[key] = value
        self._energy_cache.move_to_end(key)
        while len(self._energy_cache) > self.max_energy_cache_items:
            self._energy_cache.popitem(last=False)

    """
    *
    """
    def statevector_energy(self, thetas: list[float], operator_indices: list[int]) -> float:
        """Evaluate <psi(theta)|H|psi(theta)> exactly from the statevector."""
        key = self.ansatz_builder.cache_key(thetas, operator_indices)
        if key in self._energy_cache:
            self._energy_cache.move_to_end(key)
            return self._energy_cache[key]

        if self.model_space is not None and self.model_space.can_handle(operator_indices):
            # Camino rapido: propagar y medir dentro del espacio del modelo.
            energy = self.model_space.energy(thetas, operator_indices)
        else:
            state = self.ansatz_builder.state(thetas, operator_indices)
            energy = qubit_operator_expectation_from_state(
                self.qubit_hamiltonian, state, self.n_qubits)
  
        self._cache_set(key, energy)
        return energy


    def exact_overlap(self, thetas_i: list[float], ops_i: list[int], thetas_k: list[float], ops_k: list[int]) -> float:
        """Compute |<psi_i|psi_k>|^2 exactly from cached simulator statevectors."""

        if (self.model_space is not None
                and self.model_space.can_handle(ops_i)
                and self.model_space.can_handle(ops_k)):
            return self.model_space.overlap(thetas_i, ops_i, thetas_k, ops_k)

        psi_i = self.ansatz_builder.state(thetas_i, ops_i)
        psi_k = self.ansatz_builder.state(thetas_k, ops_k)
        return float(abs(np.vdot(psi_i, psi_k)) ** 2)


    """
    *
    """
    def statevector_energy_gradient(self, thetas: list[float], operator_indices: list[int],  generator, *,
                                    psi=None, h_psi=None, a_psi=None):

        if psi is None:
            psi = self.ansatz_builder.state(thetas, operator_indices)
        if h_psi is None:
            h_psi = apply_qubit_operator_to_state(self.qubit_hamiltonian, psi, self.n_qubits)
        if a_psi is None:
            a_psi = apply_qubit_operator_to_state(generator, psi, self.n_qubits)
        return commutator_energy_gradient(h_psi, a_psi)



    """
    *
    """
    def statevector_vqd_gradient(self, thetas, operator_indices: list[int], generator, previous_states, betas,
                                 psi=None, h_psi=None, a_psi=None):

        if psi is None:
            psi = self.ansatz_builder.state(thetas, operator_indices)
        if h_psi is None:
            h_psi = apply_qubit_operator_to_state(self.qubit_hamiltonian, psi, self.n_qubits)
        if a_psi is None:
            a_psi = apply_qubit_operator_to_state(generator, psi, self.n_qubits)

        g_energy = self.statevector_energy_gradient(
            thetas, operator_indices, generator, psi=psi, h_psi=h_psi, a_psi=a_psi)

        dpsi = GRADIENT_PHASE * a_psi
        g_penalty = 0.0

        for prev_psi, beta in zip(previous_states, betas):

            g_penalty += float(beta) * 2.0 * float(np.real(
                np.conj(np.vdot(prev_psi, psi)) * np.vdot(prev_psi, dpsi)))

        g_total = g_energy + g_penalty

        return g_energy, g_penalty, g_total





    ####################################################################################
    ################   CODIGO aUXILIAR PARA EVALUAR PESO DE G_PENALTY   ################
    ####################################################################################
   
    # Metodo de la clase Measurementengine
    """
    *
    """
    def scan_candidate_gradients(self, thetas, operator_indices, candidates,
                                previous_ansatze=None, betas=None) -> CandidateGradientScan:
        """Gradientes de TODOS los candidatos con una sola aplicacion de operador
        por candidato.
    
        psi, H|psi> y los estados previos no dependen del candidato: se calculan
        una vez. Sustituye al barrido de select_next_operator.
        """
        previous_ansatze = previous_ansatze or []
        betas = betas or []
        if not candidates:
            raise ValueError("No hay candidatos que barrer.")

        n_pool = len(self.qubit_generator_pool)
        fuera = [int(c) for c in candidates if not 0 <= int(c) < n_pool]
        if fuera:
            raise IndexError(
                f"Candidatos fuera del pool (tamano {n_pool}): {fuera}. "
                "El pool del motor y el de HamiltonianData no coinciden."
            )
    
        psi = self.ansatz_builder.state(thetas, operator_indices)
        h_psi = apply_qubit_operator_to_state(self.qubit_hamiltonian, psi, self.n_qubits)
    
        previous_states = [self.ansatz_builder.state(p["thetas"], p["ops"])
                        for p in previous_ansatze]
        
        amplitudes = [np.vdot(prev, psi) for prev in previous_states]
        max_overlap = max((abs(a) for a in amplitudes), default=0.0)
    
        # una aplicacion de operador por candidato
        g_energy: dict[int, float] = {}
        g_penalty: dict[int, float] = {}
        g_total: dict[int, float] = {}
        for mu in candidates:
            generator = self.qubit_generator_pool[int(mu)]
            a_psi = apply_qubit_operator_to_state(generator, psi, self.n_qubits)

            g_energy[int(mu)], g_penalty[int(mu)], g_total[int(mu)] = self.statevector_vqd_gradient(thetas, operator_indices, 
                                                                                                    generator, previous_states, betas,
                                                                                                    psi=psi, h_psi=h_psi, a_psi=a_psi)
    
        return CandidateGradientScan(
            candidates=[int(c) for c in candidates],
            g_energy=g_energy,
            g_penalty=g_penalty,
            g_total=g_total,
            max_overlap_amplitude=float(max_overlap),
        )

# Clase para guardar los resultados del barrido sobre todos los A_mu
"""
*
"""
@dataclass(slots=True)
class CandidateGradientScan:
    """Gradientes de todos los candidatos en una iteracion ADAPT."""
 
    candidates: list[int]
    g_energy: dict[int, float]
    g_penalty: dict[int, float]
    g_total: dict[int, float]
    max_overlap_amplitude: float      # max_j |<psi_j|psi>|
 