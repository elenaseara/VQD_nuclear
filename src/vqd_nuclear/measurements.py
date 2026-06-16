"""Energy and overlap estimators used by ADAPT-VQD.

The sampled energy route supports two strategies:

* ``paper``: shell-model measurement classes following Perez-Obiol et al.,
  with one computational-basis circuit for diagonal terms and specific basis
  changes for single- and double-hopping terms.
* ``qwc``: generic qubit-wise commuting Pauli grouping.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from qibo import gates
from qibo.models import Circuit

if TYPE_CHECKING:
    from .ansatz import AnsatzBuilder

PauliTerm = tuple[tuple[int, str], ...]


def bitstring_to_array(bitstring: str | tuple[int, ...] | list[int] | np.ndarray | int, length: int) -> np.ndarray:
    """Convert a Qibo measurement key into an array of bits."""
    if isinstance(bitstring, str):
        bits = bitstring.replace(" ", "")
    elif isinstance(bitstring, (tuple, list, np.ndarray)):
        arr = np.array(bitstring, dtype=int).ravel()
        if len(arr) != length:
            raise ValueError(f"Expected {length} bits, got {len(arr)}.")
        return arr
    else:
        bits = format(int(bitstring), f"0{length}b")
    if len(bits) != length:
        raise ValueError(f"Expected {length} bits, got {len(bits)}.")
    return np.array([int(bit) for bit in bits], dtype=int)


def statevector_pauli_expectation(state: np.ndarray, pauli_string: PauliTerm, n_qubits: int) -> complex:
    """Compute <state|P|state> directly using Qibo statevector ordering."""
    state = np.asarray(state, dtype=complex).ravel()
    return np.vdot(state, apply_pauli_string_to_state(state, pauli_string, n_qubits))


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


def apply_qubit_operator_to_state(operator: Any, state: np.ndarray, n_qubits: int) -> np.ndarray:
    """Apply a QubitOperator-like object to a dense statevector."""
    output = np.zeros_like(np.asarray(state, dtype=complex).ravel(), dtype=complex)
    for pauli_string, coefficient in operator.terms.items():
        output += complex(coefficient) * apply_pauli_string_to_state(state, pauli_string, n_qubits)
    return output


def qwc_commutes(term_a: PauliTerm, term_b: PauliTerm) -> bool:
    """Return True if two Pauli strings commute qubit-wise and share one measurement basis."""
    basis = {int(q): str(p) for q, p in term_a}
    for qubit, pauli in term_b:
        previous = basis.get(int(qubit))
        if previous is not None and previous != str(pauli):
            return False
    return True


def group_qubit_wise_commuting_terms(terms: list[PauliTerm]) -> list[list[PauliTerm]]:
    """Greedily group Pauli strings that can be measured in the same tensor-product basis."""
    groups: list[list[PauliTerm]] = []
    for term in terms:
        if term == ():
            continue
        for group in groups:
            if all(qwc_commutes(term, other) for other in group):
                group.append(term)
                break
        else:
            groups.append([term])
    return groups


def group_measurement_basis(group: list[PauliTerm]) -> list[tuple[int, str]]:
    """Return the union basis needed to measure one qubit-wise commuting group."""
    basis: dict[int, str] = {}
    for term in group:
        for qubit, pauli in term:
            qubit = int(qubit)
            pauli = str(pauli)
            previous = basis.get(qubit)
            if previous is not None and previous != pauli:
                raise ValueError("Terms are not qubit-wise commuting.")
            basis[qubit] = pauli
    return sorted(basis.items())


def qubit_operator_expectation_from_state(operator: Any, state: np.ndarray, n_qubits: int) -> float:
    """Compute <state|operator|state> for a pynucshell QubitOperator."""
    expectation = 0.0 + 0.0j
    for pauli_string, coefficient in operator.terms.items():
        expectation += complex(coefficient) * statevector_pauli_expectation(state, pauli_string, n_qubits)
    return float(np.real_if_close(expectation).real)


def non_z_support(term: PauliTerm) -> tuple[int, ...]:
    """Return qubits carrying X or Y in a Pauli string."""
    return tuple(sorted(int(q) for q, p in term if str(p) in {"X", "Y"}))


def term_support(term: PauliTerm) -> tuple[int, ...]:
    """Return all non-identity qubits in a Pauli string."""
    return tuple(sorted(int(q) for q, _p in term))


def local_index_from_bits(bits: np.ndarray) -> int:
    """Convert left-to-right bits into an integer index."""
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return int(value)


def pauli_matrix(pauli: str) -> np.ndarray:
    """Return a single-qubit Pauli matrix."""
    if pauli == "I":
        return np.eye(2, dtype=complex)
    if pauli == "X":
        return np.array([[0, 1], [1, 0]], dtype=complex)
    if pauli == "Y":
        return np.array([[0, -1j], [1j, 0]], dtype=complex)
    if pauli == "Z":
        return np.array([[1, 0], [0, -1]], dtype=complex)
    raise ValueError(f"Unsupported Pauli operator: {pauli}")


def kron_all(matrices: list[np.ndarray]) -> np.ndarray:
    """Kronecker product of a left-to-right list of matrices."""
    result = matrices[0]
    for matrix in matrices[1:]:
        result = np.kron(result, matrix)
    return result


def local_operator_matrix(terms: dict[PauliTerm, complex], support: tuple[int, ...]) -> np.ndarray:
    """Build the matrix of a QubitOperator restricted to a local support."""
    dim = 2 ** len(support)
    out = np.zeros((dim, dim), dtype=complex)
    support_positions = {qubit: pos for pos, qubit in enumerate(support)}
    for pauli_term, coefficient in terms.items():
        by_qubit = {int(q): str(p) for q, p in pauli_term}
        matrices = [pauli_matrix(by_qubit.get(qubit, "I")) for qubit in support]
        out += complex(coefficient) * kron_all(matrices)
    return out


def apply_local_h(state: np.ndarray, position: int, n_local: int) -> np.ndarray:
    """Apply H to a local statevector using left-to-right bit ordering."""
    out = np.zeros_like(state, dtype=complex)
    mask = 1 << (n_local - 1 - position)
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    for index, amp in enumerate(state):
        bit = 1 if index & mask else 0
        base = index & ~mask
        if bit == 0:
            out[base] += inv_sqrt2 * amp
            out[base | mask] += inv_sqrt2 * amp
        else:
            out[base] += inv_sqrt2 * amp
            out[base | mask] -= inv_sqrt2 * amp
    return out


def apply_local_cnot(state: np.ndarray, control: int, target: int, n_local: int) -> np.ndarray:
    """Apply CNOT to a local statevector using left-to-right bit ordering."""
    out = np.zeros_like(state, dtype=complex)
    control_mask = 1 << (n_local - 1 - control)
    target_mask = 1 << (n_local - 1 - target)
    for index, amp in enumerate(state):
        target_index = index ^ target_mask if index & control_mask else index
        out[target_index] += amp
    return out


def local_basis_change_unitary(kind: str, active: tuple[int, ...], support: tuple[int, ...]) -> np.ndarray:
    """Return the local M basis-change unitary for a shell-model term class."""
    n_local = len(support)
    positions = {qubit: pos for pos, qubit in enumerate(support)}
    dim = 2 ** n_local
    unitary = np.zeros((dim, dim), dtype=complex)

    def apply_sequence(vec: np.ndarray) -> np.ndarray:
        out = vec
        if kind == "single_hopping":
            j, k = active
            out = apply_local_cnot(out, positions[k], positions[j], n_local)
            out = apply_local_h(out, positions[k], n_local)
            out = apply_local_cnot(out, positions[k], positions[j], n_local)
            return out
        if kind == "double_hopping":
            i, j, k, l = active
            out = apply_local_cnot(out, positions[i], positions[j], n_local)
            out = apply_local_cnot(out, positions[k], positions[i], n_local)
            out = apply_local_cnot(out, positions[l], positions[k], n_local)
            out = apply_local_h(out, positions[l], n_local)
            out = apply_local_cnot(out, positions[l], positions[k], n_local)
            out = apply_local_cnot(out, positions[k], positions[i], n_local)
            out = apply_local_cnot(out, positions[i], positions[j], n_local)
            return out
        if kind == "diagonal":
            return out
        raise ValueError(f"Unknown paper measurement kind: {kind}")

    for column in range(dim):
        basis = np.zeros(dim, dtype=complex)
        basis[column] = 1.0
        unitary[:, column] = apply_sequence(basis)
    return unitary


@dataclass(slots=True)
class PaperMeasurementGroup:
    """One sampled measurement circuit in the shell-model strategy."""

    kind: str
    active: tuple[int, ...]
    support: tuple[int, ...]
    terms: dict[PauliTerm, complex]
    diagonal_values: np.ndarray


class PaperMeasurementPlan:
    """Measurement classes inspired by the shell-model simulation paper.

    The plan separates Hamiltonian terms into the classes used in the paper:
    diagonal number-density terms, single-hopping terms and double-hopping
    terms. For hopping terms, it applies the M_jk or M_ijkl basis-change
    circuits and evaluates the diagonalized operator from measured bitstrings.
    If a term cannot be diagonalized by these nuclear-shell-model basis changes,
    it is kept as a residual QWC Pauli group to preserve correctness.
    """

    def __init__(self, qubit_hamiltonian: Any, *, diagonalization_tol: float = 1e-8) -> None:
        self.identity = complex(qubit_hamiltonian.terms.get((), 0.0))
        self.groups: list[PaperMeasurementGroup] = []
        self.residual_terms: dict[PauliTerm, complex] = {}
        self.diagonalization_tol = float(diagonalization_tol)
        self._build(qubit_hamiltonian)
        self.residual_qwc_groups = group_qubit_wise_commuting_terms(list(self.residual_terms.keys()))

    @staticmethod
    def classify(term: PauliTerm) -> tuple[str, tuple[int, ...]]:
        active = non_z_support(term)
        if len(active) == 0:
            return "diagonal", active
        if len(active) == 2:
            return "single_hopping", active
        if len(active) == 4:
            return "double_hopping", active
        return "residual", active

    def _try_make_group(self, kind: str, active: tuple[int, ...], terms: dict[PauliTerm, complex]) -> PaperMeasurementGroup | None:
        support = tuple(sorted(set().union(*(set(term_support(term)) for term in terms)))) if terms else active
        if not support:
            return None
        if kind == "diagonal":
            operator = local_operator_matrix(terms, support)
            diagonalized = operator
        else:
            operator = local_operator_matrix(terms, support)
            unitary = local_basis_change_unitary(kind, active, support)
            diagonalized = unitary.conj().T @ operator @ unitary
        off_diagonal = diagonalized - np.diag(np.diag(diagonalized))
        if np.linalg.norm(off_diagonal) > self.diagonalization_tol:
            return None
        diagonal_values = np.real_if_close(np.diag(diagonalized), tol=1000).real.astype(float)
        return PaperMeasurementGroup(kind=kind, active=active, support=support, terms=terms, diagonal_values=diagonal_values)

    def _build(self, qubit_hamiltonian: Any) -> None:
        buckets: dict[tuple[str, tuple[int, ...]], dict[PauliTerm, complex]] = {}
        for pauli_term, coefficient in qubit_hamiltonian.terms.items():
            if pauli_term == ():
                continue
            kind, active = self.classify(pauli_term)
            if kind == "residual":
                self.residual_terms[pauli_term] = complex(coefficient)
                continue
            buckets.setdefault((kind, active), {})[pauli_term] = complex(coefficient)
        for (kind, active), terms in buckets.items():
            group = self._try_make_group(kind, active, terms)
            if group is None:
                self.residual_terms.update(terms)
            else:
                self.groups.append(group)

    def summary(self) -> dict[str, int]:
        """Return a compact count of measurement circuits by class."""
        out = {"diagonal": 0, "single_hopping": 0, "double_hopping": 0, "residual_qwc": len(self.residual_qwc_groups)}
        for group in self.groups:
            out[group.kind] = out.get(group.kind, 0) + 1
        out["total"] = sum(out.values())
        return out


def add_paper_basis_change(circuit: Circuit, group: PaperMeasurementGroup) -> None:
    """Append the paper basis-change circuit for one measurement group."""
    if group.kind == "single_hopping":
        j, k = group.active
        circuit.add(gates.CNOT(k, j))
        circuit.add(gates.H(k))
        circuit.add(gates.CNOT(k, j))
    elif group.kind == "double_hopping":
        i, j, k, l = group.active
        circuit.add(gates.CNOT(i, j))
        circuit.add(gates.CNOT(k, i))
        circuit.add(gates.CNOT(l, k))
        circuit.add(gates.H(l))
        circuit.add(gates.CNOT(l, k))
        circuit.add(gates.CNOT(k, i))
        circuit.add(gates.CNOT(i, j))
    elif group.kind == "diagonal":
        return
    else:
        raise ValueError(f"Unsupported paper measurement kind: {group.kind}")


class MeasurementEngine:
    """Evaluate energies and overlaps using exact statevectors or sampled circuits."""

    def __init__(self, *, ansatz_builder: "AnsatzBuilder", qubit_hamiltonian: Any, n_qubits: int, available_workers: int, measurement_strategy: str = "paper") -> None:
        self.ansatz_builder = ansatz_builder
        self.qubit_hamiltonian = qubit_hamiltonian
        self.n_qubits = int(n_qubits)
        self.available_workers = int(available_workers)
        self.measurement_strategy = str(measurement_strategy)
        if self.measurement_strategy not in {"paper", "qwc"}:
            raise ValueError("measurement_strategy must be 'paper' or 'qwc'.")
        self._energy_cache: OrderedDict[tuple[tuple[float, ...], tuple[int, ...]], float] = OrderedDict()
        self._qwc_groups = group_qubit_wise_commuting_terms(list(self.qubit_hamiltonian.terms.keys()))
        self._paper_plan = PaperMeasurementPlan(self.qubit_hamiltonian)
        self.max_energy_cache_items = 128

    @property
    def measurement_summary(self) -> dict[str, int]:
        """Return measurement circuit counts for the active strategy."""
        if self.measurement_strategy == "paper":
            return self._paper_plan.summary()
        return {"qwc": len(self._qwc_groups), "total": len(self._qwc_groups)}

    def clear_cache(self) -> None:
        self.ansatz_builder.clear_cache()
        self._energy_cache.clear()

    def _cache_set(self, key: tuple[tuple[float, ...], tuple[int, ...]], value: float) -> None:
        self._energy_cache[key] = value
        self._energy_cache.move_to_end(key)
        while len(self._energy_cache) > self.max_energy_cache_items:
            self._energy_cache.popitem(last=False)

    def statevector_energy(self, thetas: list[float], operator_indices: list[int]) -> float:
        """Evaluate <psi(theta)|H|psi(theta)> exactly from the statevector."""
        key = self.ansatz_builder.cache_key(thetas, operator_indices)
        if key in self._energy_cache:
            self._energy_cache.move_to_end(key)
            return self._energy_cache[key]
        state = self.ansatz_builder.state(thetas, operator_indices)
        energy = qubit_operator_expectation_from_state(self.qubit_hamiltonian, state, self.n_qubits)
        self._cache_set(key, energy)
        return energy

    def exact_overlap(self, thetas_i: list[float], ops_i: list[int], thetas_k: list[float], ops_k: list[int]) -> float:
        """Compute |<psi_i|psi_k>|^2 exactly from cached simulator statevectors."""
        psi_i = self.ansatz_builder.state(thetas_i, ops_i)
        psi_k = self.ansatz_builder.state(thetas_k, ops_k)
        return float(abs(np.vdot(psi_i, psi_k)) ** 2)

    def sampled_pauli_expectation(self, thetas: list[float], operator_indices: list[int], pauli_term: PauliTerm, nshots: int) -> float:
        """Estimate one Pauli-string expectation value with measurement shots."""
        if pauli_term == ():
            return 1.0
        term_circuit = self.ansatz_builder.build(thetas, operator_indices).copy(deep=True)
        measured_qubits: list[int] = []
        for qubit, pauli in pauli_term:
            q = int(qubit)
            measured_qubits.append(q)
            if pauli == "X":
                term_circuit.add(gates.H(q))
            elif pauli == "Y":
                term_circuit.add(gates.SDG(q))
                term_circuit.add(gates.H(q))
            elif pauli == "Z":
                pass
            else:
                raise ValueError(f"Unsupported Pauli operator: {pauli}")
        term_circuit.add(gates.M(*measured_qubits))
        result = term_circuit(nshots=int(nshots))
        expectation = 0.0
        total_counts = 0
        for bitstring, counts in result.frequencies(binary=True).items():
            bits = bitstring_to_array(bitstring, length=len(measured_qubits))
            sign = 1.0 if int(np.sum(bits) % 2) == 0 else -1.0
            expectation += sign * int(counts)
            total_counts += int(counts)
        if total_counts == 0:
            raise RuntimeError("No measurement shots returned for Pauli expectation.")
        return float(expectation / total_counts)

    def sampled_group_expectations(self, thetas: list[float], operator_indices: list[int], group: list[PauliTerm], nshots: int) -> dict[PauliTerm, float]:
        """Estimate all Pauli-string expectations in one qubit-wise commuting group."""
        basis = group_measurement_basis(group)
        measured_qubits = [qubit for qubit, _pauli in basis]
        positions = {qubit: pos for pos, qubit in enumerate(measured_qubits)}
        circuit = self.ansatz_builder.build(thetas, operator_indices).copy(deep=True)
        for qubit, pauli in basis:
            if pauli == "X":
                circuit.add(gates.H(qubit))
            elif pauli == "Y":
                circuit.add(gates.SDG(qubit))
                circuit.add(gates.H(qubit))
            elif pauli == "Z":
                pass
            else:
                raise ValueError(f"Unsupported Pauli operator: {pauli}")
        circuit.add(gates.M(*measured_qubits))
        result = circuit(nshots=int(nshots))
        expectations = {term: 0.0 for term in group}
        total_counts = 0
        for bitstring, counts in result.frequencies(binary=True).items():
            bits = bitstring_to_array(bitstring, length=len(measured_qubits))
            for term in group:
                parity = 0
                for qubit, _pauli in term:
                    parity += int(bits[positions[int(qubit)]])
                expectations[term] += (1.0 if parity % 2 == 0 else -1.0) * int(counts)
            total_counts += int(counts)
        if total_counts == 0:
            raise RuntimeError("No measurement shots returned for grouped Pauli expectation.")
        return {term: value / total_counts for term, value in expectations.items()}

    def sampled_qwc_energy(self, thetas: list[float], operator_indices: list[int], *, nshots: int) -> float:
        """Estimate <psi(theta)|H|psi(theta)> using grouped QWC Pauli circuits."""
        value = 0.0
        identity = self.qubit_hamiltonian.terms.get((), 0.0)
        value += float(np.real_if_close(identity))
        for group in self._qwc_groups:
            expectations = self.sampled_group_expectations(thetas, operator_indices, group, int(nshots))
            for pauli_term, expectation in expectations.items():
                coefficient = self.qubit_hamiltonian.terms[pauli_term]
                value += float(np.real_if_close(coefficient)) * float(expectation)
        return float(np.real(value))

    def sampled_paper_group_value(self, thetas: list[float], operator_indices: list[int], group: PaperMeasurementGroup, nshots: int) -> float:
        """Estimate one shell-model measurement group after its basis-change circuit."""
        measured_qubits = list(group.support)
        circuit = self.ansatz_builder.build(thetas, operator_indices).copy(deep=True)
        add_paper_basis_change(circuit, group)
        circuit.add(gates.M(*measured_qubits))
        result = circuit(nshots=int(nshots))
        total = 0.0
        total_counts = 0
        for bitstring, counts in result.frequencies(binary=True).items():
            bits = bitstring_to_array(bitstring, length=len(measured_qubits))
            local_index = local_index_from_bits(bits)
            total += float(group.diagonal_values[local_index]) * int(counts)
            total_counts += int(counts)
        if total_counts == 0:
            raise RuntimeError("No measurement shots returned for paper measurement group.")
        return float(total / total_counts)

    def sampled_paper_energy(self, thetas: list[float], operator_indices: list[int], *, nshots: int) -> float:
        """Estimate energy using shell-model basis-change measurement circuits."""
        value = float(np.real_if_close(self._paper_plan.identity).real)
        for group in self._paper_plan.groups:
            value += self.sampled_paper_group_value(thetas, operator_indices, group, int(nshots))
        for group in self._paper_plan.residual_qwc_groups:
            expectations = self.sampled_group_expectations(thetas, operator_indices, group, int(nshots))
            for pauli_term, expectation in expectations.items():
                value += float(np.real_if_close(self._paper_plan.residual_terms[pauli_term]).real) * float(expectation)
        return float(value)

    def sampled_energy(self, thetas: list[float], operator_indices: list[int], *, nshots: int, n_jobs: int | str | None) -> float:
        """Estimate <psi(theta)|H|psi(theta)> using the configured sampled measurement route."""
        del n_jobs
        if self.measurement_strategy == "paper":
            return self.sampled_paper_energy(thetas, operator_indices, nshots=nshots)
        return self.sampled_qwc_energy(thetas, operator_indices, nshots=nshots)

    def analytic_energy_gradient(
        self,
        thetas,
        operator_indices,
        candidate_generator,
        previous_ansatze=None,
        betas=None,
    ):
        """Analytic ADAPT gradient of the Hamiltonian energy only.

        This is the standard ADAPT-VQE commutator gradient
        ``d <H> / d theta_mu`` evaluated at the candidate parameter
        ``theta_mu = 0``. The optional VQD arguments are accepted for
        backward compatibility but are intentionally ignored here. Use
        :meth:`analytic_vqd_gradient` when the deflated VQD objective is
        required.
        """
        del previous_ansatze, betas
        psi = self.ansatz_builder.state(thetas, operator_indices)

        h_psi = apply_qubit_operator_to_state(
            self.qubit_hamiltonian,
            psi,
            self.n_qubits,
        )

        g_psi = apply_qubit_operator_to_state(
            candidate_generator,
            psi,
            self.n_qubits,
        )

        hg_psi = apply_qubit_operator_to_state(
            self.qubit_hamiltonian,
            g_psi,
            self.n_qubits,
        )

        gh_psi = apply_qubit_operator_to_state(
            candidate_generator,
            h_psi,
            self.n_qubits,
        )

        gradient = -1j * np.vdot(psi, hg_psi - gh_psi)

        return float(np.real_if_close(gradient).real)

    def analytic_vqd_gradient(
        self,
        thetas,
        operator_indices,
        candidate_generator,
        previous_ansatze=None,
        betas=None,
    ):
        """Analytic ADAPT gradient of the full VQD objective.

        For a candidate generator ``G`` appended with parameter ``theta_mu``
        and evaluated at ``theta_mu = 0``, the state derivative used by the
        ansatz convention is ``|dpsi> = -i G |psi>``. Therefore this method
        computes

            d/dtheta_mu [ <H> + sum_i beta_i |<psi_i|psi>|^2 ]

        as the standard commutator contribution plus the analytic derivative
        of every overlap-penalty term. This is the gradient described by the
        VQD objective in the thesis/paper and should be used for excited-state
        ADAPT selection.
        """
        previous_ansatze = previous_ansatze or []
        betas = betas or []

        psi = self.ansatz_builder.state(thetas, operator_indices)

        h_psi = apply_qubit_operator_to_state(
            self.qubit_hamiltonian,
            psi,
            self.n_qubits,
        )

        g_psi = apply_qubit_operator_to_state(
            candidate_generator,
            psi,
            self.n_qubits,
        )

        hg_psi = apply_qubit_operator_to_state(
            self.qubit_hamiltonian,
            g_psi,
            self.n_qubits,
        )

        gh_psi = apply_qubit_operator_to_state(
            candidate_generator,
            h_psi,
            self.n_qubits,
        )

        energy_gradient = -1j * np.vdot(psi, hg_psi - gh_psi)

        penalty_gradient = 0.0
        if previous_ansatze and betas:
            dpsi = -1j * g_psi
            for previous, beta in zip(previous_ansatze, betas):
                previous_state = self.ansatz_builder.state(previous["thetas"], previous["ops"])
                overlap_amplitude = np.vdot(previous_state, psi)
                overlap_derivative = np.vdot(previous_state, dpsi)
                penalty_gradient += float(beta) * 2.0 * float(np.real(np.conj(overlap_amplitude) * overlap_derivative))

        gradient = energy_gradient + penalty_gradient
        return float(np.real_if_close(gradient).real)

    @staticmethod
    def copy_gate_with_offset(gate: Any, offset: int) -> Any:
        """Copy a Qibo gate while shifting its qubit indices."""
        shifted_qubits = [int(q) + int(offset) for q in gate.qubits]
        kwargs = getattr(gate, "init_kwargs", {}) or {}
        try:
            return gate.__class__(*shifted_qubits, **kwargs)
        except TypeError:
            params = getattr(gate, "parameters", ())
            return gate.__class__(*shifted_qubits, *params, **kwargs)

    def add_ansatz_register(self, target_circuit: Circuit, thetas: list[float], operator_indices: list[int], offset: int) -> None:
        """Append one ansatz circuit to a selected register of a larger circuit."""
        ansatz = self.ansatz_builder.build(thetas, operator_indices)
        for gate in ansatz.queue:
            target_circuit.add(self.copy_gate_with_offset(gate, offset))

    def destructive_swap_circuit(self, thetas_i: list[float], ops_i: list[int], thetas_k: list[float], ops_k: list[int]) -> Circuit:
        """Build the destructive-SWAP circuit for |<psi_i|psi_k>|^2."""
        circuit = Circuit(2 * self.n_qubits)
        self.add_ansatz_register(circuit, thetas_i, ops_i, offset=0)
        self.add_ansatz_register(circuit, thetas_k, ops_k, offset=self.n_qubits)
        for q in range(self.n_qubits):
            circuit.add(gates.CNOT(q, q + self.n_qubits))
            circuit.add(gates.H(q))
        circuit.add(gates.M(*range(2 * self.n_qubits)))
        return circuit

    def destructive_swap_overlap(self, thetas_i: list[float], ops_i: list[int], thetas_k: list[float], ops_k: list[int], *, nshots: int) -> float:
        """Estimate |<psi_i|psi_k>|^2 with the destructive-SWAP circuit."""
        result = self.destructive_swap_circuit(thetas_i, ops_i, thetas_k, ops_k)(nshots=int(nshots))
        overlap = 0.0
        total_shots = 0
        for bitstring, counts in result.frequencies(binary=True).items():
            bits = bitstring_to_array(bitstring, length=2 * self.n_qubits)
            first_register = bits[: self.n_qubits]
            second_register = bits[self.n_qubits :]
            sign = 1.0 if int(np.sum(first_register * second_register) % 2) == 0 else -1.0
            overlap += sign * int(counts)
            total_shots += int(counts)
        if total_shots == 0:
            raise RuntimeError("No measurement shots returned for destructive SWAP.")
        return float(overlap / total_shots)
