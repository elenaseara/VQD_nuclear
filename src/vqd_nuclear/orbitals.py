"""One physical-orbital convention for determinants and fermionic operators."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from pynucshell.operators import FermionicOperator


@dataclass(frozen=True)
class OrbitalMapping:
    global_to_qubit: dict[int, int]
    qubit_species: tuple[int, ...]
    n_protons: int
    n_neutrons: int

    @property
    def n_qubits(self) -> int:
        return len(self.qubit_species)

    @classmethod
    def from_nucleus(cls, nucleus: Any) -> "OrbitalMapping":
        """Keep all modes of each populated species, in global-label order."""
        space = nucleus.get_space()
        species = tuple(int(value) for value in space.isospin)
        labels = tuple(int(value) for value in space.mapping)
        if len(labels) != len(species) or len(set(labels)) != len(labels):
            raise ValueError("Invalid single-particle space mapping.")
        if any(value not in (-1, 1) for value in species):
            raise ValueError("Expected proton (+1) and neutron (-1) orbital species.")
        n_protons, n_neutrons = int(nucleus.N_protons), int(nucleus.N_neutrons)
        active = sorted(
            (label, kind) for label, kind in zip(labels, species)
            if (kind == 1 and n_protons > 0) or (kind == -1 and n_neutrons > 0)
        )
        if not active:
            raise ValueError("At least one valence particle is required for ADAPT-VQD.")
        return cls({label: q for q, (label, _) in enumerate(active)},
                   tuple(kind for _, kind in active), n_protons, n_neutrons)

    def determinant(self, orbitals: tuple[int, ...]) -> tuple[int, ...]:
        if len(set(orbitals)) != len(orbitals):
            raise ValueError("A determinant contains repeated orbitals.")
        try:
            occupied = tuple(sorted(self.global_to_qubit[int(label)] for label in orbitals))
        except KeyError as exc:
            raise ValueError(f"Orbital {exc.args[0]} is outside the active physical space.") from exc
        protons = sum(self.qubit_species[q] == 1 for q in occupied)
        neutrons = sum(self.qubit_species[q] == -1 for q in occupied)
        if (protons, neutrons) != (self.n_protons, self.n_neutrons):
            raise ValueError("Determinant does not preserve the requested proton/neutron counts.")
        return occupied

    def operator_to_qubit(self, operators: Iterable[Any]) -> dict[int, int]:
        """Resolve one collection's global or already-local fermionic labels.

        Check every used label against both complete mode spaces. Resolve the
        Hamiltonian and pool independently; never infer a convention from only
        the largest label or from an individual sparse pool element. If both
        conventions fit but map a used label differently, require an explicit
        map rather than silently changing the physical orbital.
        """
        labels = {int(label) for operator in operators
                  for term, coefficient in operator.terms.items() if coefficient != 0
                  for label, _ in term}
        candidates = [mapping for mapping in (self.global_to_qubit, {q: q for q in range(self.n_qubits)})
                      if labels.issubset(mapping)]
        if not candidates:
            raise ValueError("Fermionic indices fit neither the global nor the local active orbital space.")
        if len(candidates) == 2 and any(candidates[0][label] != candidates[1][label] for label in labels):
            raise ValueError("Ambiguous fermionic orbital indices; an explicit operator index mapping is required.")
        return dict(candidates[0])

    def fermionic_operator(self, operator: Any, *, index_mapping: dict[int, int] | None = None) -> FermionicOperator:
        """Relabel before JW so that parity strings use the same active modes."""
        source_to_qubit = self.global_to_qubit if index_mapping is None else index_mapping
        mapped = FermionicOperator()
        for term, coefficient in operator.terms.items():
            if coefficient == 0:
                continue
            try:
                local_term = tuple((source_to_qubit[int(label)], int(action)) for label, action in term)
            except KeyError as exc:
                raise ValueError(f"Operator uses inactive orbital {exc.args[0]}.") from exc
            if any(not 0 <= q < self.n_qubits for q, _ in local_term):
                raise ValueError("Operator index mapping lies outside the qubit register.")
            for species in (1, -1):
                change = sum(2 * action - 1 for q, action in local_term if self.qubit_species[q] == species)
                if change != 0:
                    raise ValueError("Operator changes the proton or neutron number.")
            mapped.terms[local_term] = coefficient
        return mapped
