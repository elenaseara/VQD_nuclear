"""Explicit physical-orbital mapping shared by determinants, H and the pool."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pynucshell.operators import FermionicOperator


@dataclass(frozen=True)
class OrbitalMapping:
    global_to_qubit: dict[int, int]
    species: tuple[int, ...]
    twice_m: tuple[int, ...]
    n_protons: int
    n_neutrons: int

    @property
    def n_qubits(self) -> int:
        return len(self.species)

    @classmethod
    def from_nucleus(cls, nucleus: Any) -> "OrbitalMapping":
        space = nucleus.get_space()
        labels = [int(q) for q in space.mapping]
        kinds = [int(s) for s in space.isospin]
        moments = [int(m) for m in space.array_M]
        if len(set(labels)) != len(labels) or not len(labels) == len(kinds) == len(moments):
            raise ValueError("Invalid single-particle orbital mapping.")
        if any(s not in (-1, 1) for s in kinds):
            raise ValueError("Expected proton (+1) and neutron (-1) species.")
        protons, neutrons = int(nucleus.N_protons), int(nucleus.N_neutrons)
        modes = sorted((q, s, m) for q, s, m in zip(labels, kinds, moments)
                       if (s == 1 and protons > 0) or (s == -1 and neutrons > 0))
        if not modes:
            raise ValueError("At least one valence particle is required.")
        return cls({q: i for i, (q, _, _) in enumerate(modes)}, tuple(s for _, s, _ in modes),
                   tuple(m for _, _, m in modes), protons, neutrons)

    def determinant(self, occupied) -> tuple[int, ...]:
        if len(set(occupied)) != len(occupied):
            raise ValueError("A determinant contains repeated orbitals.")
        try:
            mapped = tuple(sorted(self.global_to_qubit[int(q)] for q in occupied))
        except KeyError as exc:
            raise ValueError("Determinant contains an inactive physical orbital.") from exc
        counts = (sum(self.species[q] == 1 for q in mapped), sum(self.species[q] == -1 for q in mapped))
        if counts != (self.n_protons, self.n_neutrons):
            raise ValueError("Determinant does not preserve the proton/neutron sector.")
        return mapped

    def operator_mapping(self, operators) -> dict[int, int]:
        labels = {int(q) for op in operators for term, value in op.terms.items() if value != 0 for q, _ in term}
        choices = [m for m in (self.global_to_qubit, {q: q for q in range(self.n_qubits)}) if labels.issubset(m)]
        if not choices:
            raise ValueError("Operator indices fit neither the global nor the local orbital space.")
        if len(choices) == 2 and any(choices[0][q] != choices[1][q] for q in labels):
            raise ValueError("Ambiguous global/local operator indices; an explicit mapping is required.")
        return dict(choices[0])

    def operator(self, operator, index_mapping: dict[int, int]):
        mapped = FermionicOperator()
        for term, value in operator.terms.items():
            if value == 0:
                continue
            local = tuple((index_mapping[int(q)], int(action)) for q, action in term)
            for species in (1, -1):
                if sum(2 * a - 1 for q, a in local if self.species[q] == species) != 0:
                    raise ValueError("Fermionic operator changes the proton/neutron sector.")
            if sum((2 * a - 1) * self.twice_m[q] for q, a in local) != 0:
                raise ValueError("Fermionic operator changes the M-scheme projection.")
            mapped.terms[local] = value
        return mapped
