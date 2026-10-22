"""
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any



@dataclass(slots=True)
class GradientDiagnostics:
    """Comparacion, en una iteracion ADAPT, entre seleccion por g^E y por g^total."""

    state_index: int
    iteration: int
    candidates: list[int]

    g_energy: dict[int, float]
    g_penalty: dict[int, float]
    g_total: dict[int, float]
    ratio: dict[int, float]                 # |gP| / (|gE| + eps)

    winner_energy_only: int                 # lo que elegiria energy_only
    winner_full_vqd: int                    # lo que elige vqd
    ranking_changed: bool

    gt_at_energy_winner: float
    gt_at_vqd_winner: float
    regret: float                           # 1 - gt_energy/gt_vqd, en [0,1]

    max_overlap_amplitude: float            # explica CUANDO pesa la penalizacion
    max_ratio: float

    @property
    def selected(self) -> int:
        """El operador que ADAPT debe anadir (criterio correcto)."""
        return self.winner_full_vqd

    @property
    def selected_gradient(self) -> float:
        return self.g_total[self.winner_full_vqd]

    def as_row(self) -> dict[str, Any]:
        """Fila plana para volcar a CSV."""
        return {
            "state": self.state_index,
            "iteration": self.iteration,
            "n_candidates": len(self.candidates),
            "winner_energy_only": self.winner_energy_only,
            "winner_full_vqd": self.winner_full_vqd,
            "ranking_changed": self.ranking_changed,
            "gt_at_energy_winner": self.gt_at_energy_winner,
            "gt_at_vqd_winner": self.gt_at_vqd_winner,
            "regret": self.regret,
            "max_overlap_amplitude": self.max_overlap_amplitude,
            "max_ratio": self.max_ratio,
            "g_energy": ";".join(f"{self.g_energy[c]:.6e}" for c in self.candidates),
            "g_penalty": ";".join(f"{self.g_penalty[c]:.6e}" for c in self.candidates),
            "g_total": ";".join(f"{self.g_total[c]:.6e}" for c in self.candidates),
        }


# diagnosis
def build_diagnostics(scan, *, state_index: int = -1, iteration: int = -1,
                      epsilon: float = 1e-12) -> GradientDiagnostics:
    """Convierte un CandidateGradientScan en la comparacion de criterios.

    No calcula nada: solo compara los numeros que ya vienen dados.
    """
    candidates = list(scan.candidates)
    if not candidates:
        raise ValueError("El barrido no contiene candidatos.")

    ratio = {
        mu: abs(scan.g_penalty[mu]) / (abs(scan.g_energy[mu]) + epsilon)
        for mu in candidates
    }

    winner_e = max(candidates, key=lambda m: abs(scan.g_energy[m]))
    winner_t = max(candidates, key=lambda m: abs(scan.g_total[m]))

    gt_e = abs(scan.g_total[winner_e])
    gt_t = abs(scan.g_total[winner_t])
    regret = 0.0 if gt_t <= epsilon else float(1.0 - gt_e / gt_t)

    return GradientDiagnostics(
        state_index=int(state_index),
        iteration=int(iteration),
        candidates=candidates,
        g_energy=dict(scan.g_energy),
        g_penalty=dict(scan.g_penalty),
        g_total=dict(scan.g_total),
        ratio=ratio,
        winner_energy_only=int(winner_e),
        winner_full_vqd=int(winner_t),
        ranking_changed=bool(winner_e != winner_t),
        gt_at_energy_winner=gt_e,
        gt_at_vqd_winner=gt_t,
        regret=regret,
        max_overlap_amplitude=float(scan.max_overlap_amplitude),
        max_ratio=float(max(ratio.values())),
    )


def diagnostics_dataframe(records: list[GradientDiagnostics]):
    """DataFrame con una fila por iteracion, para guardar junto al resto de salidas."""
    import pandas as pd
    return pd.DataFrame([r.as_row() for r in records])