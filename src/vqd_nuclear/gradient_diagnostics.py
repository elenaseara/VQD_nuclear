"""
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# registro

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

    winner_energy_only: int                 # lo que elegiria la ruta full quantum
    winner_full_vqd: int                    # lo que se elige realmente
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

    No calcula nada cuantico: solo compara los numeros que ya vienen dados.
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


#  informes

def summarize_diagnostics(records: list[GradientDiagnostics]) -> str:
    """Informe de texto con el desglose POR ITERACION.

    Una fraccion agregada ('coincidieron el 90% de las veces') puede ocultar
    que las discrepancias esten concentradas en la iteracion 0, que es el caso
    peligroso: ADAPT es voraz y una primera eleccion mala condiciona el resto
    del ansatz. Con la politica de consumir operadores, ademas gasta uno de
    los pocos cupos disponibles.
    """
    if not records:
        return "Sin registros."

    lineas: list[str] = []
    cambios = sum(1 for r in records if r.ranking_changed)
    lineas.append("=" * 76)
    lineas.append("  INFLUENCIA DE g^penalty SOBRE LA SELECCION ADAPT")
    lineas.append("=" * 76)
    lineas.append(f"  iteraciones registradas : {len(records)}")
    lineas.append(f"  el ranking cambia en    : {cambios} ({100*cambios/len(records):.0f}%)")
    lineas.append("")

    por_estado: dict[int, list[GradientDiagnostics]] = {}
    for r in records:
        por_estado.setdefault(r.state_index, []).append(r)

    for estado in sorted(por_estado):
        filas = sorted(por_estado[estado], key=lambda r: r.iteration)
        lineas.append(f"  Estado {estado}")
        lineas.append(f"    {'iter':>4} {'win_E':>6} {'win_T':>6} {'cambia':>7} "
                      f"{'regret':>9} {'|<psi_j|psi>|':>14} {'max ratio':>11}")
        for r in filas:
            marca = "SI" if r.ranking_changed else "-"
            lineas.append(f"    {r.iteration:>4} {r.winner_energy_only:>6} "
                          f"{r.winner_full_vqd:>6} {marca:>7} "
                          f"{r.regret:>9.3f} {r.max_overlap_amplitude:>14.3e} "
                          f"{r.max_ratio:>11.3e}")
        if any(r.iteration == 0 and r.ranking_changed for r in filas):
            lineas.append("    AVISO: el ranking cambia en la ITERACION 0. ADAPT es voraz y")
            lineas.append("           esa eleccion condiciona todo el ansatz posterior.")
        lineas.append("")

    lineas.append("  Lectura:")
    lineas.append("    regret = 1 - |gT[ganador por energia]| / |gT[ganador correcto]|")
    lineas.append("      0.00  -> mismo operador, o empate practico: la aproximacion no cuesta nada")
    lineas.append("      ->1   -> energia sola elige un operador muy inferior en el objetivo real")
    lineas.append("    |<psi_j|psi>| es la AMPLITUD de solapamiento, no su cuadrado: es la")
    lineas.append("    variable que explica cuando pesa la penalizacion (grande al empezar")
    lineas.append("    cada estado excitado, pequena al converger).")
    lineas.append("    Un candidato con g^E = 0 y g^penalty != 0 es posible y legitimo:")
    lineas.append("    conmuta con H (no cambia la energia) pero aleja de los estados previos.")
    return "\n".join(lineas)


def diagnostics_dataframe(records: list[GradientDiagnostics]):
    """DataFrame con una fila por iteracion, para guardar junto al resto de salidas."""
    import pandas as pd
    return pd.DataFrame([r.as_row() for r in records])