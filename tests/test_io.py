import pandas as pd
import pytest

from vqd_nuclear.io import build_energy_matched_validation_dataframe, build_energy_sorted_dataframe


def test_energy_sorted_summary_detects_permuted_states():
    summary = pd.DataFrame([
        {"run_id": "r", "method": "statevector", "cost_mode": "statevector_exact", "state": 0, "exact_energy": -5.0, "computed_energy": -3.0, "operators": "[0]", "thetas": "[0.1]"},
        {"run_id": "r", "method": "statevector", "cost_mode": "statevector_exact", "state": 1, "exact_energy": -3.0, "computed_energy": -1.0, "operators": "[1]", "thetas": "[0.2]"},
        {"run_id": "r", "method": "statevector", "cost_mode": "statevector_exact", "state": 2, "exact_energy": -1.0, "computed_energy": -5.0, "operators": "[2]", "thetas": "[0.3]"},
    ])

    sorted_summary = build_energy_sorted_dataframe(summary)

    assert list(sorted_summary["computed_state"]) == [2, 0, 1]
    assert list(sorted_summary["matched_exact_state"]) == [0, 1, 2]
    assert sorted_summary.loc[0, "abs_error_sorted"] == pytest.approx(0.0)


def test_matched_validation_summary_uses_optimal_energy_assignment():
    summary = pd.DataFrame([
        {"run_id": "r", "method": "statevector", "cost_mode": "statevector_exact", "state": 0, "exact_energy": -5.0, "computed_energy": -3.02, "n_layers": 1, "n_iterations": 1, "operators": "[0]", "thetas": "[0.1]"},
        {"run_id": "r", "method": "statevector", "cost_mode": "statevector_exact", "state": 1, "exact_energy": -3.0, "computed_energy": -1.01, "n_layers": 1, "n_iterations": 1, "operators": "[1]", "thetas": "[0.2]"},
        {"run_id": "r", "method": "statevector", "cost_mode": "statevector_exact", "state": 2, "exact_energy": -1.0, "computed_energy": -5.03, "n_layers": 1, "n_iterations": 1, "operators": "[2]", "thetas": "[0.3]"},
    ])

    validation = build_energy_matched_validation_dataframe(summary)

    assert list(validation["computed_state"]) == [2, 0, 1]
    assert list(validation["matched_exact_state"]) == [0, 1, 2]
    assert validation.loc[0, "matched_abs_error"] == pytest.approx(0.03)
    assert validation.loc[0, "raw_index_abs_error"] == pytest.approx(4.03)
    assert bool(validation.loc[0, "state_label_changed"]) is True


@pytest.mark.parametrize("builder", [build_energy_sorted_dataframe, build_energy_matched_validation_dataframe])
def test_energy_validation_keeps_beta_families_separate(builder):
    summary = pd.DataFrame([
        {"run_id": "r", "method": "statevector", "cost_mode": "statevector_exact",
         "beta_method": beta_method, "state": state, "exact_energy": energy,
         "computed_energy": energy + offset}
        for beta_method, offset in [("exact", 0.01), ("approximate", 0.02)]
        for state, energy in enumerate([-5.0, -3.0, -1.0])
    ])
    result = builder(summary)
    assert len(result) == 6
    for _, family in result.groupby("beta_method"):
        assert list(family["matched_exact_state"]) == [0, 1, 2]
