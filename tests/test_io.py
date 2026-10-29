import pandas as pd
import pytest

from vqd_nuclear.io import build_energy_matched_validation_dataframe, build_energy_sorted_dataframe


def test_exact_report_does_not_build_sampled_measurement_plan(tmp_path):
    from types import SimpleNamespace
    from vqd_nuclear import FullQuantumLimits, NumericalConfig, ProblemConfig, RunConfig
    from vqd_nuclear.io import write_single_readable_txt_report

    class DeferredMeasurements:
        measurement_strategy = "paper"

        @property
        def measurement_summary(self):
            pytest.fail("Exact reporting constructed a sampled measurement plan")

    context = SimpleNamespace(available_workers=1, measurements=DeferredMeasurements(),
                              reference=SimpleNamespace(occupied_orbitals=[0], diagonal_energy=0., basis_index=0),
                              hamiltonian=SimpleNamespace(n_qubits=24, pool_metadata={}, exact_energies=[0.]))
    output = tmp_path / "report.txt"
    write_single_readable_txt_report(
        output, run_id="exact", context=context, beta_check={}, problem=ProblemConfig(),
        run=RunConfig(n_states=1, run_full_quantum_circuit=False), numerical=NumericalConfig(),
        full_limits=FullQuantumLimits(), summary_df=pd.DataFrame(), history_df=pd.DataFrame(),
        parameters_df=pd.DataFrame(), overlap_df=pd.DataFrame())
    assert "Measurement circuits: n/a (statevector_exact)" in output.read_text(encoding="utf-8")


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
def test_validation_keeps_beta_variants_separate(builder):
    summary = pd.DataFrame([
        {"run_id": "r", "method": "statevector", "cost_mode": "statevector_exact",
         "beta_method": method, "state": state, "exact_energy": energy, "computed_energy": energy + offset}
        for method, offset in [("exact", 0.01), ("approximate", 0.02)]
        for state, energy in enumerate([-5.0, -3.0, -1.0])
    ])
    result = builder(summary)
    assert len(result) == 6
    for _, group in result.groupby("beta_method"):
        assert list(group["matched_exact_state"]) == [0, 1, 2]
