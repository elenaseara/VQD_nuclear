import importlib.util

import pytest

from vqd_nuclear import FullQuantumLimits, NumericalConfig, ProblemConfig, RunConfig, run_workflow


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        importlib.util.find_spec("pynucshell") is None,
        reason="pynucshell is not installed in this environment",
    ),
]


def test_be6_statevector_regression():
    problem = ProblemConfig(
        shell="p",
        n_valence_protons=2,
        n_valence_neutrons=0,
        total_j=0,
    )

    run = RunConfig(
        n_states=1,
        beta=0.0,
        run_statevector=True,
        run_full_quantum_circuit=False,
        n_workers=1,
        seed=12345,
    )

    numerical = NumericalConfig(
        max_adapt_iter=6,
        grad_tol=1e-5,
        max_candidates=32,
        optimizer_maxiter=1000,
        overlap_shots=10000,
        energy_shots=10000,
        new_theta_tol=1e-5,
        energy_improvement_tol=1e-6,
        gradient_method="analytic",
        measurement_strategy="paper",
        n_jobs=1,
    )

    output = run_workflow(
        problem,
        run,
        numerical,
        FullQuantumLimits(),
        qibo_backend="numpy",
    )

    result = output["statevector_results"][0]

    assert result["energy"] == pytest.approx(-3.048795045453, abs=5e-4)
    assert result["abs_energy_error"] < 5e-4
    assert result["n_layers"] >= 2